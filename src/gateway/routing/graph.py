"""Router agent plus a verifier that can send the prompt back to a stronger model.

LangGraph owns the cycle: classify, select, generate, verify, and select again
when the draft is too weak. Transport failures (429, 5xx, timeouts) stay in
the executor. This loop is only about answer quality.
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from gateway.cache.embeddings import Embedder
from gateway.catalog import label_tier
from gateway.errors import ProviderError
from gateway.providers.registry import Registry
from gateway.reliability.executor import Executor
from gateway.routing.classifier import classify_text
from gateway.routing.features import prompt_tier
from gateway.routing.policies import choose_model
from gateway.routing.verifier import verify_answer
from gateway.textutil import user_text


class RouteState(TypedDict, total=False):
    messages: list[dict[str, str]]
    policy: str
    requested_model: str
    temperature: float
    max_tokens: int
    complexity: str
    complexity_score: float
    classifier: str
    prompt_text: str
    attempt: int
    model_id: str
    initial_model: str
    chosen_tier: int
    provider: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    failover_count: int
    verified: bool
    verify_reason: str
    escalated: bool
    allow_escalation: bool
    error: str
    error_status: int


class RouterGraph:
    def __init__(
        self,
        registry: Registry,
        executor: Executor,
        embedder: Embedder,
        *,
        allow_escalation: bool,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.embedder = embedder
        self.allow_escalation = allow_escalation
        self.graph = self._compile()

    async def run(
        self,
        *,
        messages: list[dict[str, str]],
        policy: str,
        requested_model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> RouteState:
        state: RouteState = {
            "messages": messages,
            "policy": policy,
            "requested_model": requested_model or "",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "attempt": 0,
            "failover_count": 0,
            "escalated": False,
            "allow_escalation": self.allow_escalation and policy != "explicit",
            "verified": False,
            "error": "",
            "error_status": 0,
            "content": "",
            "model_id": "",
            "initial_model": "",
        }
        result: RouteState = await self.graph.ainvoke(state, config={"recursion_limit": 12})
        return result

    def _compile(self) -> Any:
        registry = self.registry
        executor = self.executor
        embedder = self.embedder

        async def classify_node(state: RouteState) -> dict[str, Any]:
            text = user_text(state.get("messages", []))
            classification = classify_text(text, embedder)
            return {
                "prompt_text": text,
                "complexity": classification.label,
                "complexity_score": classification.score,
                "classifier": classification.source,
            }

        async def select_node(state: RouteState) -> dict[str, Any]:
            policy = state.get("policy", "cost")
            attempt = int(state.get("attempt", 0))
            explicit = policy == "explicit"
            models = registry.routing_models(explicit=explicit)
            if explicit:
                chosen = choose_model(
                    "explicit",
                    models,
                    min_tier=0,
                    requested=state.get("requested_model"),
                )
            else:
                if attempt == 0:
                    min_tier = (
                        2 if policy == "quality" else label_tier(state.get("complexity", "medium"))
                    )
                else:
                    previous = int(state.get("chosen_tier", 0))
                    min_tier = max(prompt_tier(state.get("prompt_text", "")), previous + 1)
                chosen = choose_model(policy, models, min_tier=min_tier)
            if chosen is None:
                return {
                    "error": "No model available for this policy",
                    "error_status": 503,
                    "content": "",
                }
            update: dict[str, Any] = {
                "model_id": chosen.id,
                "chosen_tier": chosen.tier,
                "error": "",
            }
            if attempt == 0:
                update["initial_model"] = chosen.id
            return update

        async def generate_node(state: RouteState) -> dict[str, Any]:
            model_id = state.get("model_id", "")
            if state.get("error") and not model_id:
                return {}
            try:
                result, failovers = await executor.execute(
                    model_id,
                    state.get("messages", []),
                    temperature=float(state.get("temperature", 0.0)),
                    max_tokens=int(state.get("max_tokens", 256)),
                )
            except ProviderError as exc:
                return {
                    "content": "",
                    "error": exc.message,
                    "error_status": exc.status_code,
                    "verified": False,
                    "verify_reason": "provider_error",
                }
            spec = registry.get(result.model)
            tier = spec.tier if spec is not None else int(state.get("chosen_tier", 0))
            return {
                "content": result.content,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "provider": result.provider,
                "model_id": result.model,
                "chosen_tier": tier,
                "failover_count": int(state.get("failover_count", 0)) + failovers,
                "error": "",
                "error_status": 0,
            }

        async def verify_node(state: RouteState) -> dict[str, Any]:
            if state.get("error"):
                return {
                    "verified": False,
                    "verify_reason": state.get("verify_reason") or "provider_error",
                }
            ok, reason = verify_answer(state.get("prompt_text", ""), state.get("content", ""))
            return {"verified": ok, "verify_reason": reason}

        async def escalate_node(state: RouteState) -> dict[str, Any]:
            return {"attempt": int(state.get("attempt", 0)) + 1, "escalated": True}

        def decide(state: RouteState) -> str:
            if state.get("error"):
                return "done"
            if state.get("verified"):
                return "done"
            if not state.get("allow_escalation", False):
                return "done"
            if int(state.get("attempt", 0)) >= 2:
                return "done"
            return "escalate"

        builder: StateGraph[RouteState] = StateGraph(RouteState)
        builder.add_node("classify", classify_node)
        builder.add_node("select", select_node)
        builder.add_node("generate", generate_node)
        builder.add_node("verify", verify_node)
        builder.add_node("escalate", escalate_node)
        builder.add_edge(START, "classify")
        builder.add_edge("classify", "select")
        builder.add_edge("select", "generate")
        builder.add_edge("generate", "verify")
        builder.add_conditional_edges("verify", decide, {"done": END, "escalate": "escalate"})
        builder.add_edge("escalate", "select")
        return builder.compile()
