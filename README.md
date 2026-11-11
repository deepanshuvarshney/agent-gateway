# Agent Gateway

OpenAI-compatible LLM router for **Deepanshu Varshney**. It sits in front of several free model backends, picks a model for each prompt, caches repeated work, and fails over when a provider is unhealthy.

The default backend is a deterministic mock. Tests, GitHub Actions, and the free Render deploy run with **no API keys**.

**Live demo:** [https://agent-gateway-gx4a.onrender.com](https://agent-gateway-gx4a.onrender.com) ([dashboard](https://agent-gateway-gx4a.onrender.com/dashboard), [health](https://agent-gateway-gx4a.onrender.com/health)). The public demo key is `gw_demo_public`. Free Render services spin down after inactivity, so the first request may be a cold start.

## What it does

- `POST /v1/chat/completions` and `GET /v1/models`, including server-sent streaming
- Routing policies: `cost`, `latency`, `quality`, or an explicit model id
- A LangGraph loop: classify, route, generate, verify, and escalate to a stronger model when the draft is too thin
- Exact and semantic cache (Redis, or memory if Redis is down), scoped per API key
- Token bucket, retries, a circuit breaker, and failover on 429 / 5xx / timeouts
- PII redaction, prompt-injection heuristics, per-key token quotas, and pluggable hooks
- JSON logs, Prometheus `/metrics`, and OpenTelemetry (console or OTLP)
- Virtual API keys and usage in SQLite (Postgres optional)
- A small dashboard and a labelled benchmark

## Architecture

```mermaid
flowchart TD
  client[OpenAI SDK or curl] --> api[FastAPI /v1/chat/completions]
  api --> guards[Guardrails: PII, injection, quota, hooks]
  guards --> cache{Per-tenant cache}
  cache -->|exact or semantic hit| respond[Chat completion plus gateway metadata]
  cache -->|miss| graph[LangGraph]
  graph --> classify[Complexity classifier]
  classify --> route[Router agent]
  route --> exec[Retry, token bucket, circuit breaker]
  exec --> providers[Mock, Groq, Gemini, Ollama]
  providers --> verify[Verifier agent]
  verify -->|draft too weak| route
  verify -->|pass| store[Usage store]
  store --> respond
  api --> obs[JSON logs, Prometheus, OpenTelemetry]
  respond --> dash[Dashboard]
```

The verifier does not trust the router's label. It re-reads the prompt. If the draft is too shallow, the router is called again with a higher tier. On this benchmark the first pick is already strong enough, so the escalation rate is zero. The loop is covered by a test that forces the classifier to under-call a hard prompt and checks that the gateway lands on `mock-large`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn gateway.main:app --host 0.0.0.0 --port 8000
```

The local demo key is `gw_demo_public`. The admin key defaults to `gw_admin_dev`. Both are in `.env.example`.

Docker Compose runs the gateway and Redis. Jaeger is behind a profile:

```bash
docker compose up --build
# optional traces:
# GATEWAY_OTEL_EXPORTER=otlp GATEWAY_OTEL_ENDPOINT=http://jaeger:4318/v1/traces \
#   docker compose --profile observability up --build
```

Open [http://localhost:8000/dashboard](http://localhost:8000/dashboard). Health is `GET /health`.

### curl

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer gw_demo_public" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "policy": "cost",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

The body matches the OpenAI chat completion shape and adds a `gateway` object: provider, complexity, cache result, escalation, and the planning-price cost versus always calling the strongest model.

`X-Routing-Policy: latency` is the header form. A concrete model id such as `mock-large` or `openai/gpt-oss-20b` skips automatic routing.

### OpenAI Python SDK

```python
from openai import OpenAI

# gw_demo_public is the local demo tenant key, not a Groq or Gemini secret.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="gw_demo_public")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain what a database index is and when it helps."}],
    extra_body={"policy": "cost"},
)
print(response.choices[0].message.content)
print(response.model)
```

Streaming (`"stream": true`) is buffered. The verifier has to see the full draft before the first token, so a weak answer can be replaced. Time to first token includes that check.

## Benchmark

`benchmarks/prompts.jsonl` has 36 prompts (12 easy, 12 medium, 12 hard) plus 5 paraphrases. Reproduce with:

```bash
python -m gateway.benchmark
```

Numbers below are from one local mock run (`benchmarks/results.json`). Latency includes a simulated provider delay of 8 / 16 / 24 ms for small / medium / large, plus the gateway. Accuracy, cache rates, and savings do not depend on that delay. Prices are planning rates, not invoices.

| Metric | Result |
| --- | --- |
| Routing accuracy | 100% (36/36) |
| Final model strong enough for the label | 100% |
| Escalation rate | 0% |
| Exact cache hit rate, second pass | 100% |
| Semantic cache hit rate | 100% (5/5 paraphrases) |
| Latency p50 / p95 | 19.1 ms / 27.2 ms |
| Savings vs always using the largest model | 27.7% |

Savings by label, same run, cost-first policy:

| Label | Accuracy | Savings vs largest |
| --- | --- | --- |
| easy | 100% | 87.6% |
| medium | 100% | 73.0% |
| hard | 100% | 0% |

Hard prompts are supposed to use the large model, so they do not contribute savings. The 27.7% overall figure is the mix of that set. Spend on the run was about $0.00339 versus $0.00469 if every call had used the large model.

## Providers

| Provider | When it is enabled | Models |
| --- | --- | --- |
| mock | Default. Also on whenever no real provider is configured | `mock-small`, `mock-medium`, `mock-large` |
| Groq | `GATEWAY_GROQ_API_KEY` or `GROQ_API_KEY` | `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-120b` |
| Gemini | `GATEWAY_GEMINI_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY` | `gemini-2.0-flash-lite`, `gemini-2.0-flash`, `gemini-2.5-flash` |
| Ollama | `GATEWAY_OLLAMA_BASE_URL` | `ollama/llama3.2:1b`, `ollama/llama3.2:3b`, `ollama/qwen2.5:7b` |

Groq ids are the ones that succeed on this account. Llama ids return 404 there, so they are not in the catalog.

If a real provider is configured, automatic policies skip the mock. `model: "mock-small"` still works, which keeps the offline path available next to a live key.

## Dashboard and admin keys

`GET /` and `GET /dashboard` serve the same page: request count, cache hit rate, estimated savings, and the latest routes. It does not show prompt text. Usage rows store model, complexity, and cost only.

Create a tenant key with the admin key. The raw key is returned once. Only its SHA-256 hash is stored.

```bash
curl -s http://localhost:8000/admin/keys \
  -H "Authorization: Bearer gw_admin_dev" \
  -H "Content-Type: application/json" \
  -d '{"name": "acme", "quota_tokens": 50000}'
```

Hooks are ordinary callables on `app.state.hooks`:

```python
app.state.hooks.add_pre(my_pre_hook)    # GuardContext -> GuardContext
app.state.hooks.add_post(my_post_hook)  # (content, GuardContext) -> str
```

Built-in checks run first: size, injection, then PII redaction. `GATEWAY_INJECTION_MODE=annotate` keeps the request and prepends a warning instead of rejecting it.

## Design decisions

**Mock by default.** A portfolio gateway that needs a secret to boot cannot pass CI or a free deploy. The mock returns a stable answer per model tier. Small drafts fail the verifier on hard prompts, which is what makes escalation testable offline.

**Planning prices, not invoices.** Groq, Gemini, and Ollama free tiers often bill $0, which would make cost-first routing a tie. Each model carries a public-style USD-per-million rate used only to rank models and to estimate savings against the strongest enabled model. Ollama's rate is zero, so cost-first prefers it when it is configured. Latency-first prefers the lowest latency rank (hosted Groq before a local CPU model).

**Hash embeddings by default.** `sentence-transformers` pulls in a large model and is a poor fit for a 512 MB free instance and for hermetic CI. The default embedder is signed feature hashing over content words. Semantic hits also require token overlap, so a hash collision between unrelated short prompts does not count as a hit. Install `agent-gateway[embeddings]` and set `GATEWAY_EMBEDDINGS=sentence-transformers` to use a local MiniLM model.

**Cache is per API key.** A global semantic cache would raise the hit rate and leak one tenant's completion to another. Entries also store policy, temperature, and `max_tokens`, so a quality-first answer is not reused for a cost-first call.

**Buffered streaming.** A proxy that forwards tokens immediately cannot swap in a stronger model after a quality check. This gateway collects the draft, runs the verifier, and then writes SSE chunks. Real provider streaming would be the next step if time-to-first-token mattered more than the check.

**SQLite on an ephemeral disk.** Render's free web plan is a single instance with no persistent disk and no paid add-ons. Usage survives for the life of the process and is wiped on spin-down or redeploy. `GATEWAY_DATABASE_URL=postgresql://...` switches the same schema to Postgres (`pip install 'agent-gateway[postgres]'`). That is optional and is not part of `render.yaml`, because a free Postgres expires after 30 days and would be a second service.

**Redis is optional.** Compose includes Redis. If the URL is unset or the ping fails, the process keeps serving with an in-memory cache. The free Render blueprint does not attach Key Value.

**Heuristic verifier, not an LLM judge.** A judge model would cost a call and a key. The verifier checks length and, for hard prompts, that the draft actually contains reasoning language. It will miss a hard question that is phrased without words like "prove" or "implement". That limit is deliberate.

**Failover is not escalation.** 429, 5xx, and timeouts retry with backoff, then move to another provider of the same or higher tier. A bad answer on a healthy provider is the verifier's job. Client errors (4xx other than 429) are returned as-is so a bad request is not replayed everywhere.

## Configuration

| Variable | Default | Role |
| --- | --- | --- |
| `PORT` | `8000` | Listen port. Render sets this. The process binds `0.0.0.0` |
| `GATEWAY_DATABASE_URL` | `sqlite:///./data/gateway.db` | SQLite, or Postgres when the URL is `postgres://` / `postgresql://` |
| `GATEWAY_REDIS_URL` | empty | Redis cache. Memory is used when this is empty or unreachable |
| `GATEWAY_EMBEDDINGS` | `hash` | `hash` or `sentence-transformers` |
| `GATEWAY_DEFAULT_POLICY` | `cost` | `cost`, `latency`, or `quality` |
| `GATEWAY_CACHE_SIMILARITY_THRESHOLD` | `0.80` | Cosine cutoff for semantic hits |
| `GATEWAY_CACHE_TTL_SECONDS` | `3600` | Cache TTL |
| `GATEWAY_BOOTSTRAP_API_KEY` | `gw_demo_public` | Demo tenant key created at startup |
| `GATEWAY_ADMIN_API_KEY` | `gw_admin_dev` | Admin key. `render.yaml` generates one |
| `GATEWAY_DEFAULT_QUOTA_TOKENS` | `1000000` | Quota for the demo key and the default for new keys |
| `GATEWAY_MAX_TOKENS_PER_REQUEST` | `2048` | Per-request cap |
| `GATEWAY_INJECTION_MODE` | `block` | `block` or `annotate` |
| `GATEWAY_OTEL_EXPORTER` | `none` | `none`, `console`, or `otlp` |
| `GATEWAY_OTEL_ENDPOINT` | `http://localhost:4318/v1/traces` | OTLP HTTP endpoint |
| `GATEWAY_GROQ_API_KEY` | empty | Also read from `GROQ_API_KEY` |
| `GATEWAY_GEMINI_API_KEY` | empty | Also read from `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `GATEWAY_OLLAMA_BASE_URL` | empty | For example `http://localhost:11434` |
| `GATEWAY_ENABLE_MOCK` | on if no real provider | Set `true` or `false` to force it |

Copy `.env.example` to `.env` for local overrides. The file is not required.

## Deploy on Render

`render.yaml` is a single **free** web service in Oregon. There is no Postgres and no Key Value. The service installs the package, listens on `$PORT`, and uses `/health`. Mock mode is on. Optional provider keys are `sync: false`, so the Blueprint asks for them in the dashboard and the repo never contains them. The admin key is generated by Render.

Free web services spin down after about 15 minutes without traffic, and the local SQLite file is lost when that happens. That is the zero-cost tradeoff. The public demo key in the blueprint is `gw_demo_public`. Rotate it if you put a real upstream key behind the service.

Python is pinned with `PYTHON_VERSION=3.12.8` and `.python-version`.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
mypy src
pytest          # coverage gate is 80%; the suite is currently above 90%
python -m gateway.benchmark
```

CI on every push runs that same ruff, mypy, and pytest job on Python 3.12. No secrets are required.

## Layout

```
src/gateway/
  app.py            HTTP routes
  service.py        guardrails, cache, route, usage
  catalog.py        models and planning prices
  routing/          classifier, policies, verifier, LangGraph
  providers/        mock, Groq, Gemini, Ollama
  cache/            hash embeddings and the semantic cache
  reliability/      token bucket, circuit breaker, failover
  guardrails/       PII, injection, quotas, hooks
  tenancy/          API keys and SQLite/Postgres
  observability/    Prometheus and OpenTelemetry
  dashboard/        static page
  benchmark/        eval harness
benchmarks/prompts.jsonl
benchmarks/results.json
render.yaml
docker-compose.yml
```

## Limits

The classifier is a feature heuristic plus an embedding tie-break near the decision boundary. It is accurate on the labelled set because those prompts use the same kind of signals (short questions, "explain"/"describe", "prove"/"implement"). It is not a general exam of model quality.

The mock ignores `max_tokens` when writing its canned answer so the eval does not depend on character rounding. Real providers receive `max_tokens`.

Free Render will cold-start. The health check is a cheap `SELECT 1` plus the process status, so a cold start still has to finish booting before the platform marks the deploy live.
