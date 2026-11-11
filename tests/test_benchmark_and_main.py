"""Benchmark harness and the process entrypoint."""

from gateway.benchmark.harness import main, run_benchmark
from gateway.config import get_settings


def test_benchmark_meets_the_bar(tmp_path) -> None:  # type: ignore[no-untyped-def]
    results = run_benchmark(simulate_latency=False)
    assert results["routing_accuracy"] >= 0.9
    assert results["sufficient_model_rate"] == 1
    assert results["exact_cache_hit_rate"] == 1
    assert results["semantic_cache_hit_rate"] >= 0.6
    assert results["savings_pct"] > 20
    output = tmp_path / "results.json"
    main(["--no-latency", "--output", str(output)])
    assert output.exists()


def test_main_binds_host_and_port(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    get_settings.cache_clear()
    import gateway.main as entry

    called: dict[str, object] = {}

    def fake_run(app: object, host: str, port: int) -> None:
        called["app"] = app
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr(entry.uvicorn, "run", fake_run)
    monkeypatch.setenv("PORT", "8123")
    monkeypatch.setenv("HOST", "0.0.0.0")
    entry.main()
    assert called["port"] == 8123
    assert called["host"] == "0.0.0.0"
    assert entry.app.title == "Agent Gateway"
