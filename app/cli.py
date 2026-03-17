import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx

from app.config import settings


ENV_TEMPLATE = """CRAWL4AI_LLM_BASE_URL=http://173.10.88.250:8000/v1
VLLM_API_KEY=jaaspire-key
VLLM_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO
"""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_path() -> Path:
    return _project_root() / ".env"


def _health_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def ensure_env(force: bool = False) -> None:
    env_path = _env_path()
    if env_path.exists() and not force:
        print(f"[OK] Using existing .env at {env_path}")
        return

    env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
    print(f"[OK] Wrote .env at {env_path}")


def _uvicorn_cmd(host: str, port: int, reload_mode: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        settings.log_level.lower(),
    ]
    if reload_mode:
        cmd.append("--reload")
    return cmd


def run_doctor(base_url: str, with_search: bool) -> int:
    base = base_url.rstrip("/")
    failures = 0

    with httpx.Client(timeout=60.0) as client:
        print("[CHECK] GET /health")
        try:
            health = client.get(f"{base}/health")
            health.raise_for_status()
            print("[OK] /health")
        except Exception as exc:
            print(f"[FAIL] /health -> {exc}")
            failures += 1

        print("[CHECK] GET /models")
        try:
            models = client.get(f"{base}/models")
            models.raise_for_status()
            print("[OK] /models")
        except Exception as exc:
            print(f"[FAIL] /models -> {exc}")
            failures += 1

        print("[CHECK] POST /chat/completions (search_web=false)")
        no_search_payload = {
            "model": settings.vllm_model_name,
            "messages": [{"role": "user", "content": "What is 2+2? Reply only number."}],
            "stream": False,
            "search_web": False,
        }
        try:
            resp = client.post(f"{base}/chat/completions", json=no_search_payload)
            resp.raise_for_status()
            print("[OK] /chat/completions without search")
        except Exception as exc:
            print(f"[FAIL] /chat/completions without search -> {exc}")
            failures += 1

        if with_search:
            print("[CHECK] POST /chat/completions (search_web=true)")
            search_payload = {
                "model": settings.vllm_model_name,
                "messages": [{"role": "user", "content": "Latest AI news today in one line"}],
                "stream": False,
                "search_web": True,
            }
            try:
                resp = client.post(f"{base}/chat/completions", json=search_payload)
                resp.raise_for_status()
                print("[OK] /chat/completions with search")
            except Exception as exc:
                print(f"[FAIL] /chat/completions with search -> {exc}")
                failures += 1

    if failures:
        print(f"[RESULT] Doctor found {failures} issue(s)")
    else:
        print("[RESULT] All checks passed")
    return failures


def wait_for_server(base_url: str, timeout_sec: int) -> None:
    base = base_url.rstrip("/")
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=3.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)

    raise TimeoutError(f"Server did not become ready within {timeout_sec}s")


def cmd_init(args: argparse.Namespace) -> int:
    ensure_env(force=args.force_env)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ensure_env(force=args.force_env)
    cmd = _uvicorn_cmd(args.host, args.port, args.reload)
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_project_root()))


def cmd_doctor(args: argparse.Namespace) -> int:
    failures = run_doctor(args.base_url, args.with_search)
    return 1 if failures else 0


def cmd_auto(args: argparse.Namespace) -> int:
    ensure_env(force=args.force_env)

    cmd = _uvicorn_cmd(args.host, args.port, args.reload)
    proc = subprocess.Popen(cmd, cwd=str(_project_root()))

    local_base_url = args.base_url or f"http://{_health_host(args.host)}:{args.port}/v1"

    try:
        print("[INFO] Waiting for server startup...")
        wait_for_server(local_base_url, args.wait_timeout)
        print(f"[OK] Server is live at {local_base_url}")

        failures = run_doctor(local_base_url, args.with_search)
        if failures:
            print("[WARN] Startup checks completed with warnings.")
        else:
            print("[OK] Startup checks completed successfully.")

        print("[INFO] Server is running. Press Ctrl+C to stop.")
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0
    except Exception as exc:
        print(f"[FAIL] Auto mode failed: {exc}")
        proc.terminate()
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaspire",
        description="JASPIRE Chat API command line helper",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Create .env if missing")
    init_parser.add_argument("--force-env", action="store_true", help="Overwrite existing .env")
    init_parser.set_defaults(func=cmd_init)

    run_parser = sub.add_parser("run", help="Run FastAPI server")
    run_parser.add_argument("--host", default=settings.api_host)
    run_parser.add_argument("--port", type=int, default=settings.api_port)
    run_parser.add_argument("--reload", action="store_true", default=settings.environment == "development")
    run_parser.add_argument("--force-env", action="store_true", help="Overwrite existing .env")
    run_parser.set_defaults(func=cmd_run)

    doctor_parser = sub.add_parser("doctor", help="Run API health checks")
    doctor_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    doctor_parser.add_argument("--with-search", action="store_true", help="Also test search_web=true path")
    doctor_parser.set_defaults(func=cmd_doctor)

    auto_parser = sub.add_parser("auto", help="Start server + run checks automatically")
    auto_parser.add_argument("--host", default=settings.api_host)
    auto_parser.add_argument("--port", type=int, default=settings.api_port)
    auto_parser.add_argument("--reload", action="store_true", default=settings.environment == "development")
    auto_parser.add_argument("--wait-timeout", type=int, default=40)
    auto_parser.add_argument("--base-url", default="")
    auto_parser.add_argument("--with-search", action="store_true", help="Also test search_web=true path")
    auto_parser.add_argument("--force-env", action="store_true", help="Overwrite existing .env")
    auto_parser.set_defaults(func=cmd_auto)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
