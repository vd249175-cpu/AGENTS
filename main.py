"""Local entrypoint for Long River Agent services."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "service",
        nargs="?",
        choices=["mainserver", "seedagent"],
        default="mainserver",
    )
    args = parser.parse_args()

    if args.service == "seedagent":
        from Deepagents.SeedAgent.AgentServer.service import main as run_seed_agent

        return run_seed_agent()

    import uvicorn

    uvicorn.run(
        "MainServer.main_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
