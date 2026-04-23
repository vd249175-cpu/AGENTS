"""SeedAgent sandbox accessor."""

from pathlib import Path

from sandbox import LocalDockerSandbox, get_deepagent_workspace_sandbox

from .server.path_resolver import WORKSPACE_ROOT


def get_seed_agent_sandbox(
    workspace_root: Path | None = None,
    *,
    agent_name: str = "SeedAgent",
) -> LocalDockerSandbox:
    return get_deepagent_workspace_sandbox(
        workspace_root or WORKSPACE_ROOT,
        agent_name=agent_name,
    )
