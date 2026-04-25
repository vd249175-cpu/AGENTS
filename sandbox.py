"""DeepAgent sandbox backend built on a persistent real Docker container.

The container exposes real paths directly and keeps the project workspace
mounted at /workspace. File operations and shell execution share the same
container filesystem semantics.
"""

import hashlib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox


def get_docker_cmd() -> str:
    """Return the most likely Docker executable on this machine."""
    if shutil.which("docker"):
        return "docker"

    orb_path = os.path.expanduser("~/.orbstack/bin/docker")
    if os.path.exists(orb_path):
        return orb_path

    desktop_path = "/usr/local/bin/docker"
    if os.path.exists(desktop_path):
        return desktop_path

    return "docker"


DOCKER_CMD = get_docker_cmd()
DEEPAGENT_SANDBOX_IMAGE = "deepagent-sandbox-browser:latest"
_DEEPAGENT_SANDBOX_CACHE: dict[str, "LocalDockerSandbox"] = {}


def get_deepagent_sandbox_dockerfile() -> Path:
    return Path(__file__).resolve().parent / "deepagent_sandbox.Dockerfile"


def _safe_agent_name(agent_name: str | None) -> str:
    raw = str(agent_name or "agent").strip().lower()
    safe = re.sub(r"[^a-z0-9_.-]+", "-", raw).strip("-.")
    return safe or "agent"


def build_deepagent_sandbox_container_name(
    workspace_root: Path,
    *,
    agent_name: str | None = None,
) -> str:
    normalized = str(workspace_root.expanduser().resolve())
    suffix = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"deepagent-{_safe_agent_name(agent_name)}-workspace-{suffix}"


class LocalDockerSandbox(BaseSandbox):
    """Docker sandbox that exposes real container paths without translation."""

    def __init__(
        self,
        container_name: str,
        image: str = DEEPAGENT_SANDBOX_IMAGE,
        mount_dir: str | None = None,
        mount_dest: str = "/workspace",
        network_mode: str = "bridge",
    ) -> None:
        self.container_name = container_name
        self.image = image
        self.mount_dest = mount_dest
        self.root_dest = mount_dest
        self.network_mode = network_mode

        workspace_root = (
            Path(mount_dir)
            if mount_dir is not None
            else Path(__file__).resolve().parent / "workspace"
        )
        self.workspace_root = workspace_root.expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        self._ensure_image_available()
        self._ensure_container_running()

    @property
    def id(self) -> str:
        return self.container_name

    def _ensure_image_available(self) -> None:
        inspect_result = subprocess.run(
            [DOCKER_CMD, "image", "inspect", self.image],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspect_result.returncode == 0:
            return

        dockerfile_path = get_deepagent_sandbox_dockerfile()
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"DeepAgent sandbox dockerfile not found: {dockerfile_path}")

        subprocess.run(
            [
                DOCKER_CMD,
                "build",
                "-t",
                self.image,
                "-f",
                str(dockerfile_path),
                str(dockerfile_path.parent),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _ensure_container_running(self) -> None:
        result = subprocess.run(
            [DOCKER_CMD, "ps", "-a", "-q", "-f", f"name={self.container_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not result.stdout.strip():
            self._create_container()
            return

        if self._container_image_id() != self._image_id():
            self._remove_container()
            self._create_container()
            return

        running = subprocess.run(
            [DOCKER_CMD, "ps", "-q", "-f", f"name={self.container_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not running.stdout.strip():
            subprocess.run(
                [DOCKER_CMD, "start", self.container_name],
                check=True,
                capture_output=True,
                text=True,
            )

    def _create_container(self) -> None:
        run_cmd = [
            DOCKER_CMD,
            "run",
            "-d",
            "--name",
            self.container_name,
            "--network",
            self.network_mode,
            "-v",
            f"{self.workspace_root}:{self.mount_dest}",
            "-w",
            self.root_dest,
            self.image,
            "tail",
            "-f",
            "/dev/null",
        ]
        subprocess.run(run_cmd, check=True, capture_output=True, text=True)

    def _remove_container(self) -> None:
        subprocess.run(
            [DOCKER_CMD, "rm", "-f", self.container_name],
            check=True,
            capture_output=True,
            text=True,
        )

    def _image_id(self) -> str:
        result = subprocess.run(
            [DOCKER_CMD, "image", "inspect", "--format", "{{.Id}}", self.image],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    def _container_image_id(self) -> str:
        result = subprocess.run(
            [DOCKER_CMD, "inspect", "--format", "{{.Image}}", self.container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    def _docker_exec_args(self, command: str, *, interactive: bool = False) -> list[str]:
        exec_cmd = [DOCKER_CMD, "exec"]
        if interactive:
            exec_cmd.append("-i")
        exec_cmd.extend(
            [
                "-w",
                self.root_dest,
                self.container_name,
                "sh",
                "-lc",
                command,
            ]
        )
        return exec_cmd

    @staticmethod
    def _normalize_container_path(path: str) -> str:
        candidate = path if path.startswith("/") else f"/{path}"
        raw_parts = PurePosixPath(candidate).parts
        resolved_parts: list[str] = []
        for part in raw_parts:
            if part in ("", ".", "/"):
                continue
            if part == "..":
                if not resolved_parts:
                    raise ValueError(f"Path escapes DeepAgent workspace root: {path}")
                resolved_parts.pop()
                continue
            resolved_parts.append(part)
        return "/" if not resolved_parts else "/" + "/".join(resolved_parts)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        exec_cmd = self._docker_exec_args(command)

        try:
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=timeout or 120,
                check=False,
            )
            output = result.stdout
            if result.stderr:
                output = f"{output}\n{result.stderr}" if output else result.stderr

            return ExecuteResponse(
                output=output.strip(),
                exit_code=result.returncode,
                truncated=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return ExecuteResponse(
                output=f"{output}\n[DeepAgent sandbox timeout]",
                exit_code=124,
                truncated=True,
            )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for original_path, content in files:
            container_path = self._normalize_container_path(original_path)
            parent = str(PurePosixPath(container_path).parent) or "/"
            mkdir_result = self.execute(f"mkdir -p {shlex.quote(parent)}")
            if mkdir_result.exit_code != 0:
                responses.append(FileUploadResponse(path=original_path, error="invalid_path"))
                continue

            write_result = subprocess.run(
                self._docker_exec_args(f"cat > {shlex.quote(container_path)}", interactive=True),
                input=content,
                capture_output=True,
                check=False,
            )
            if write_result.returncode != 0:
                stderr = (write_result.stderr or b"").decode("utf-8", errors="ignore")
                error = "permission_denied" if "Permission denied" in stderr else "invalid_path"
                responses.append(FileUploadResponse(path=original_path, error=error))
                continue

            responses.append(FileUploadResponse(path=original_path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for original_path in paths:
            container_path = self._normalize_container_path(original_path)
            probe = self.execute(
                "python3 -c "
                + shlex.quote(
                    (
                        "import os,sys;"
                        f"path={container_path!r};"
                        "print("
                        "'file_not_found' if not os.path.exists(path) else "
                        "'is_directory' if os.path.isdir(path) else "
                        "'permission_denied' if not os.access(path, os.R_OK) else "
                        "'ok'"
                        ")"
                    )
                )
            )
            status = probe.output.strip()
            if status != "ok":
                error = status if status in {"file_not_found", "permission_denied", "is_directory"} else "invalid_path"
                responses.append(FileDownloadResponse(path=original_path, content=None, error=error))
                continue

            read_result = subprocess.run(
                self._docker_exec_args(f"cat {shlex.quote(container_path)}"),
                capture_output=True,
                check=False,
            )
            if read_result.returncode != 0:
                stderr = (read_result.stderr or b"").decode("utf-8", errors="ignore")
                error = "permission_denied" if "Permission denied" in stderr else "invalid_path"
                responses.append(FileDownloadResponse(path=original_path, content=None, error=error))
                continue

            responses.append(FileDownloadResponse(path=original_path, content=read_result.stdout, error=None))
        return responses


def get_deepagent_workspace_sandbox(
    workspace_root: Path | None = None,
    *,
    agent_name: str | None = None,
) -> LocalDockerSandbox:
    resolved_root = (
        workspace_root.expanduser().resolve()
        if workspace_root is not None
        else (Path(__file__).resolve().parent / "workspace").resolve()
    )
    cache_key = f"{_safe_agent_name(agent_name)}:{resolved_root}"
    sandbox = _DEEPAGENT_SANDBOX_CACHE.get(cache_key)
    if sandbox is None:
        sandbox = LocalDockerSandbox(
            container_name=build_deepagent_sandbox_container_name(
                resolved_root,
                agent_name=agent_name,
            ),
            mount_dir=str(resolved_root),
            network_mode="bridge",
        )
        _DEEPAGENT_SANDBOX_CACHE[cache_key] = sandbox
    return sandbox
