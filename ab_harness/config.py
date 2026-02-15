"""Shared configuration defaults for the A/B test harness.

All values can be overridden via environment variables.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Root of this test harness repository.
HARNESS_ROOT: Path = Path(__file__).resolve().parents[1]

# Root of the mavlink-camera-manager repository.
# Auto-detected as a sibling directory, or set via MCM_REPO_ROOT env var.
MCM_REPO_ROOT: Path = Path(
    os.environ.get(
        "MCM_REPO_ROOT",
        str(HARNESS_ROOT.parent / "mavlink-camera-manager"),
    )
)

# SSH defaults for connecting to the BlueOS device.
SSH_USER: str = os.environ.get("SSH_USER", "pi")
SSH_PWD: str = os.environ.get("SSH_PWD", "raspberry")
SSH_PORT: int = int(os.environ.get("SSH_PORT", "22"))

# Default BlueOS container name.
CONTAINER: str = os.environ.get("BLUEOS_CONTAINER", "blueos-core")

# Optional BlueOS Docker image tag override (e.g. "1.4.3-beta.11").
# When set, the isolated container will use this tag instead of
# auto-detecting the image from the currently running container.
BLUEOS_TAG: str | None = os.environ.get("BLUEOS_TAG") or None


# ---------------------------------------------------------------------------
# SSH configuration and helpers
# ---------------------------------------------------------------------------


@dataclass
class SshConfig:
    """Resolved SSH connection parameters."""

    user: str
    pwd: str
    port: int
    container: str

    @classmethod
    def from_overrides(
        cls,
        user: str | None = None,
        pwd: str | None = None,
        port: int | None = None,
        container: str | None = None,
    ) -> SshConfig:
        """Create a config, falling back to module-level defaults for any None value."""
        return cls(
            user=user if user is not None else SSH_USER,
            pwd=pwd if pwd is not None else SSH_PWD,
            port=port if port is not None else SSH_PORT,
            container=container if container is not None else CONTAINER,
        )


def ssh_run(
    host: str,
    ssh: SshConfig,
    cmd: str,
    *,
    timeout: int = 30,
) -> str:
    """Execute a command on a remote host via SSH and return stdout.

    Returns the stripped stdout on success, or an empty string on failure.
    """
    try:
        result = subprocess.run(
            [
                "sshpass",
                "-p",
                ssh.pwd,
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-p",
                str(ssh.port),
                f"{ssh.user}@{host}",
                cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except Exception as exc:
        log.debug("SSH command failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Consumer / client configuration
# ---------------------------------------------------------------------------


@dataclass
class ConsumerSpec:
    """A single consumer target (WebRTC producer, RTSP path, or thumbnail source).

    Parsed from the CLI format ``NAME[:COUNT]`` where COUNT defaults to 1.
    """

    name: str
    count: int = 1


@dataclass
class ClientConfig:
    """Unified configuration for all stream consumer clients.

    Replaces the old separate ``webrtc_producer``, ``rtsp_path``, and
    ``thumbnail_sources`` parameters that were threaded through the call chain.
    """

    webrtc: list[ConsumerSpec] = field(default_factory=list)
    rtsp: list[ConsumerSpec] = field(default_factory=list)
    thumbnails: list[ConsumerSpec] = field(default_factory=list)
    thumbnail_interval: int = 1  # seconds between thumbnail probes


def parse_consumer_spec(spec: str) -> ConsumerSpec:
    """Parse ``"NAME[:COUNT]"`` into a :class:`ConsumerSpec`.

    Examples::

        parse_consumer_spec("RadCam")       -> ConsumerSpec("RadCam", 1)
        parse_consumer_spec("RadCam:3")     -> ConsumerSpec("RadCam", 3)
        parse_consumer_spec("UDP Stream:2") -> ConsumerSpec("UDP Stream", 2)
    """
    if ":" in spec:
        # Split on the *last* colon so names with colons still work
        name, count_str = spec.rsplit(":", 1)
        try:
            count = int(count_str)
        except ValueError:
            # The part after ":" wasn't a number -- treat the whole thing as the name
            return ConsumerSpec(name=spec, count=1)
        if count < 1:
            raise ValueError(f"Consumer count must be >= 1, got {count} in {spec!r}")
        return ConsumerSpec(name=name, count=count)
    return ConsumerSpec(name=spec, count=1)


def build_client_config(
    *,
    webrtc: list[str] | None = None,
    rtsp: list[str] | None = None,
    thumbnails: list[str] | None = None,
    thumbnail_interval: int = 1,
) -> ClientConfig:
    """Build a :class:`ClientConfig` from raw CLI / YAML string lists."""
    return ClientConfig(
        webrtc=[parse_consumer_spec(s) for s in (webrtc or [])],
        rtsp=[parse_consumer_spec(s) for s in (rtsp or [])],
        thumbnails=[parse_consumer_spec(s) for s in (thumbnails or [])],
        thumbnail_interval=thumbnail_interval,
    )


def client_config_to_dict(cfg: ClientConfig) -> dict[str, Any]:
    """Serialize a :class:`ClientConfig` to a JSON-friendly dict."""
    return {
        "webrtc": [f"{s.name}:{s.count}" for s in cfg.webrtc],
        "rtsp": [f"{s.name}:{s.count}" for s in cfg.rtsp],
        "thumbnails": [f"{s.name}:{s.count}" for s in cfg.thumbnails],
        "thumbnail_interval": cfg.thumbnail_interval,
    }
