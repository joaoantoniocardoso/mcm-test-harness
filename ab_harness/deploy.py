"""Build and deploy MCM -- local (cargo build+run) or remote (SSH deploy script)."""

from __future__ import annotations

import json as _json
import logging
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import ClassVar

from ab_harness.config import MCM_REPO_ROOT

log = logging.getLogger(__name__)

DEPLOY_SCRIPT = MCM_REPO_ROOT / "dev" / "deploy_and_restart_mcm_in_container.sh"
TARGET = os.environ.get("CROSS_TARGET", "armv7-unknown-linux-gnueabihf")
LOCAL_BINARY = MCM_REPO_ROOT / "target" / TARGET / "release" / "mavlink-camera-manager"


def _wait_for_api(base_url: str, timeout: float = 90, interval: float = 3) -> bool:
    """Poll until the stats API responds or timeout."""
    deadline = time.monotonic() + timeout
    endpoint = f"{base_url}/stats/pipeline-analysis"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5):
                log.info("API is ready at %s", endpoint)
                return True
        except Exception:
            remaining = deadline - time.monotonic()
            log.debug("API not ready yet, %.0fs remaining...", remaining)
            time.sleep(interval)
    log.error("API did not become ready within %.0fs", timeout)
    return False


class LocalDeployment:
    """Build with cargo and run the binary as a local subprocess."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 6020,
        extra_args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
        mcm_config: Path | None = None,
    ):
        self.host = host
        self.port = port
        self.extra_args = extra_args or []
        self.env_overrides = env_overrides or {}
        self.mcm_config = mcm_config
        self._process: subprocess.Popen[str] | None = None

        # When the user provides a config file we work with a copy so MCM's
        # runtime mutations never touch the original.
        self._settings_working_copy: Path | None = None
        if mcm_config is not None:
            if not mcm_config.exists():
                raise FileNotFoundError(f"MCM config file not found: {mcm_config}")
            self._settings_working_copy = Path("/tmp/mcm_settings_harness.json")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _restore_mcm_config(self) -> None:
        """Copy the original user-provided config to the working copy.

        Called before every MCM invocation so the process always starts from
        a pristine configuration.
        """
        if self.mcm_config is None or self._settings_working_copy is None:
            return
        shutil.copy2(self.mcm_config, self._settings_working_copy)
        log.info(
            "Restored MCM settings: %s -> %s",
            self.mcm_config,
            self._settings_working_copy,
        )

    @property
    def settings_file_path(self) -> Path | None:
        """Return the path MCM is using for its settings, or None for default."""
        return self._settings_working_copy

    def download_mcm_config(self, dest: Path) -> Path | None:
        """Copy MCM's current settings file to *dest*/settings.json.

        Returns the local path on success, or ``None`` if no settings file exists.
        """
        src = self._settings_working_copy
        if src is None or not src.exists():
            return None
        dest.mkdir(parents=True, exist_ok=True)
        local_path = dest / "settings.json"
        shutil.copy2(src, local_path)
        log.info("MCM config copied to %s", local_path)
        return local_path

    def build(self) -> None:
        log.info("Building MCM with cargo build --release ...")
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=MCM_REPO_ROOT,
            check=True,
            env={**os.environ, "SKIP_WEB": "1"},
        )
        log.info("Build succeeded.")

    def start(self) -> None:
        binary = MCM_REPO_ROOT / "target" / "release" / "mavlink-camera-manager"
        if not binary.exists():
            raise FileNotFoundError(f"Binary not found: {binary}")

        self._restore_mcm_config()

        cmd = [
            str(binary),
            "--rest-server",
            f"{self.host}:{self.port}",
            "--verbose",
            "--pipeline-analysis-level",
            "full",
            *self.extra_args,
        ]
        if self._settings_working_copy is not None:
            cmd.extend(["--settings-file", str(self._settings_working_copy)])

        env = {**os.environ, **self.env_overrides}
        log.info("Starting MCM locally: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            cwd=MCM_REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if not _wait_for_api(self.base_url):
            self.stop()
            raise RuntimeError("Local MCM did not start in time")

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            log.info("Stopping local MCM (pid=%d)...", self._process.pid)
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            log.info("MCM stopped.")

    def collect_logs(self, dest: Path) -> None:
        """Capture stdout from the local process."""
        dest.mkdir(parents=True, exist_ok=True)
        if self._process and self._process.stdout:
            try:
                out = self._process.stdout.read()
                if out:
                    (dest / "mcm_stdout.log").write_text(out)
            except Exception as exc:
                log.warning("Could not read local MCM stdout: %s", exc)


class RemoteDeployment:
    """Cross-compile and deploy via dev/deploy_and_restart_mcm_in_container.sh."""

    # Default env vars passed to MCM inside the container.
    _MCM_ENV_DEFAULTS: ClassVar[dict[str, str]] = {
        "GST_DEBUG": "3",
    }

    # MCM CLI args inside the container (matching the deploy script's command)
    _MCM_BINARY = "/root/mavlink-camera-manager"
    _MCM_ARGS_BASE = (
        " --default-settings BlueROVUDP"
        " --mavlink udpin:127.0.0.1:5777"
        " --mavlink-system-id 1"
        " --mavlink-camera-component-id-range=100-105"
        " --gst-feature-rank omxh264enc=0,v4l2h264enc=250,x264enc=260"
        " --log-path /var/logs/blueos/services/mavlink-camera-manager"
        " --stun-server stun://stun.l.google.com:19302"
    )
    _MCM_ARGS_ZENOH = " --zenoh"
    _MCM_ARGS_TAIL = " --verbose --pipeline-analysis-level full"

    _ISOLATED_CONTAINER = "blueos-core-ab-test"

    def __init__(
        self,
        *,
        host: str = "blueos.local",
        port: int = 6020,
        env_overrides: dict[str, str] | None = None,
        ssh_user: str | None = None,
        ssh_pwd: str | None = None,
        ssh_port: int | None = None,
        container: str | None = None,
        isolated: bool = True,
        extra_args: list[str] | None = None,
        blueos_tag: str | None = None,
        mcm_config: Path | None = None,
    ):
        from ab_harness.config import BLUEOS_TAG, SshConfig

        self.host = host
        self.port = port
        self.extra_args = extra_args or []
        self.env_overrides = env_overrides or {}
        self._ssh_config = SshConfig.from_overrides(
            user=ssh_user,
            pwd=ssh_pwd,
            port=ssh_port,
            container=container,
        )
        self.ssh_user = self._ssh_config.user
        self.ssh_pwd = self._ssh_config.pwd
        self.ssh_port = self._ssh_config.port
        self.isolated = isolated
        self.blueos_tag = blueos_tag or BLUEOS_TAG
        self.mcm_config = mcm_config
        self._original_container = self._ssh_config.container
        self.container = self._original_container
        self._isolated_prepared = False
        self._original_governor: str | None = None
        self._uploaded_tags: set[str] = set()
        self._tag_env: dict[str, dict[str, str]] = {}
        self._mcm_config_uploaded = False
        self._settings_backed_up = False
        # Captured before any containers are stopped so the versions reflect
        # the device's state prior to the test.
        self.blueos_core_version: str | None = None
        self.blueos_bootstrap_version: str | None = None
        self._versions_captured = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # Path inside the container where the harness places the working copy
    # of the user-provided settings file.  MCM is pointed here via
    # ``--settings-file`` so the original is never mutated.
    _MCM_SETTINGS_OVERRIDE = "/tmp/mcm_settings_harness.json"

    @property
    def _mcm_args(self) -> str:
        """Full MCM CLI args, skipping --zenoh in isolated mode (no router)."""
        zenoh = "" if self._isolated_prepared else self._MCM_ARGS_ZENOH
        extra = (" " + " ".join(self.extra_args)) if self.extra_args else ""
        settings = (
            f" --settings-file {self._MCM_SETTINGS_OVERRIDE}"
            if self.mcm_config is not None
            else ""
        )
        return self._MCM_ARGS_BASE + zenoh + self._MCM_ARGS_TAIL + settings + extra

    # ---- BlueOS version capture -----------------------------------------------

    def capture_blueos_versions(self) -> None:
        """Inspect running containers to record blueos-core and blueos-bootstrap versions.

        Must be called while the containers are still running (i.e. before
        ``prepare_isolated_container`` stops them).  Only runs once per
        deployer lifetime; subsequent calls are no-ops.
        """
        if self._versions_captured:
            return

        # blueos-core image (e.g. "bluerobotics/blueos-core:1.4.2")
        try:
            core_img = self._ssh(
                f"docker inspect {self._original_container} --format='{{{{.Config.Image}}}}'",
                timeout=10,
            ).strip("'\" \n")
            self.blueos_core_version = core_img or None
        except Exception as exc:
            log.warning("Could not inspect blueos-core version: %s", exc)

        # blueos-bootstrap image (e.g. "bluerobotics/blueos-bootstrap:1.4.2")
        try:
            bootstrap_img = self._ssh(
                "docker inspect blueos-bootstrap --format='{{.Config.Image}}'",
                timeout=10,
            ).strip("'\" \n")
            self.blueos_bootstrap_version = bootstrap_img or None
        except Exception as exc:
            log.warning("Could not inspect blueos-bootstrap version: %s", exc)

        log.info(
            "BlueOS versions: core=%s, bootstrap=%s",
            self.blueos_core_version,
            self.blueos_bootstrap_version,
        )
        self._versions_captured = True

    # ---- Isolated container management ---------------------------------------

    def prepare_isolated_container(self) -> None:
        """Stop all containers and start a clean one with the original image/mounts.

        The new container replicates the full bind-mount set of the original
        ``blueos-core`` container but uses a minimal entrypoint that only starts
        a tmux server (no BlueOS services).  This dramatically reduces system
        noise for A/B measurements.

        After this call, ``self.container`` points to the isolated container
        and all subsequent ``swap_and_restart`` / ``docker exec`` calls target it.
        """
        if self._isolated_prepared:
            log.info("Isolated container already prepared, skipping.")
            return

        # 0. Capture BlueOS versions while containers are still running ------
        self.capture_blueos_versions()

        # 1. Inspect the running container to get image and bind mounts -------
        detected_image = self._ssh(
            f"docker inspect {self._original_container} --format='{{{{.Config.Image}}}}'"
        ).strip("'\"")
        if not detected_image:
            raise RuntimeError(
                f"Could not inspect container '{self._original_container}'. "
                "Is it running? Use --no-isolated to skip isolation."
            )

        # Apply --blueos-tag override if specified
        if self.blueos_tag:
            # Replace the tag portion of the image (repo:tag -> repo:new_tag)
            repo = detected_image.rsplit(":", 1)[0] if ":" in detected_image else detected_image
            image = f"{repo}:{self.blueos_tag}"
            log.info(
                "Overriding BlueOS image tag: %s -> %s",
                detected_image,
                image,
            )
        else:
            image = detected_image

        binds_raw = self._ssh(
            f"docker inspect {self._original_container} --format='{{{{json .HostConfig.Binds}}}}'"
        )
        log.info(
            "Inspected %s: image=%s, binds=%d chars",
            self._original_container,
            image,
            len(binds_raw),
        )

        # 1b. Back up the device's current MCM settings before we touch anything
        self.backup_remote_settings()

        # 2. Stop ALL containers on the Pi ------------------------------------
        log.info("Stopping all containers on the Pi...")
        self._ssh("docker stop $(docker ps -q) 2>/dev/null || true", timeout=60)
        time.sleep(3)

        # 3. Remove stale isolated container if it exists from a previous run --
        self._ssh(f"docker rm -f {self._ISOLATED_CONTAINER} 2>/dev/null || true")

        # 4. Build the docker run command with all original mounts -------------
        volume_flags = ""
        if binds_raw and binds_raw not in ("null", "'null'", ""):
            # Strip surrounding single quotes that --format may add
            cleaned = binds_raw.strip("'")
            try:
                binds: list[str] = _json.loads(cleaned)
                for bind in binds:
                    volume_flags += f" -v {bind}"
            except (_json.JSONDecodeError, TypeError):
                log.warning("Could not parse bind mounts, proceeding without: %s", binds_raw)

        entrypoint = (
            "mkdir -p /var/logs/blueos/services/mavlink-camera-manager "
            "&& tmux start-server "
            "&& sleep infinity"
        )

        docker_cmd = (
            f"docker run -d --name {self._ISOLATED_CONTAINER}"
            f" --privileged --network=host --pid=host"
            f"{volume_flags}"
            f" -e BLUEOS_DISABLE_STARTUP_UPDATE=true"
            f" {image}"
            f" bash -c '{entrypoint}'"
        )

        log.info("Starting isolated container: %s", self._ISOLATED_CONTAINER)
        container_id = self._ssh(docker_cmd, timeout=60)
        if not container_id:
            raise RuntimeError(
                "Failed to start isolated container. Check Pi connectivity and Docker state."
            )

        # 5. Update the active container reference -----------------------------
        self.container = self._ISOLATED_CONTAINER
        self._isolated_prepared = True

        # Give the container a moment to fully start
        time.sleep(2)

        # Verify it is running
        status = self._ssh(
            f"docker inspect {self._ISOLATED_CONTAINER} --format='{{{{.State.Running}}}}'"
        ).strip("'\"")
        if status != "true":
            raise RuntimeError(f"Isolated container is not running (status={status!r})")

        log.info(
            "Isolated container ready: %s (image: %s)",
            self._ISOLATED_CONTAINER,
            image,
        )

    # ---- Pre-measurement hardening -------------------------------------------

    def clear_memory(self) -> None:
        """Flush page cache, dentries/inodes and swap on the Pi host.

        This removes memory-related variance between the first rep and
        later reps, ensuring a consistent starting state.
        """
        log.info("Clearing page cache and swap on the Pi...")
        self._ssh(
            "sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'",
            timeout=15,
        )
        # swapoff/swapon may fail if swap is not configured -- that's fine
        self._ssh(
            "sudo sh -c 'swapoff -a && swapon -a' 2>/dev/null || true",
            timeout=60,
        )
        log.info("Memory and swap cleared.")

    def set_cpu_governor(self, governor: str = "performance") -> None:
        """Pin all CPU cores to *governor* (e.g. ``performance``).

        The previous governor is saved so that
        :meth:`teardown_isolated_container` can restore it automatically.
        Setting ``performance`` locks the CPU at max frequency, removing
        frequency-scaling jitter from measurements.
        """
        prev = self._ssh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip()
        if prev:
            self._original_governor = prev
        log.info(
            "Setting CPU governor: %s -> %s",
            self._original_governor or "unknown",
            governor,
        )
        self._ssh(
            f"echo {governor} | sudo tee "
            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null"
        )

    def _read_temp(self) -> float:
        """Read current SoC temperature in degrees Celsius."""
        raw = self._ssh("cat /sys/class/thermal/thermal_zone0/temp")
        return int(raw) / 1000.0

    def thermal_warmup(self, *, timeout: int = 120) -> dict[str, float]:
        """Stress all CPU cores until the SoC temperature stabilises.

        Spawns one bash busy-loop per core on the Pi host, then polls the
        thermal zone every 5 s.  Stops when the temperature changes by
        less than 0.5 C over two consecutive 5 s reads (i.e. 10 s window),
        or when *timeout* seconds elapse.

        After stopping the stress workers a 5 s cool-down period allows the
        CPU frequency governor to settle before real measurements begin.

        Returns a dict with ``start_temp_c``, ``end_temp_c``, and
        ``duration_s`` suitable for inclusion in experiment metadata.
        """
        start_temp = self._read_temp()
        log.info(
            "Thermal warmup: starting at %.1f C (timeout %ds)...",
            start_temp,
            timeout,
        )

        # Spawn one busy-loop per core via a backgrounded SSH session.
        # We use a distinctive marker ('ab_thermal_stress') so we can
        # reliably pkill it later.
        self._ssh(
            "nohup bash -c '"
            "for i in $(seq 1 $(nproc)); do "
            "  (while :; do :; done) & "
            "done; "
            "wait' >/dev/null 2>&1 &",
            timeout=5,
        )

        prev_temp = start_temp
        prev_prev_temp = start_temp
        elapsed = 0.0
        poll_interval = 5

        try:
            while elapsed < timeout:
                time.sleep(poll_interval)
                elapsed += poll_interval
                temp = self._read_temp()
                log.debug(
                    "Thermal warmup: %.1f C after %.0fs (delta %.1f C)",
                    temp,
                    elapsed,
                    temp - prev_temp,
                )
                # Stable = both of the last two 5s deltas < 0.5 C
                if (
                    abs(temp - prev_temp) < 0.5
                    and abs(prev_temp - prev_prev_temp) < 0.5
                    and elapsed >= 2 * poll_interval  # need at least 2 reads
                ):
                    log.info(
                        "Thermal warmup: stable at %.1f C after %.0fs",
                        temp,
                        elapsed,
                    )
                    break
                prev_prev_temp = prev_temp
                prev_temp = temp
            else:
                log.warning(
                    "Thermal warmup: timeout after %ds at %.1f C (may not be fully stable)",
                    timeout,
                    prev_temp,
                )
        finally:
            # Kill all busy-loop workers
            self._ssh(
                "pkill -f 'while :; do :; done' 2>/dev/null || true",
                timeout=10,
            )

        # Brief cool-down so the governor settles
        cooldown = 5
        log.info("Thermal warmup: %ds cool-down...", cooldown)
        time.sleep(cooldown)
        end_temp = self._read_temp()

        result = {
            "start_temp_c": round(start_temp, 1),
            "end_temp_c": round(end_temp, 1),
            "duration_s": round(elapsed + cooldown, 1),
        }
        log.info("Thermal warmup complete: %s", result)
        return result

    def harden_system(self) -> dict[str, float]:
        """Clear memory, pin CPU governor, and run thermal warmup.

        Convenience wrapper that runs the three pre-measurement hardening
        steps in sequence and returns the thermal-warmup metadata dict.
        Designed to be callable from a ``ThreadPoolExecutor`` in parallel
        with local build operations.
        """
        self.clear_memory()
        self.set_cpu_governor("performance")
        return self.thermal_warmup()

    # ---- Teardown ------------------------------------------------------------

    def teardown_isolated_container(self, *, restore: bool = True) -> None:
        """Remove the isolated container and optionally restart BlueOS.

        When *restore* is ``True`` (the default), ``blueos-bootstrap`` is
        started after cleanup, which brings the full BlueOS stack back up
        (including ``blueos-core`` and all extensions).

        When *restore* is ``False``, the Pi is left with no running
        containers -- the caller is responsible for restoring BlueOS.
        """
        if not self._isolated_prepared:
            return

        # Restore CPU governor before tearing down
        if self._original_governor:
            log.info("Restoring CPU governor to '%s'", self._original_governor)
            self._ssh(
                f"echo {self._original_governor} | sudo tee "
                "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor "
                ">/dev/null 2>&1 || true",
                timeout=10,
            )
            self._original_governor = None

        log.info("Tearing down isolated container: %s", self._ISOLATED_CONTAINER)
        self._ssh(
            f"docker stop {self._ISOLATED_CONTAINER} 2>/dev/null || true",
            timeout=30,
        )
        self._ssh(f"docker rm {self._ISOLATED_CONTAINER} 2>/dev/null || true")

        # Restore the container reference
        self.container = self._original_container
        self._isolated_prepared = False

        if restore:
            self.restore_blueos()
            # Give the container a moment to start before restoring settings
            time.sleep(5)
            self.restore_remote_settings()
        else:
            log.info(
                "Skipping BlueOS restore (--no-restore). "
                "Run 'python -m ab_harness restore' to bring BlueOS back up."
            )

    def restore_blueos(self) -> None:
        """Start ``blueos-bootstrap`` to bring the full BlueOS stack back up.

        The bootstrap container manages the lifecycle of ``blueos-core``
        and all extensions.  It is the canonical way to (re)start BlueOS
        on the Pi.

        Safe to call at any time -- if the bootstrap is already running
        the ``docker start`` is a no-op.
        """
        log.info("Starting blueos-bootstrap to restore the full BlueOS stack...")
        self._ssh("docker start blueos-bootstrap", timeout=60)
        log.info("blueos-bootstrap started.  BlueOS will be up shortly.")

    # ---- General helpers -----------------------------------------------------

    def set_tag_env(self, tag: str, env: dict[str, str]) -> None:
        """Set extra env vars that will be applied when swap_and_restart(tag) runs."""
        self._tag_env[tag] = env

    def _build_mcm_cmd(self, tag: str) -> str:
        """Build the full MCM command string with env vars for a given tag.

        Uses 'env' to set environment variables so that 'nice' sees the binary
        as its command (not the VAR=val assignments).
        """
        merged = dict(self._MCM_ENV_DEFAULTS)
        merged.update(self._tag_env.get(tag, {}))
        env_str = " ".join(f"{k}={v}" for k, v in merged.items())
        return f"env {env_str} nice --19 {self._MCM_BINARY}{self._mcm_args}"

    def build_and_deploy(self) -> None:
        if not DEPLOY_SCRIPT.exists():
            raise FileNotFoundError(f"Deploy script not found: {DEPLOY_SCRIPT}")

        # Capture BlueOS versions while containers are still running.
        self.capture_blueos_versions()

        # Back up the device's settings before we touch anything.
        self.backup_remote_settings()

        # Place the user-provided config at the *default* location so the
        # deploy script's hardcoded MCM command picks it up.  The deploy
        # script does not touch settings.json -- it only replaces the binary
        # and restarts MCM.
        if self.mcm_config is not None:
            self._upload_mcm_config_to_host()
            self._ssh(
                f"docker cp /tmp/_mcm_settings_original.json"
                f" {self.container}:{self._MCM_SETTINGS_DEFAULT}"
            )
            log.info("Placed user-provided MCM config at default path for deploy script.")

        env = {
            **os.environ,
            "SSH_HOST": self.host,
            "SSH_USER": self.ssh_user,
            "SSH_PWD": self.ssh_pwd,
            "SSH_PORT": str(self.ssh_port),
            "CONTAINER": self.container,
            **self.env_overrides,
        }
        # In isolated mode there is no Zenoh router running
        if self._isolated_prepared:
            env["MCM_ENABLE_ZENOH"] = "false"
        log.info("Running deploy script: %s", DEPLOY_SCRIPT)
        subprocess.run(
            ["bash", str(DEPLOY_SCRIPT)],
            cwd=MCM_REPO_ROOT,
            env=env,
            check=True,
        )
        log.info("Deploy script finished. Waiting for API...")
        if not _wait_for_api(self.base_url):
            raise RuntimeError("Remote MCM API did not become ready")

    # ---- Interleaved helpers ------------------------------------------------

    def build_only(self, tag: str, *, dest_dir: Path | None = None) -> Path:
        """Cross-compile and save the binary locally under a tag.

        If *dest_dir* is given the binary is saved there as ``mcm_{tag}``.
        If a binary already exists at that location it is **reused** and the
        build is skipped entirely (build-once semantics).

        Returns the path to the tagged local binary.
        """
        if dest_dir is not None:
            cached = dest_dir / f"mcm_{tag}"
            if cached.exists():
                log.info(
                    "Reusing cached binary for tag=%s at %s (skipping build)",
                    tag,
                    cached,
                )
                return cached

        log.info("Building MCM (tag=%s) with cross build --release ...", tag)
        subprocess.run(
            [
                "cross",
                "build",
                "--release",
                "--locked",
                f"--target={TARGET}",
            ],
            cwd=MCM_REPO_ROOT,
            env={**os.environ, "SKIP_WEB": "1", **self.env_overrides},
            check=True,
        )

        if dest_dir is not None:
            dest_dir.mkdir(parents=True, exist_ok=True)
            tagged = dest_dir / f"mcm_{tag}"
        else:
            tagged = MCM_REPO_ROOT / "target" / f"mcm_{tag}"
        shutil.copy2(LOCAL_BINARY, tagged)
        log.info("Build succeeded for tag=%s -> %s", tag, tagged)
        return tagged

    def _remote_binary_exists(self, tag: str) -> bool:
        """Check whether /tmp/mcm_{tag} already exists on the Pi."""
        out = self._ssh(f"test -f /tmp/mcm_{tag} && echo yes || echo no")
        return out.strip() == "yes"

    def _scp(
        self,
        local_path: str,
        remote_path: str,
        *,
        upload: bool = True,
        check: bool = True,
        timeout: int = 120,
    ) -> None:
        """Transfer a file between local and remote via SCP.

        When *upload* is ``True`` (default), copies local -> remote.
        When ``False``, copies remote -> local.
        """
        if upload:
            src = local_path
            dst = f"{self.ssh_user}@{self.host}:{remote_path}"
        else:
            src = f"{self.ssh_user}@{self.host}:{remote_path}"
            dst = local_path

        subprocess.run(
            [
                "sshpass",
                "-p",
                self.ssh_pwd,
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-P",
                str(self.ssh_port),
                src,
                dst,
            ],
            capture_output=not check,
            check=check,
            timeout=timeout,
        )

    def upload_binary(self, tag: str, local_path: Path, *, force: bool = False) -> None:
        """SCP a pre-built binary to the Pi's /tmp/ under the given tag.

        Skips the upload if the binary is already present on the Pi
        unless *force* is ``True``.
        """
        if force:
            self._uploaded_tags.discard(tag)
        elif tag in self._uploaded_tags:
            log.info("Binary for tag=%s already uploaded this session, skipping.", tag)
            return
        elif self._remote_binary_exists(tag):
            log.info(
                "Binary /tmp/mcm_%s already exists on Pi, skipping upload.",
                tag,
            )
            self._uploaded_tags.add(tag)
            return

        remote_name = f"mcm_{tag}"
        log.info("Uploading %s -> pi:/tmp/%s ...", local_path, remote_name)
        self._scp(str(local_path), f"/tmp/{remote_name}")
        self._uploaded_tags.add(tag)
        log.info("Upload complete for tag=%s", tag)

    def swap_and_restart(self, tag: str) -> None:
        """Stop MCM, swap in a pre-uploaded binary, and restart.

        Much faster than a full build_and_deploy (seconds instead of minutes).
        """
        remote_name = f"mcm_{tag}"
        log.info("Swapping to tag=%s and restarting MCM ...", tag)

        # Kill MCM and clean up tmux
        self._ssh(
            f"docker exec {self.container} bash -c '"
            f"tmux kill-session -t video 2>/dev/null; "
            f"pkill -9 -f run-service.video 2>/dev/null; "
            f"pkill -9 mavlink-camera 2>/dev/null; "
            f"sleep 1'"
        )
        time.sleep(2)

        # Copy pre-uploaded binary into the container
        self._ssh(f"docker cp /tmp/{remote_name} {self.container}:/root/mavlink-camera-manager")

        # Restore pristine settings before MCM reads them
        self._restore_mcm_config()

        # Start MCM inside the container via tmux (full command with env vars)
        mcm_cmd = self._build_mcm_cmd(tag)
        self._ssh(f"docker exec {self.container} tmux new-session -d -s video '{mcm_cmd}'")

        if not _wait_for_api(self.base_url, timeout=60):
            raise RuntimeError(f"MCM ({tag}) did not come up after swap_and_restart")
        log.info("MCM is running with tag=%s", tag)

    # ---- MCM config (settings.json) management --------------------------------

    # Default path to MCM's settings.json inside the container.
    _MCM_SETTINGS_DEFAULT = "/root/.config/mavlink-camera-manager/settings.json"
    # Host-side path for the device's original settings backup.
    _MCM_SETTINGS_BACKUP = "/tmp/_mcm_settings_device_backup.json"

    def backup_remote_settings(self) -> None:
        """Back up the device's current MCM settings.json so we can restore later.

        Copies from the container's default settings path to the Pi host's
        ``/tmp``.  Only runs once per deployer lifetime; subsequent calls
        are no-ops.
        """
        if self._settings_backed_up:
            return
        tmp = self._MCM_SETTINGS_BACKUP
        self._ssh(
            f"docker cp {self.container}:{self._MCM_SETTINGS_DEFAULT} {tmp} 2>/dev/null || true"
        )
        exists = self._ssh(f"test -f {tmp} && echo yes || echo no").strip()
        if exists == "yes":
            log.info(
                "Backed up remote MCM settings from %s:%s -> host:%s",
                self.container,
                self._MCM_SETTINGS_DEFAULT,
                tmp,
            )
        else:
            log.info(
                "No existing MCM settings.json found in %s -- nothing to back up.",
                self.container,
            )
        self._settings_backed_up = True

    def restore_remote_settings(self) -> None:
        """Restore the device's original MCM settings.json from the backup.

        Should be called after all tests complete to leave the device in
        the same state it was found in.
        """
        tmp = self._MCM_SETTINGS_BACKUP
        exists = self._ssh(f"test -f {tmp} && echo yes || echo no").strip()
        if exists != "yes":
            log.info("No settings backup found on host -- nothing to restore.")
            return
        self._ssh(f"docker cp {tmp} {self.container}:{self._MCM_SETTINGS_DEFAULT}")
        self._ssh(f"rm -f {tmp}")
        log.info(
            "Restored original MCM settings to %s:%s",
            self.container,
            self._MCM_SETTINGS_DEFAULT,
        )

    def _upload_mcm_config_to_host(self) -> None:
        """SCP the user-provided config file to the Pi host's /tmp.

        Only runs once per deployer lifetime; subsequent calls are no-ops.
        """
        if self.mcm_config is None or self._mcm_config_uploaded:
            return
        if not self.mcm_config.exists():
            raise FileNotFoundError(f"MCM config file not found: {self.mcm_config}")
        log.info("Uploading MCM config %s -> pi:/tmp/ ...", self.mcm_config)
        self._scp(str(self.mcm_config), "/tmp/_mcm_settings_original.json", upload=True)
        self._mcm_config_uploaded = True

    def _restore_mcm_config(self) -> None:
        """Copy the pristine config into the container's override path.

        Called before every MCM invocation so the process always starts from
        a pristine configuration (MCM may mutate its settings at runtime).
        """
        if self.mcm_config is None:
            return
        self._upload_mcm_config_to_host()
        self._ssh(
            f"docker cp /tmp/_mcm_settings_original.json"
            f" {self.container}:{self._MCM_SETTINGS_OVERRIDE}"
        )
        log.info(
            "Restored MCM settings in container: %s",
            self._MCM_SETTINGS_OVERRIDE,
        )

    @property
    def _active_settings_path(self) -> str:
        """Container path MCM is actually reading its settings from."""
        if self.mcm_config is not None:
            return self._MCM_SETTINGS_OVERRIDE
        return self._MCM_SETTINGS_DEFAULT

    def download_mcm_config(self, dest: Path) -> Path | None:
        """Download MCM's current settings.json from the container to *dest*.

        Returns the local path on success, or ``None`` if the file does not
        exist on the device.
        """
        dest.mkdir(parents=True, exist_ok=True)
        container_path = self._active_settings_path
        tmp_remote = "/tmp/_mcm_settings_download.json"
        self._ssh(f"docker cp {self.container}:{container_path} {tmp_remote} 2>/dev/null || true")
        exists = self._ssh(f"test -f {tmp_remote} && echo yes || echo no").strip()
        if exists != "yes":
            log.warning("MCM settings.json not found in container at %s.", container_path)
            return None

        local_path = dest / "settings.json"
        try:
            self._scp(str(local_path), tmp_remote, upload=False, check=True, timeout=15)
        except Exception as exc:
            log.warning("Failed to download MCM config: %s", exc)
            return None
        finally:
            self._ssh(f"rm -f {tmp_remote}")

        log.info("MCM config downloaded to %s", local_path)
        return local_path

    # ---- Log collection ------------------------------------------------------

    def collect_logs(self, dest: Path) -> None:
        """SCP container logs from the remote device."""
        dest.mkdir(parents=True, exist_ok=True)
        log_files = [
            ("/tmp/mcm_stdout.log", "mcm_stdout.log"),
            ("/tmp/mcm_debug.log", "mcm_debug.log"),
            ("/tmp/mcm_frame_log.jsonl", "mcm_frame_log.jsonl"),
        ]
        for remote_path, local_name in log_files:
            # First copy from container to host /tmp
            self._ssh(
                f"docker cp {self.container}:{remote_path} {remote_path} 2>/dev/null || true"
            )
            # Then SCP to local
            try:
                self._scp(
                    str(dest / local_name),
                    remote_path,
                    upload=False,
                    check=False,
                    timeout=15,
                )
            except Exception as exc:
                log.debug("Could not fetch %s: %s", remote_path, exc)

    def _ssh(self, cmd: str, timeout: int = 30) -> str:
        from ab_harness.config import ssh_run

        return ssh_run(self.host, self._ssh_config, cmd, timeout=timeout)


def create_deployer(
    mode: str,
    *,
    host: str,
    port: int = 6020,
    env_overrides: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    isolated: bool = True,
    blueos_tag: str | None = None,
    mcm_config: Path | None = None,
) -> LocalDeployment | RemoteDeployment:
    """Factory: create the right deployer based on mode.

    When *isolated* is ``True`` (the default) and mode is ``"remote"``, the
    deployer will stop all containers on the Pi and start a clean one with only
    a tmux server -- no BlueOS services.  Pass ``--no-isolated`` to inject MCM
    into the running ``blueos-core`` instead.
    """
    if mode == "local":
        return LocalDeployment(
            host=host,
            port=port,
            extra_args=extra_args,
            env_overrides=env_overrides,
            mcm_config=mcm_config,
        )
    elif mode == "remote":
        return RemoteDeployment(
            host=host,
            port=port,
            env_overrides=env_overrides,
            isolated=isolated,
            extra_args=extra_args,
            blueos_tag=blueos_tag,
            mcm_config=mcm_config,
        )
    else:
        raise ValueError(f"Unknown deploy mode: {mode!r}")
