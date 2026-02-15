"""Optional stream clients: WebRTC (Playwright) and RTSP (GStreamer)."""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import sys
from io import TextIOWrapper
from pathlib import Path

log = logging.getLogger(__name__)

# Bundled WebRTC client script (lives next to this file).
_WEBRTC_CLIENT = Path(__file__).parent / "webrtc_client.py"


class WebRTCClient:
    """Manage a webrtc_client.py subprocess."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 6020,
        producer: str | None = None,
        duration: int = 60,
        log_dir: Path | None = None,
        log_suffix: str = "",
    ):
        self.host = host
        self.port = port
        self.producer = producer
        self.duration = duration
        self.log_dir = log_dir
        self.log_suffix = log_suffix
        self._process: subprocess.Popen[str] | None = None
        self._log_fh: TextIOWrapper | None = None
        self.log_path: Path | None = None

    def start(self) -> None:
        cmd = [
            sys.executable,
            str(_WEBRTC_CLIENT),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--duration",
            str(self.duration),
        ]
        if self.producer:
            cmd.extend(["--producer", self.producer])

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.log_dir / f"webrtc_client{self.log_suffix}.log"
            self._log_fh = open(self.log_path, "w")  # noqa: SIM115

        log.info("Starting WebRTC client: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self._process is None:
            return
        rc = self._process.poll()
        if rc is None:
            log.info(
                "Stopping WebRTC client (pid=%d, producer=%s)...", self._process.pid, self.producer
            )
            self._process.send_signal(signal.SIGTERM)
            try:
                rc = self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                rc = self._process.wait()
        if rc and rc != -signal.SIGTERM:
            log.warning(
                "WebRTC client (producer=%s) exited with code %d",
                self.producer,
                rc,
            )
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def wait(self) -> int:
        """Wait for the client to finish and return exit code."""
        if self._process:
            return self._process.wait()
        return 0

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None


class RTSPClient:
    """Manage a GStreamer RTSP consumer subprocess."""

    def __init__(
        self,
        *,
        host: str,
        rtsp_port: int = 8554,
        path: str = "video_stream_0",
        log_dir: Path | None = None,
        log_suffix: str = "",
    ):
        self.host = host
        self.rtsp_port = rtsp_port
        self.path = path.lstrip("/")
        self.log_dir = log_dir
        self.log_suffix = log_suffix
        self._process: subprocess.Popen[str] | None = None
        self._log_fh: TextIOWrapper | None = None
        self.log_path: Path | None = None

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://{self.host}:{self.rtsp_port}/{self.path}"

    def start(self) -> None:
        gst_launch = shutil.which("gst-launch-1.0")
        if not gst_launch:
            raise FileNotFoundError("gst-launch-1.0 not found. Install GStreamer.")

        pipeline = f"rtspsrc location={self.rtsp_url} latency=0 ! decodebin ! fakesink sync=false"
        cmd = [gst_launch, *pipeline.split()]

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.log_dir / f"rtsp_client{self.log_suffix}.log"
            self._log_fh = open(self.log_path, "w")  # noqa: SIM115

        log.info("Starting RTSP client: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self._process is None:
            return
        rc = self._process.poll()
        if rc is None:
            log.info("Stopping RTSP client (pid=%d, path=%s)...", self._process.pid, self.path)
            self._process.send_signal(signal.SIGTERM)
            try:
                rc = self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                rc = self._process.wait()
        if rc and rc != -signal.SIGTERM:
            log.warning(
                "RTSP client (path=%s) exited with code %d",
                self.path,
                rc,
            )
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None
