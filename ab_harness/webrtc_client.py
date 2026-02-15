#!/usr/bin/env python3
"""
WebRTC client that connects to MCM's WebRTC frontend using Playwright.

Opens a headless browser, clicks through the UI to establish a WebRTC session,
holds it for a specified duration, then cleans up.

Can be run standalone::

    python -m ab_harness.webrtc_client --host 192.168.2.2 --duration 60

Or spawned as a subprocess by :class:`ab_harness.clients.WebRTCClient`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

log = logging.getLogger(__name__)


def run(
    host: str,
    port: int,
    duration: int,
    producer: str | None = None,
    log_file: str | None = None,
) -> None:
    """Connect to MCM WebRTC UI, hold a session, then clean up."""
    # Deferred import so the module can be imported without playwright
    # installed (the rest of the harness doesn't need it).
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(fh)

    url = f"http://{host}:{port}/webrtc"
    log.info("Connecting to %s, will hold session for %ds", url, duration)
    if producer:
        log.info("Will target producer matching: %r", producer)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--ignore-certificate-errors",
                "--autoplay-policy=no-user-gesture-required",
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
            ],
        )
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()

        # Navigate to WebRTC page
        log.info("Navigating to WebRTC page...")
        try:
            page.goto(url, timeout=30000, wait_until="networkidle")
        except PlaywrightTimeout:
            log.warning("Page load timed out, proceeding anyway...")

        # Click "Add consumer"
        log.info("Clicking 'Add consumer'...")
        add_consumer = page.locator("#add-consumer")
        add_consumer.wait_for(state="visible", timeout=10000)
        add_consumer.click()

        # Wait for streams/producers to appear
        log.info("Waiting for streams to appear...")
        first_add_session = page.locator("#add-session").first
        try:
            first_add_session.wait_for(state="visible", timeout=30000)
        except PlaywrightTimeout:
            log.error("No streams appeared within 30s. Aborting.")
            browser.close()
            sys.exit(1)

        # Each producer lives inside a div.stream; enumerate them
        stream_divs = page.locator("div.stream")
        count = stream_divs.count()
        log.info("Found %d producer(s):", count)
        available_names: list[str] = []
        for i in range(count):
            text = stream_divs.nth(i).text_content() or ""
            available_names.append(text)
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("Stream:"):
                    log.info("  [%d] %s", i, line)
                    break

        # Select the target producer
        if producer:
            needle = producer.lower()
            target_idx = None
            for i, name in enumerate(available_names):
                if needle in name.lower():
                    target_idx = i
                    break
            if target_idx is None:
                log.error(
                    "No producer matching %r found. Available producers (%d):",
                    producer,
                    count,
                )
                for i, name in enumerate(available_names):
                    for line in name.splitlines():
                        line = line.strip()
                        if line.startswith("Stream:"):
                            log.error("  [%d] %s", i, line)
                            break
                browser.close()
                sys.exit(1)
            log.info("Selected producer [%d] matching %r", target_idx, producer)
        else:
            target_idx = 0
            log.info("No --producer specified, using first producer [0]")

        # Click "Add Session" on the target producer
        target_div = stream_divs.nth(target_idx)
        add_session = target_div.locator("#add-session")
        log.info("Clicking 'Add Session' on producer [%d]...", target_idx)
        add_session.click()

        # Wait for "Status: Playing"
        log.info("Waiting for 'Status: Playing'...")
        try:
            page.locator("#session-status").first.wait_for(
                state="visible",
                timeout=30000,
            )
            text = ""
            for _ in range(30):
                text = page.locator("#session-status").first.text_content() or ""
                if "Playing" in text:
                    break
                time.sleep(1)
            log.info("WebRTC session status: %s", text.strip())
        except PlaywrightTimeout:
            log.warning("Session status element not found within 30s, holding anyway...")

        # Hold the session for the specified duration
        log.info("Holding session for %ds...", duration)
        start = time.monotonic()
        while time.monotonic() - start < duration:
            remaining = duration - (time.monotonic() - start)
            if remaining > 10:
                time.sleep(10)
                log.info("  %.0fs remaining...", remaining)
            else:
                time.sleep(max(0, remaining))

        # Clean up
        log.info("Cleaning up: removing consumer...")
        try:
            remove_consumer = page.locator("#remove-all-consumers")
            if remove_consumer.is_visible():
                remove_consumer.click()
            else:
                remove = page.locator("#remove-consumer")
                if remove.is_visible():
                    remove.click()
        except Exception as e:
            log.warning("Cleanup error (non-fatal): %s", e)

        time.sleep(2)
        browser.close()
        log.info("WebRTC client finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTC client for MCM testing")
    parser.add_argument("--host", default="blueos.local", help="MCM host")
    parser.add_argument("--port", type=int, default=6020, help="MCM HTTP port")
    parser.add_argument("--duration", type=int, default=60, help="Hold duration (s)")
    parser.add_argument(
        "--producer",
        default=None,
        help="Substring to match against producer stream name "
        '(e.g. "RadCam" or "UDP Stream 0"). '
        "If omitted, the first producer is used.",
    )
    parser.add_argument("--log-file", default=None, help="Log file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args.host, args.port, args.duration, args.producer, args.log_file)


if __name__ == "__main__":
    main()
