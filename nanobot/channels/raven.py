"""Raven outbound channel — delivers message-tool output to Raven via Frappe API."""

import json
from pathlib import Path

import aiohttp
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


class RavenChannel(BaseChannel):
    """
    Outbound-only channel that posts messages to Raven via the
    ``deliver_bot_message`` Frappe API endpoint.

    This channel is registered as ``"raven"`` in the ChannelManager so
    that the ``message`` tool can target Raven conversations. It reads
    credentials from ``skill_gateway.json`` in the workspace.

    It does NOT listen for inbound messages — those arrive via the
    API channel (``/chat`` endpoint) called by Frappe's Raven hook.
    """

    name = "raven"

    def __init__(self, workspace_path: Path, bus: MessageBus):
        # BaseChannel expects (config, bus) — pass workspace_path as config
        super().__init__(workspace_path, bus)
        self._workspace_path = workspace_path
        self._creds: dict | None = None

    def _load_creds(self) -> dict | None:
        """Read skill_gateway.json for Frappe API credentials."""
        gw_path = self._workspace_path / "skill_gateway.json"
        if not gw_path.exists():
            return None
        try:
            data = json.loads(gw_path.read_text(encoding="utf-8"))
            url = data.get("url", "")
            api_key = data.get("api_key", "")
            api_secret = data.get("api_secret", "")
            nanobot_token = data.get("nanobot_token", "")
            if all([url, api_key, api_secret, nanobot_token]):
                return {
                    "url": url.rstrip("/"),
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "nanobot_token": nanobot_token,
                }
        except Exception as e:
            logger.warning(f"Failed to read skill_gateway.json: {e}")
        return None

    async def start(self) -> None:
        """Load credentials and mark as running (outbound-only, no listener)."""
        self._creds = self._load_creds()
        self._running = True
        if self._creds:
            logger.info("Raven channel ready (outbound-only)")
        else:
            logger.warning("Raven channel: no gateway credentials — messages will be dropped")

    async def stop(self) -> None:
        self._running = False
        logger.info("Raven channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Deliver an outbound message to Raven via deliver_bot_message.

        If ``chat_id`` is a specific Raven channel ID (not "owner"),
        a ``channel:<id>`` directive is appended so deliver_bot_message
        routes to the correct channel.
        """
        if not self._creds:
            self._creds = self._load_creds()
            if not self._creds:
                logger.warning("Raven channel: no credentials, dropping message")
                return

        content = msg.content
        if not content or not content.strip():
            return

        # If chat_id is a specific Raven channel (not "owner"), add a
        # channel directive so deliver_bot_message routes correctly.
        chat_id = msg.chat_id or ""
        if chat_id and chat_id != "owner":
            content = f"{content}\n\nchannel:{chat_id}"

        url = f"{self._creds['url']}/api/method/nanonet.api.messaging.deliver_bot_message"
        auth_header = f"token {self._creds['api_key']}:{self._creds['api_secret']}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={
                        "nanobot_token": self._creds["nanobot_token"],
                        "content": content,
                        "notice_type": "message",
                    },
                    headers={"Authorization": auth_header},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Raven message delivered (chat_id={chat_id})")
                    else:
                        body = await resp.text()
                        logger.warning(
                            f"Raven delivery failed ({resp.status}): {body[:200]}"
                        )
        except Exception as e:
            logger.error(f"Raven delivery error: {e}")
