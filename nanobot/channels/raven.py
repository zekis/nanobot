"""Raven outbound channel — delivers messages directly to Raven via Frappe API."""

import aiohttp
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import RavenConfig


class RavenChannel(BaseChannel):
    """
    Outbound-only channel that posts messages directly to Raven's
    ``send_message`` API endpoint using Frappe API key auth.

    Registered as ``"raven"`` in the ChannelManager so that the
    ``message`` tool and startup greeting can target Raven conversations.

    It does NOT listen for inbound messages — those arrive via the
    API channel (``/chat`` endpoint) called by Frappe's Raven hook.
    """

    name = "raven"

    def __init__(self, config: RavenConfig, bus: MessageBus):
        super().__init__(config, bus)
        self._config = config

    async def start(self) -> None:
        """Mark as running (outbound-only, no listener)."""
        self._running = True
        if self._config.site_url and self._config.api_key:
            logger.info("Raven channel ready (outbound-only)")
        else:
            logger.warning("Raven channel: missing site_url or api_key — messages will be dropped")

    async def stop(self) -> None:
        self._running = False
        logger.info("Raven channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Deliver an outbound message to a Raven channel.

        Resolves chat_id:
        - "owner" or empty → config.owner_dm_channel
        - anything else → used as literal Raven Channel ID
        """
        if not self._config.site_url or not self._config.api_key:
            logger.warning("Raven channel: no credentials, dropping message")
            return

        content = msg.content
        if not content or not content.strip():
            return

        # Resolve target channel
        chat_id = msg.chat_id or ""
        if not chat_id or chat_id == "owner":
            channel_id = self._config.owner_dm_channel
        else:
            channel_id = chat_id

        if not channel_id:
            logger.warning("Raven channel: no channel_id resolved, dropping message")
            return

        site_url = self._config.site_url.rstrip("/")
        url = f"{site_url}/api/method/raven.api.raven_message.send_message"
        auth_header = f"token {self._config.api_key}:{self._config.api_secret}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={
                        "channel_id": channel_id,
                        "text": content,
                    },
                    headers={"Authorization": auth_header},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Raven message delivered (channel={channel_id})")
                    else:
                        body = await resp.text()
                        logger.warning(
                            f"Raven delivery failed ({resp.status}): {body[:200]}"
                        )
        except Exception as e:
            logger.error(f"Raven delivery error: {e}")
