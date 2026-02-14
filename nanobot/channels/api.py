"""HTTP API channel for direct chat access (e.g. from Frappe UI)."""

import asyncio
import uuid

from aiohttp import web
from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import GatewayConfig


class ApiChannel(BaseChannel):
    """
    HTTP API channel that exposes a REST endpoint for chat.

    Runs an aiohttp server on the gateway host:port.
    Frappe (or any HTTP client) sends POST /chat and gets the agent's
    response back synchronously.

    Endpoints:
        POST /chat    →  synchronous chat (new session per request)
        POST /notify  →  inject a message into an existing channel session
                         (fire-and-forget; agent response goes to the channel)
        GET  /health  →  health check
    """

    name = "api"

    def __init__(self, config: GatewayConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: GatewayConfig = config
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._pending: dict[str, asyncio.Future] = {}

    async def start(self) -> None:
        """Start the HTTP API server."""
        self._running = True

        self._app = web.Application()
        self._app.router.add_post("/chat", self._handle_chat)
        self._app.router.add_post("/notify", self._handle_notify)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(
            self._runner,
            host=self.config.host,
            port=self.config.port,
        )
        await site.start()

        logger.info(
            f"API channel listening on http://{self.config.host}:{self.config.port}"
        )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the HTTP API server."""
        self._running = False

        # Cancel any pending requests
        for req_id, future in self._pending.items():
            if not future.done():
                future.cancel()
        self._pending.clear()

        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._app = None

        logger.info("API channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """
        Resolve a pending chat request with the agent's response.

        Only messages marked with ``is_final`` in metadata actually
        resolve the pending HTTP request.  Intermediate messages
        (debug tool-call logs, ``message``-tool sends) are silently
        skipped so they don't consume the Future before the real
        response arrives.
        """
        if not msg.metadata.get("is_final"):
            # Intermediate message (debug, message-tool, etc.) — skip.
            # The Telegram channel would send these, but the API channel
            # is request/response and can only return one response.
            logger.debug(
                f"Skipping non-final message for chat_id={msg.chat_id} "
                f"(metadata={msg.metadata})"
            )
            return

        future = self._pending.pop(msg.chat_id, None)
        if future and not future.done():
            future.set_result(msg.content)
        else:
            logger.warning(f"No pending request for chat_id={msg.chat_id}")

    async def _handle_chat(self, request: web.Request) -> web.Response:
        """
        Handle POST /chat requests.

        Request JSON:
            {"message": "Hello!", "session_id": "optional-session-id"}

        Response JSON:
            {"response": "Agent's reply...", "session_id": "..."}
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": "Invalid JSON body"}, status=400
            )

        message = data.get("message", "").strip()
        if not message:
            return web.json_response(
                {"error": "message field is required"}, status=400
            )

        session_id = data.get("session_id", "api:default")
        request_id = f"api-{uuid.uuid4().hex[:12]}"

        # Create a Future that will be resolved when the agent responds
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future

        # Publish to the message bus
        await self._handle_message(
            sender_id=session_id,
            chat_id=request_id,
            content=message,
            metadata={"session_id": session_id, "request_id": request_id},
        )

        # Wait for the agent to respond (timeout after 120 seconds)
        try:
            response_text = await asyncio.wait_for(future, timeout=120.0)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            return web.json_response(
                {"error": "Agent response timed out"}, status=504
            )
        except asyncio.CancelledError:
            return web.json_response(
                {"error": "Request cancelled"}, status=499
            )

        return web.json_response({
            "response": response_text,
            "session_id": session_id,
        })

    async def _handle_notify(self, request: web.Request) -> web.Response:
        """
        Handle POST /notify requests — inject a message into an existing
        channel session.

        Unlike /chat (which creates a new one-off API session), /notify
        publishes an InboundMessage using the caller-specified channel and
        chat_id.  This means the agent loads the **same session history**
        as the target channel (e.g. ``telegram:123456789``), has full
        conversation context, and the response is automatically routed
        to that channel by the ChannelManager.

        Fire-and-forget: returns immediately after publishing to the bus.
        The agent processes asynchronously and the response goes to the
        target channel (e.g. Telegram), not back in the HTTP response.

        Request JSON:
            {
                "message": "Notification text...",
                "channel": "telegram",
                "chat_id": "123456789"
            }

        Response JSON:
            {"status": "ok"}
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": "Invalid JSON body"}, status=400
            )

        message = data.get("message", "").strip()
        if not message:
            return web.json_response(
                {"error": "message field is required"}, status=400
            )

        channel = data.get("channel", "").strip()
        chat_id = data.get("chat_id", "").strip()
        if not channel or not chat_id:
            return web.json_response(
                {"error": "channel and chat_id fields are required"}, status=400
            )

        # Publish directly to the bus with the target channel/chat_id.
        # This bypasses the API channel's own session — the agent will
        # load the session for f"{channel}:{chat_id}" (e.g. telegram:123456789)
        # and the response OutboundMessage will be routed to that channel.
        msg = InboundMessage(
            channel=channel,
            sender_id="system",
            chat_id=chat_id,
            content=message,
            metadata={"source": "notify"},
        )
        await self.bus.publish_inbound(msg)

        logger.info(
            f"[notify] Injected message into {channel}:{chat_id} "
            f"({len(message)} chars)"
        )

        return web.json_response({"status": "ok"})

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health requests."""
        return web.json_response({
            "status": "ok",
            "channel": "api",
            "running": self._running,
        })

