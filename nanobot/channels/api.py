"""HTTP API channel for direct chat access (e.g. from Frappe UI)."""

import asyncio
import uuid

from aiohttp import web
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import GatewayConfig


class ApiChannel(BaseChannel):
    """
    HTTP API channel that exposes a REST endpoint for chat.

    Runs an aiohttp server on the gateway host:port.
    Frappe (or any HTTP client) sends POST /chat and gets the agent's
    response back synchronously.

    Flow:
        POST /chat  →  publish InboundMessage to bus
                    →  agent processes via AgentLoop
                    →  outbound dispatched to ApiChannel.send()
                    →  resolves the pending Future
                    →  HTTP response returned
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

        The chat_id on the OutboundMessage matches the request_id
        we stored when the HTTP request came in.
        """
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

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health requests."""
        return web.json_response({
            "status": "ok",
            "channel": "api",
            "running": self._running,
        })

