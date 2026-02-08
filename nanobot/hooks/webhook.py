"""Webhook emitter for nanobot event capture.

Async fire-and-forget HTTP client that POSTs events to a configured
webhook URL (typically the Frappe events API endpoint).

Events are emitted without blocking the agent loop — failures are
logged at warning level so they're visible in normal logs.
"""

import asyncio
from datetime import datetime, timezone

import httpx
from loguru import logger


class WebhookEmitter:
    """Async fire-and-forget HTTP client for webhook events.

    Keeps strong references to background tasks to prevent garbage
    collection before completion (a common asyncio pitfall).

    Args:
        url: The webhook URL to POST events to.
        auth: Authorization header value (e.g. "token api_key:api_secret").
        nanobot_token: The nanobot's unique identity token.
    """

    def __init__(self, url: str, auth: str = "", nanobot_token: str = ""):
        self.url = url
        self.auth = auth
        self.nanobot_token = nanobot_token
        self._client: httpx.AsyncClient | None = None
        self._background_tasks: set[asyncio.Task] = set()

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        return self._client

    async def emit(self, event_type: str, **kwargs) -> None:
        """Emit an event to the webhook URL (fire-and-forget).

        Creates an asyncio task to POST the event so the caller
        is never blocked. The task reference is kept in a set to
        prevent garbage collection before completion.

        Args:
            event_type: One of user_message, assistant_message, tool_call, tool_result.
            **kwargs: Additional event fields (session_key, channel, role, content, etc.)
        """
        if not self.url:
            return

        payload = {
            "event_type": event_type,
            "nanobot_token": self.nanobot_token,
            "event_timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }

        try:
            task = asyncio.create_task(self._post(payload))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except Exception as e:
            logger.warning(f"Failed to schedule webhook event: {e}")

    async def _post(self, payload: dict) -> None:
        """POST event payload to the webhook URL.

        Args:
            payload: The event data to send.
        """
        event_type = payload.get("event_type", "unknown")
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth:
                headers["Authorization"] = self.auth
            client = self._get_client()
            resp = await client.post(self.url, json=payload, headers=headers)
            if resp.status_code >= 300:
                logger.warning(f"Webhook POST {event_type} returned {resp.status_code}: {resp.text[:200]}")
            else:
                logger.debug(f"Webhook event {event_type} sent OK ({resp.status_code})")
        except Exception as e:
            logger.warning(f"Webhook POST {event_type} failed: {e}")

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

