"""Gateway tool — thin proxy that calls server-side tools via the Frappe API.

Each tool defined in ``skill_gateway.json`` becomes a native callable tool
so the LLM can invoke it directly instead of constructing curl commands.
"""

import json
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool


class GatewayTool(Tool):
    """Proxy tool that forwards calls to the nanonet gateway API.

    The LLM calls this like any other tool. Under the hood it POSTs to
    ``/api/method/nanonet.api.skills.execute_tool`` and returns the result.
    """

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict[str, Any],
        gateway_url: str,
        api_key: str,
        api_secret: str,
        nanobot_token: str,
    ):
        self._name = tool_name
        self._description = tool_description
        self._parameters = tool_parameters or {"type": "object", "properties": {}}
        self._gateway_url = gateway_url.rstrip("/")
        self._auth = f"token {api_key}:{api_secret}"
        self._nanobot_token = nanobot_token
        self._metadata: dict[str, Any] = {}

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Store per-request metadata from the inbound message.

        Called by the agent loop before processing each message so that
        gateway tool calls can forward opaque metadata (e.g. context_token
        for race-safe approval routing) back to the Frappe execute_tool
        endpoint.
        """
        self._metadata = metadata or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        url = f"{self._gateway_url}/api/method/nanonet.api.skills.execute_tool"
        payload = {
            "tool_name": self._name,
            "params": kwargs,
            "nanobot_token": self._nanobot_token,
        }
        # Thread context_token from message metadata for race-safe approval routing
        context_token = self._metadata.get("context_token")
        if context_token:
            payload["context_token"] = context_token

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": self._auth,
                        "Content-Type": "application/json",
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()

            data = resp.json()
            message = data.get("message", {})

            if isinstance(message, str):
                return message

            if message.get("success"):
                return str(message.get("result", ""))

            # Tool returned an error or pending approval
            return str(message.get("result") or message.get("error") or json.dumps(message))

        except httpx.HTTPStatusError as e:
            logger.warning(f"Gateway tool {self._name} HTTP error: {e}")
            return f"Error calling {self._name}: HTTP {e.response.status_code}"
        except Exception as e:
            logger.warning(f"Gateway tool {self._name} failed: {e}")
            return f"Error calling {self._name}: {e}"


def load_gateway_tools(workspace: Path) -> list[GatewayTool]:
    """Load gateway tool definitions from ``skill_gateway.json``.

    Args:
        workspace: Path to the nanobot workspace directory.

    Returns:
        List of GatewayTool instances ready for registration.
    """
    gateway_path = workspace / "skill_gateway.json"
    if not gateway_path.exists():
        logger.debug("No skill_gateway.json found — skipping gateway tools")
        return []

    try:
        data = json.loads(gateway_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read skill_gateway.json: {e}")
        return []

    gateway_url = data.get("url", "")
    api_key = data.get("api_key", "")
    api_secret = data.get("api_secret", "")
    nanobot_token = data.get("nanobot_token", "")

    if not all([gateway_url, api_key, api_secret, nanobot_token]):
        logger.warning("skill_gateway.json missing credentials — skipping gateway tools")
        return []

    tools = []
    for tool_def in data.get("tools", []):
        name = tool_def.get("name")
        if not name:
            continue
        tools.append(GatewayTool(
            tool_name=name,
            tool_description=tool_def.get("description", ""),
            tool_parameters=tool_def.get("input_schema", {}),
            gateway_url=gateway_url,
            api_key=api_key,
            api_secret=api_secret,
            nanobot_token=nanobot_token,
        ))

    logger.info(f"Loaded {len(tools)} gateway tools: {[t.name for t in tools]}")
    return tools
