"""Gateway tools — proxy calls to Skillgate for server-side tool execution.

Each tool defined in the skillgate config becomes a native callable tool
so the LLM can invoke it directly. A companion CheckResultTool lets the
LLM poll for approval outcomes.
"""

import json
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.config.schema import SkillgateConfig


class GatewayTool(Tool):
    """Proxy tool that forwards calls to the Skillgate execute_tool API.

    The LLM calls this like any other tool. Under the hood it POSTs to
    ``/api/method/skillgate.api.gateway.execute_tool`` and returns the result.
    """

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict[str, Any],
        site_url: str,
        api_key: str,
        api_secret: str,
    ):
        self._name = tool_name
        self._description = tool_description
        self._parameters = tool_parameters or {"type": "object", "properties": {}}
        self._site_url = site_url.rstrip("/")
        self._auth = f"token {api_key}:{api_secret}"

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
        url = f"{self._site_url}/api/method/skillgate.api.gateway.execute_tool"
        payload = {
            "tool_name": self._name,
            "params": kwargs,
            "format": "json",
        }

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

            if message.get("pending_approval"):
                request_id = message.get("request_id", "")
                hint = message.get("result", "This tool requires approval.")
                return (
                    f"{hint}\n\n"
                    f"Approval pending — request_id: {request_id}\n"
                    f"Use the check_approval_result tool with this request_id to poll for the outcome."
                )

            if message.get("success"):
                return str(message.get("result", ""))

            return str(message.get("result") or message.get("error") or json.dumps(message))

        except httpx.HTTPStatusError as e:
            logger.warning(f"Gateway tool {self._name} HTTP error: {e}")
            return f"Error calling {self._name}: HTTP {e.response.status_code}"
        except Exception as e:
            logger.warning(f"Gateway tool {self._name} failed: {e}")
            return f"Error calling {self._name}: {e}"


class CheckApprovalResultTool(Tool):
    """Poll for the result of a pending Skillgate approval request."""

    def __init__(self, site_url: str, api_key: str, api_secret: str):
        self._site_url = site_url.rstrip("/")
        self._auth = f"token {api_key}:{api_secret}"

    @property
    def name(self) -> str:
        return "check_approval_result"

    @property
    def description(self) -> str:
        return (
            "Check the result of a pending tool approval request. "
            "Use the request_id returned by a tool that required approval."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request_id": {
                    "type": "string",
                    "description": "The request_id from the pending approval response.",
                },
            },
            "required": ["request_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        request_id = kwargs.get("request_id", "")
        if not request_id:
            return "Error: request_id is required."

        url = f"{self._site_url}/api/method/skillgate.api.gateway.check_result"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json={"request_id": request_id, "format": "json"},
                    headers={
                        "Authorization": self._auth,
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()

            data = resp.json()
            message = data.get("message", {})

            if isinstance(message, str):
                return message

            status = message.get("status", "")
            result = message.get("result")

            if status == "Pending":
                return f"Request {request_id} is still pending approval. Try again shortly."
            elif status == "Completed":
                return str(result) if result else "Tool executed successfully (no output)."
            elif status == "Approved":
                if result:
                    return str(result)
                return f"Request {request_id} was approved but result is not yet available. Try again."
            elif status in ("Denied", "Expired"):
                return f"Request {request_id} was {status.lower()}."
            else:
                return json.dumps(message)

        except httpx.HTTPStatusError as e:
            logger.warning(f"check_approval_result HTTP error: {e}")
            return f"Error checking result: HTTP {e.response.status_code}"
        except Exception as e:
            logger.warning(f"check_approval_result failed: {e}")
            return f"Error checking result: {e}"


def load_gateway_tools(config: SkillgateConfig) -> list[Tool]:
    """Load gateway tool definitions from skillgate config.

    Args:
        config: SkillgateConfig with tool definitions and credentials.

    Returns:
        List of Tool instances (GatewayTools + CheckApprovalResultTool).
    """
    if not config.enabled:
        return []

    if not config.url or not config.api_key or not config.api_secret:
        logger.warning("Skillgate config missing credentials — skipping gateway tools")
        return []

    tools: list[Tool] = []
    for tool_def in config.tools:
        name = tool_def.get("name")
        if not name:
            continue
        tools.append(GatewayTool(
            tool_name=name,
            tool_description=tool_def.get("description", ""),
            tool_parameters=tool_def.get("input_schema", {}),
            site_url=config.url,
            api_key=config.api_key,
            api_secret=config.api_secret,
        ))

    # Always add the approval polling tool when skillgate is enabled
    tools.append(CheckApprovalResultTool(
        site_url=config.url,
        api_key=config.api_key,
        api_secret=config.api_secret,
    ))

    logger.info(f"Loaded {len(tools)} gateway tools: {[t.name for t in tools]}")
    return tools
