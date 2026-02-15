"""Agent loop: the core processing engine."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.gateway import load_gateway_tools
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        webhook_emitter: "WebhookEmitter | None" = None,
        memory_config: "MemoryConfig | None" = None,
        debug_config: "DebugConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig, MemoryConfig, DebugConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.webhook_emitter = webhook_emitter
        self.memory_config = memory_config
        self.debug_config = debug_config or DebugConfig()
        
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Gateway tools (server-side tools from skill_gateway.json)
        for gtool in load_gateway_tools(self.workspace):
            self.tools.register(gtool)
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        # Mark as final so the API channel knows to resolve
                        # the pending HTTP request.  Intermediate messages
                        # (debug tool calls, message-tool sends) are NOT
                        # marked and will be skipped by ApiChannel.send().
                        response.metadata["is_final"] = True
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}",
                        metadata={"is_final": True},
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        # Emit user_message event
        await self._emit_event("user_message",
            session_key=msg.session_key, channel=msg.channel,
            role="user", content=msg.content)

        # Get or create session — use explicit session_id from metadata
        # when available (e.g. API /chat passes session_id for conversation
        # continuity while using a unique request_id as chat_id for routing)
        effective_session_key = msg.metadata.get("session_id") or msg.session_key
        session = self.sessions.get_or_create(effective_session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)

        # Propagate message metadata to gateway tools so they can forward
        # opaque fields (e.g. context_token) back to the Frappe API.
        for tool in self.tools._tools.values():
            if hasattr(tool, "set_metadata"):
                tool.set_metadata(msg.metadata)

        # Retrieve relevant memories if enabled
        retrieved_memories = await self._retrieve_memories(msg.content)
        if retrieved_memories:
            await self._emit_event("memory_retrieval",
                session_key=msg.session_key, channel=msg.channel,
                role="system", content=retrieved_memories)

        # Build initial messages with structured context
        structured_ctx = session.get_structured_context()
        messages = self.context.build_messages(
            structured_context=structured_ctx,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            retrieved_memories=retrieved_memories,
        )
        
        # Agent loop — track cumulative token usage across iterations
        iteration = 0
        final_content = None
        total_prompt_tokens = 0
        total_completion_tokens = 0
        tool_actions: list[dict[str, str]] = []  # tool summaries for session storage

        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )

            # Write debug dump of the LLM call if enabled
            self._dump_llm_call(messages, response, iteration)

            # Accumulate token usage from every LLM call
            iter_usage = response.usage or {}
            total_prompt_tokens += iter_usage.get("prompt_tokens", 0)
            total_completion_tokens += iter_usage.get("completion_tokens", 0)

            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")

                    # Emit tool_call event
                    await self._emit_event("tool_call",
                        session_key=msg.session_key, channel=msg.channel,
                        role="tool", tool_name=tool_call.name,
                        tool_arguments=args_str, model=self.model)

                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                    # Capture tool summary for session storage
                    tool_actions.append({
                        "tool": tool_call.name,
                        "args_summary": self._summarize_args(tool_call.name, tool_call.arguments),
                        "outcome": self._summarize_outcome(result),
                    })

                    # Emit tool_result event
                    await self._emit_event("tool_result",
                        session_key=msg.session_key, channel=msg.channel,
                        role="tool", tool_name=tool_call.name,
                        content=result[:5000] if result else "")

                    # Send debug message to channel if enabled
                    await self._debug_tool_call(
                        msg.channel, msg.chat_id,
                        tool_call.name, args_str, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content

                # Emit assistant_message event with token usage
                usage = response.usage or {}
                await self._emit_event("assistant_message",
                    session_key=msg.session_key, channel=msg.channel,
                    role="assistant", content=final_content,
                    model=self.model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0))
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        # Save to session (clean content without usage footer)
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content, tool_actions=tool_actions)
        self.sessions.save(session)

        # Update task list via secondary LLM call + sync to Frappe
        frappe_channel = self._extract_frappe_channel(msg.metadata)
        if frappe_channel:
            await self._update_task_list(session, msg.content, final_content, tool_actions, frappe_channel)
            self.sessions.save(session)  # save again with updated metadata

        # Append token usage footer for display only (not saved to session)
        display_content = final_content
        if self.debug_config.show_token_usage:
            total_tokens = total_prompt_tokens + total_completion_tokens
            display_content += (
                f"\n\n---\n"
                f"📊 `{total_prompt_tokens}` in · `{total_completion_tokens}` out · "
                f"`{total_tokens}` total · `{iteration}` call{'s' if iteration != 1 else ''}"
            )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=display_content
        )
    
    async def _emit_event(self, event_type: str, **kwargs) -> None:
        """Emit a webhook event if a webhook emitter is configured.

        Fire-and-forget — never blocks the agent loop or raises exceptions.

        Args:
            event_type: One of user_message, assistant_message, tool_call, tool_result.
            **kwargs: Additional event fields.
        """
        if self.webhook_emitter:
            try:
                await self.webhook_emitter.emit(event_type, **kwargs)
            except Exception as e:
                logger.debug(f"Webhook emit failed: {e}")

    async def _debug_tool_call(
        self, channel: str, chat_id: str,
        tool_name: str, args_str: str, result: str | None
    ) -> None:
        """Send a debug message to the channel with tool call details.

        Only sends if debug.log_tool_calls is enabled. Fire-and-forget —
        never blocks the agent loop.

        Args:
            channel: Originating channel name.
            chat_id: Originating chat ID.
            tool_name: Name of the tool that was called.
            args_str: JSON-encoded tool arguments.
            result: Tool execution result (may be long).
        """
        if not self.debug_config.log_tool_calls:
            return

        # Truncate for readability
        args_preview = args_str[:300] + ("..." if len(args_str) > 300 else "")
        result_preview = (result or "")[:500]
        if result and len(result) > 500:
            result_preview += "..."

        debug_msg = (
            f"🔧 **Tool Call:** `{tool_name}`\n"
            f"**Args:** ```\n{args_preview}\n```\n"
            f"**Result:** ```\n{result_preview}\n```"
        )

        try:
            await self.bus.publish_outbound(OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=debug_msg,
                metadata={"is_debug": True},
            ))
        except Exception as e:
            logger.debug(f"Debug tool call message failed: {e}")

    async def _retrieve_memories(self, query: str) -> str:
        """Retrieve relevant memories from the Frappe memory API.

        Called before each LLM call to inject relevant memories into context.
        Returns an empty string if memory retrieval is disabled, the query is
        too short, or the API call fails.

        Args:
            query: The user's message text to search against.

        Returns:
            Formatted memories markdown string, or empty string.
        """
        if not self.memory_config or not self.memory_config.enabled:
            return ""

        if not self.memory_config.retrieval_url:
            return ""

        # Skip trivial messages
        stripped = query.strip()
        if len(stripped) < 5:
            logger.debug(f"Memory retrieval skipped: query too short ({len(stripped)} chars)")
            return ""

        logger.info(f"Memory retrieval: querying with '{stripped[:60]}...'")

        try:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.memory_config.retrieval_auth:
                headers["Authorization"] = self.memory_config.retrieval_auth

            payload = {
                "query": stripped,
                "nanobot_token": self.memory_config.nanobot_token,
                "top_k": self.memory_config.top_k,
            }

            async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
                resp = await client.post(
                    self.memory_config.retrieval_url,
                    json=payload,
                    headers=headers,
                )

            if resp.status_code != 200:
                logger.warning(f"Memory retrieval returned {resp.status_code}: {resp.text[:200]}")
                return ""

            data = resp.json()
            # Frappe wraps responses in {"message": ...}
            if "message" in data:
                data = data["message"]

            memories = data.get("memories", "")
            count = data.get("count", 0)
            if memories and count > 0:
                logger.info(f"Memory retrieval: injecting {count} memories into context")
            else:
                logger.info("Memory retrieval: no relevant memories found")
            return memories

        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return ""

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)
        
        # Build messages with structured context
        structured_ctx = session.get_structured_context()
        messages = self.context.build_messages(
            structured_context=structured_ctx,
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )

        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        tool_actions: list[dict[str, str]] = []
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                    tool_actions.append({
                        "tool": tool_call.name,
                        "args_summary": self._summarize_args(tool_call.name, tool_call.arguments),
                        "outcome": self._summarize_outcome(result),
                    })

                    # Send debug message to origin channel if enabled
                    await self._debug_tool_call(
                        origin_channel, origin_chat_id,
                        tool_call.name, args_str, result
                    )
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content, tool_actions=tool_actions)
        self.sessions.save(session)

        # Update task list via secondary LLM call + sync to Frappe
        frappe_channel = self._extract_frappe_channel(msg.metadata)
        if frappe_channel:
            await self._update_task_list(session, msg.content, final_content, tool_actions, frappe_channel)
            self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""

    @staticmethod
    def _extract_frappe_channel(metadata: dict) -> str | None:
        """Extract the Frappe Nanonet Channel name from message metadata.

        The session_id follows these patterns:
          nanonet-messaging:{channel}[:v{N}]
          nanonet-dm:{channel}
          nanonet-ctx:{channel}:{bot_name}

        Returns None if the session_id is missing or doesn't match.
        """
        session_id = (metadata or {}).get("session_id", "")
        if not session_id:
            return None

        for prefix in ("nanonet-messaging:", "nanonet-dm:", "nanonet-ctx:"):
            if session_id.startswith(prefix):
                remainder = session_id[len(prefix):]
                # Strip version suffix (:v2) or bot name suffix (:bot-name)
                # Channel names are Frappe hashes (alphanumeric, ~10 chars)
                parts = remainder.split(":")
                if parts and parts[0]:
                    return parts[0]

        return None

    @staticmethod
    def _summarize_args(tool_name: str, arguments: dict) -> str:
        """Human-readable summary of tool arguments (max 200 chars)."""
        if tool_name == "exec":
            return (arguments.get("command", "") or "")[:180]
        if tool_name in ("read_file", "write_file", "edit_file"):
            return (arguments.get("path", "") or "")[:200]
        if tool_name == "web_search":
            return (arguments.get("query", "") or "")[:200]
        if tool_name == "web_fetch":
            return (arguments.get("url", "") or "")[:200]
        if tool_name == "message":
            ch = arguments.get("channel", "")
            txt = (arguments.get("text", "") or "")[:100]
            return f"channel={ch} text={txt}"
        raw = json.dumps(arguments, ensure_ascii=False)
        return raw[:200] + "..." if len(raw) > 200 else raw

    @staticmethod
    def _summarize_outcome(result: str | None) -> str:
        """Short summary of tool result (max 300 chars)."""
        if not result:
            return "OK: (empty)"
        is_error = result.lower().startswith("error")
        prefix = "ERROR: " if is_error else "OK: "
        first_line = result.split("\n", 1)[0].strip()
        max_len = 300 - len(prefix)
        if len(first_line) > max_len:
            first_line = first_line[:max_len] + "..."
        return prefix + first_line

    async def _update_task_list(
        self,
        session: Session,
        user_message: str,
        assistant_response: str,
        tool_actions: list[dict[str, str]],
        channel: str,
    ) -> None:
        """Update the session task list via a secondary LLM call.

        1. Calls LLM to analyze the exchange and produce an updated task list
        2. Stores in session.metadata for context injection
        3. POSTs to Frappe API for messaging app display
        """
        current_tasks = session.metadata.get("task_list", [])

        if current_tasks:
            tasks_text = "\n".join(
                f"- [{t['status']}] {t['task']}" for t in current_tasks
            )
        else:
            tasks_text = "(no tasks yet)"

        if tool_actions:
            tools_text = "\n".join(
                f"- {a['tool']}({a['args_summary']}) -> {a['outcome']}"
                for a in tool_actions
            )
        else:
            tools_text = "(no tools used)"

        prompt = (
            "Update the task list based on this conversation exchange.\n\n"
            f"CURRENT TASK LIST:\n{tasks_text}\n\n"
            f"USER MESSAGE:\n{user_message}\n\n"
            f"TOOLS USED:\n{tools_text}\n\n"
            f"ASSISTANT RESPONSE:\n{assistant_response[:500]}\n\n"
            "Rules:\n"
            "- Add new tasks from the user's request (if any)\n"
            '- Mark tasks as "completed" if the assistant fulfilled them this turn\n'
            "- Keep existing incomplete tasks unchanged\n"
            "- Merge duplicate/similar tasks\n"
            "- Maximum 10 tasks total (drop oldest completed if over limit)\n"
            "- Each task: short description (under 80 chars)\n\n"
            'Return ONLY a JSON array. Each element: {"task": "description", "status": "pending|in_progress|completed"}\n'
            "No markdown, no explanation — just the JSON array."
        )

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a task tracker. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self.model,
            )

            text = response.content.strip()
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                task_list = json.loads(json_match.group())
                validated = []
                for t in task_list[:10]:
                    if isinstance(t, dict) and "task" in t and "status" in t:
                        validated.append({
                            "task": str(t["task"])[:80],
                            "status": t["status"] if t["status"] in ("pending", "in_progress", "completed") else "pending",
                        })
                session.metadata["task_list"] = validated
                await self._post_task_list_to_frappe(channel, validated)
        except Exception as e:
            logger.debug(f"Failed to update task list: {e}")

    async def _post_task_list_to_frappe(self, channel: str, task_list: list) -> None:
        """POST the task list to Frappe API for messaging app display."""
        if not self.webhook_emitter:
            return

        hooks = getattr(self.webhook_emitter, "config", None)
        if not hooks or not getattr(hooks, "webhook_url", None):
            return

        # Derive task list URL from webhook URL base
        # webhookUrl: https://site/api/method/nanonet.api.events.receive
        base_url = hooks.webhook_url.rsplit("/", 1)[0]
        url = f"{base_url}/nanonet.api.tasks.update_task_list"

        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
                await client.post(url, json={
                    "nanobot_token": hooks.nanobot_token,
                    "channel": channel,
                    "task_list": task_list,
                }, headers={
                    "Authorization": hooks.webhook_auth,
                })
        except Exception as e:
            logger.debug(f"Failed to sync task list to Frappe: {e}")

    def _dump_llm_call(self, messages: list[dict], response: Any, iteration: int) -> None:
        """Write the last LLM request/response to a debug file.

        Only writes when debug_config.log_llm_context is enabled.
        The file is written to .debug/last_llm_call.json in the workspace
        so the Frappe host can read it from the mounted volume.

        Overwrites on every call so it always reflects the most recent
        LLM interaction (including mid-loop tool iterations).
        """
        if not self.debug_config.log_llm_context:
            return

        try:
            from datetime import datetime

            debug_dir = self.workspace / ".debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Separate system prompt from history and user message
            system_prompt = None
            history = []
            user_message = None

            for m in messages:
                role = m.get("role", "")
                if role == "system":
                    system_prompt = m.get("content", "")
                elif role == "user":
                    user_message = m  # last one wins
                    history.append(m)
                else:
                    history.append(m)

            # Remove the system message and current user message from
            # history for clarity (they're shown separately)
            if history and history[-1] is user_message:
                history = history[:-1]

            dump = {
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "model": self.model,
                "system_prompt": system_prompt,
                "system_prompt_tokens": len(system_prompt.split()) if system_prompt else 0,
                "history_message_count": len(history),
                "history": history,
                "user_message": user_message.get("content", "") if user_message else "",
                "response": {
                    "content": response.content,
                    "has_tool_calls": response.has_tool_calls,
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in (response.tool_calls or [])
                    ],
                    "usage": response.usage,
                },
            }

            dump_path = debug_dir / "last_llm_call.json"
            dump_path.write_text(json.dumps(dump, indent=2, default=str))
        except Exception as e:
            logger.debug(f"Failed to write LLM debug dump: {e}")
