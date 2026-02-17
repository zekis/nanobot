"""Session management for conversation history."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """
    A conversation session.
    
    Stores messages in JSONL format for easy reading and persistence.
    """
    
    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """
        Get message history for LLM context.
        
        Args:
            max_messages: Maximum messages to return.
        
        Returns:
            List of messages in LLM format.
        """
        # Get recent messages
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        
        # Convert to LLM format (just role and content)
        return [{"role": m["role"], "content": m["content"]} for m in recent]
    
    def get_structured_context(
        self,
        min_pairs: int = 3,
        max_pairs: int = 20,
        recency_minutes: int = 30,
        max_tool_entries: int = 30,
    ) -> dict[str, Any]:
        """Build structured context instead of raw history dump.

        Instead of returning all messages as raw history, decomposes the
        session into three parts:
        - recent_pairs: recent user+assistant exchanges as LLM-format dicts
        - task_list: LLM-maintained task list from session metadata
        - tool_log: chronological tool action summaries from older messages

        Pair selection: always include the last ``min_pairs`` exchanges,
        plus any additional exchanges from the last ``recency_minutes``,
        up to ``max_pairs`` total.  This keeps recent conversations
        coherent while avoiding unbounded history growth.

        This prevents history contamination where older assistant messages
        that describe tool actions (instead of calling them) teach the model
        bad behavior via in-context learning.

        Args:
            min_pairs: Minimum number of user+assistant pairs (always included).
            max_pairs: Maximum pairs to include (hard cap).
            recency_minutes: Include pairs from this many minutes ago.
            max_tool_entries: Maximum tool action entries to include.

        Returns:
            Dict with recent_pairs, task_list, and tool_log keys.
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(minutes=recency_minutes)

        # --- Collect candidate pairs walking backwards ---
        all_pairs: list[tuple[int, int, dict, dict]] = []  # (user_idx, asst_idx, user_msg, asst_msg)
        idx = len(self.messages) - 1
        while idx >= 0 and len(all_pairs) < max_pairs:
            m = self.messages[idx]
            if m["role"] == "assistant" and idx > 0 and self.messages[idx - 1]["role"] == "user":
                all_pairs.append((idx - 1, idx, self.messages[idx - 1], m))
                idx -= 2
            else:
                idx -= 1

        # Determine how many to keep: at least min_pairs, plus any within the time window
        keep = min(min_pairs, len(all_pairs))
        for i in range(keep, len(all_pairs)):
            ts = all_pairs[i][3].get("timestamp") or all_pairs[i][2].get("timestamp")
            if ts:
                try:
                    msg_time = datetime.fromisoformat(ts)
                    if msg_time >= cutoff:
                        keep = i + 1
                        continue
                except (ValueError, TypeError):
                    pass
            break  # Once we hit a pair outside the window, stop

        selected = all_pairs[:keep]
        selected.reverse()  # chronological order

        recent: list[dict[str, Any]] = []
        for _ui, _ai, user_m, asst_m in selected:
            recent.append({"role": "user", "content": user_m["content"]})
            recent.append({"role": "assistant", "content": asst_m["content"]})

        # Indices of messages included in recent pairs (to exclude from older context)
        recent_indices: set[int] = set()
        for ui, ai, _u, _a in selected:
            recent_indices.add(ui)
            recent_indices.add(ai)

        # --- Task list (LLM-maintained, from session metadata) ---
        task_list = self.metadata.get("task_list", [])

        # --- Tool log (from tool_actions stored on older assistant messages) ---
        tool_entries: list[dict[str, str]] = []
        for i, m in enumerate(self.messages):
            if i in recent_indices:
                continue
            if m["role"] != "assistant":
                continue
            for action in m.get("tool_actions", []):
                tool_entries.append({
                    "timestamp": m.get("timestamp", ""),
                    "tool": action.get("tool", "unknown"),
                    "args_summary": action.get("args_summary", ""),
                    "outcome": action.get("outcome", ""),
                })
        tool_entries = tool_entries[-max_tool_entries:]

        return {
            "recent_pairs": recent,
            "task_list": task_list,
            "tool_log": tool_entries,
        }

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.metadata = {}
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.
    
    Sessions are stored as JSONL files in the sessions directory.
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(Path.home() / ".nanobot" / "sessions")
        self._cache: dict[str, Session] = {}
    
    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"
    
    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            key: Session key (usually channel:chat_id).
        
        Returns:
            The session.
        """
        # Check cache
        if key in self._cache:
            return self._cache[key]
        
        # Try to load from disk
        session = self._load(key)
        if session is None:
            session = Session(key=key)
        
        self._cache[key] = session
        return session
    
    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        
        if not path.exists():
            return None
        
        try:
            messages = []
            metadata = {}
            created_at = None
            
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                    else:
                        messages.append(data)
            
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)
        
        with open(path, "w") as f:
            # Write metadata first
            metadata_line = {
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata
            }
            f.write(json.dumps(metadata_line) + "\n")
            
            # Write messages
            for msg in session.messages:
                f.write(json.dumps(msg) + "\n")
        
        self._cache[session.key] = session
    
    def delete(self, key: str) -> bool:
        """
        Delete a session.
        
        Args:
            key: Session key.
        
        Returns:
            True if deleted, False if not found.
        """
        # Remove from cache
        self._cache.pop(key, None)
        
        # Remove file
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session info dicts.
        """
        sessions = []
        
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            sessions.append({
                                "key": path.stem.replace("_", ":"),
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
