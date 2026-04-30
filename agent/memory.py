from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Message:
    role: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationMemory:
    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def history(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
