""" Local storage for chat history. """
from typing import Any

from bedrock_toolkit.chat_history.base import ChatHistoryStorage


class LocalChatHistoryStorage(ChatHistoryStorage):
    def __init__(self) -> None:
        self.messages: list = []

    def get_messages(self) -> list[dict[str, Any]]:
        return self.messages

    def save_messages(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages

    def clear_messages(self) -> None:
        self.messages = []
