"""Factory for creating chat history storage objects."""

from typing import Any

from .base import ChatHistoryStorage
from .dynamodb_storage import DynamoDBChatHistoryStorage
from .local_storage import LocalChatHistoryStorage


def create_chat_history_storage(config: dict[str, Any]) -> ChatHistoryStorage:
    storage_type = config.get("type", "local")

    if storage_type == "local":
        return LocalChatHistoryStorage()
    elif storage_type == "dynamodb":
        table_name = config.get("table_name")
        region = config.get("region", "us-east-1")
        chat_id = config.get("chat_id", "00000")
        delete_existing_table = config.get("delete_existing_table", False)
        if not table_name:
            raise ValueError("DynamoDB table name must be provided in the config")
        storage = DynamoDBChatHistoryStorage(
            region=region,
            table_name=table_name,
            chat_id=chat_id,
            delete_existing_table=delete_existing_table,
        )
        return storage
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
