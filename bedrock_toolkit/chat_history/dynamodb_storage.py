"""DynamoDB implementation of ChatHistoryStorage"""

from typing import Any

from botocore.exceptions import ClientError

from bedrock_toolkit.chat_history.base import ChatHistoryStorage
from bedrock_toolkit.common.dynamodb import DynamoDBBase
from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()


class DynamoDBChatHistoryStorage(DynamoDBBase, ChatHistoryStorage):
    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        delete_existing_table: bool = False,
        chat_id: str = "00000",
    ) -> None:
        super().__init__(table_name, region, delete_existing_table, chat_id)

    def _get_key_schema(self) -> list[dict[str, str]]:
        return [{"AttributeName": "chat_id", "KeyType": "HASH"}]

    def _get_attribute_definitions(self) -> list[dict[str, str]]:
        return [{"AttributeName": "chat_id", "AttributeType": "S"}]

    def get_messages(self) -> list[dict[str, Any]]:
        with self._table_lock():
            try:
                response = self.table.get_item(Key={"chat_id": self.chat_id})
                messages = response.get("Item", {}).get("messages", [])
                return self._convert_decimals_to_float(messages)
            except ClientError as e:
                logger.error(
                    f"Error retrieving messages for chat_id {self.chat_id}: {e}"
                )
                return []

    def save_messages(self, messages: list[dict[str, Any]]) -> None:
        with self._table_lock():
            try:
                decimal_messages = self._convert_floats_to_decimal(messages)
                self.table.put_item(
                    Item={"chat_id": self.chat_id, "messages": decimal_messages}
                )
            except ClientError as e:
                logger.error(f"Error saving messages for chat_id {self.chat_id}: {e}")
                raise

    def clear_messages(self) -> None:
        with self._table_lock():
            try:
                self.table.delete_item(Key={"chat_id": self.chat_id})
            except ClientError as e:
                logger.error(f"Error clearing messages for chat_id {self.chat_id}: {e}")
                raise
