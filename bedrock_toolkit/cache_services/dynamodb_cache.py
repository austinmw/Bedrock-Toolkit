import json
import time
from decimal import Decimal

import boto3
import numpy as np
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

from bedrock_toolkit.cache_services.base import CacheService
from bedrock_toolkit.common.dynamodb import DynamoDBBase
from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()


class DynamoDBCachingService(DynamoDBBase, CacheService):
    def __init__(
        self,
        table_name: str = "bedrock-cache",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        embedding_size: int = 256,
        cache_similarity_threshold: float = 0.95,
        cache_time_interval: int = 60 * 5,
        max_cache_size: int = 1000,
        region: str = "us-east-1",
        delete_existing_table: bool = False,
        chat_id: str = "00000",
    ) -> None:
        super().__init__(table_name, region, delete_existing_table, chat_id)
        self.embedding_model_id = embedding_model_id
        self.embedding_size = embedding_size
        self.cache_similarity_threshold = cache_similarity_threshold
        self.cache_time_interval = cache_time_interval
        self.max_cache_size = max_cache_size
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    def _get_key_schema(self):
        return [
            {"AttributeName": "question", "KeyType": "HASH"},
            {"AttributeName": "chat_id", "KeyType": "RANGE"},
        ]

    def _get_attribute_definitions(self):
        return [
            {"AttributeName": "question", "AttributeType": "S"},
            {"AttributeName": "chat_id", "AttributeType": "S"},
        ]

    def get(self, question: str) -> str | None:
        question_embedding = self.generate_embedding(question)
        similar_question = self.find_similar_question(question, question_embedding)

        if similar_question:
            key = {"question": similar_question, "chat_id": self.chat_id}
            response = self.table.get_item(Key=key)
            if "Item" in response:
                logger.info(
                    f"Found similar question in cache: '{similar_question}' for input: '{question}' with chat_id: '{self.chat_id}'"
                )
                return response["Item"]["response"]
        return None

    def put(self, question: str, response: str) -> None:
        question_embedding = self.generate_embedding(question)
        current_time = int(time.time())

        self.evict_old_entries()

        embedding_decimal = [Decimal(str(x)) for x in question_embedding]

        item = {
            "question": question,
            "embedding": embedding_decimal,
            "response": response,
            "timestamp": current_time,
            "chat_id": self.chat_id,
        }

        self.table.put_item(Item=item)

    def find_similar_question(
        self, question: str, question_embedding: list[float]
    ) -> str | None:
        current_time = int(time.time())
        cutoff_time = current_time - self.cache_time_interval

        filter_expression = Attr("timestamp").gt(cutoff_time) & Attr("chat_id").eq(
            self.chat_id
        )

        # Scan the table for recent items
        response = self.table.scan(FilterExpression=filter_expression)
        items = response.get("Items", [])

        max_similarity: float = -1
        most_similar_question = None

        for item in items:
            cached_embedding = [float(x) for x in item["embedding"]]
            similarity = self.cosine_similarity(question_embedding, cached_embedding)
            if (
                similarity >= self.cache_similarity_threshold
                and similarity > max_similarity
            ):
                max_similarity = similarity
                most_similar_question = item["question"]

        if most_similar_question:
            logger.debug(
                f"Found similar question in cache: '{most_similar_question}' for input: '{question}' with chat_id: '{self.chat_id}'"
            )
        else:
            logger.debug(
                f"No similar question found in cache for input: '{question}' with chat_id: '{self.chat_id}'"
            )

        return most_similar_question

    def evict_old_entries(self):
        # Count the total number of items
        response = self.table.scan(Select="COUNT")
        total_items = response["Count"]

        if total_items >= self.max_cache_size:
            # Calculate how many items to remove
            items_to_remove = (
                total_items - self.max_cache_size + 1
            )  # +1 to make room for the new item
            logger.debug(
                f"Cache size limit reached. Evicting {items_to_remove} oldest entries."
            )

            # Scan for the oldest items
            response = self.table.scan(
                ProjectionExpression="question,#ts",
                ExpressionAttributeNames={"#ts": "timestamp"},
                Select="SPECIFIC_ATTRIBUTES",
            )
            items = response["Items"]

            # Sort items by timestamp and select the oldest ones
            items_to_delete = sorted(items, key=lambda x: x["timestamp"])[
                :items_to_remove
            ]

            # Delete the oldest items
            with self.table.batch_writer() as batch:
                for item in items_to_delete:
                    batch.delete_item(
                        Key={"question": item["question"], "chat_id": item["chat_id"]}
                    )
                    logger.debug(f"Evicted cache entry: '{item['question']}'")

            logger.debug(f"Evicted {items_to_remove} oldest entries from the cache.")

    def cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        return float(
            np.dot(embedding1, embedding2)
            / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        )

    def generate_embedding(self, text: str) -> list[float]:
        logger.debug(
            f"Generating embeddings with Amazon Titan Text Embeddings V2 model {self.embedding_model_id}"
        )

        body = json.dumps(
            {"inputText": text, "dimensions": self.embedding_size, "normalize": True}
        )

        try:
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=self.embedding_model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            embedding = response_body["embedding"]

            return embedding
        except ClientError as e:
            logger.error(f"Error generating embedding: {e}")
            return []
