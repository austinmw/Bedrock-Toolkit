"""Factory for creating cache services based on the configuration"""

from typing import Any, Dict

from .base import CacheService
from .dynamodb_cache import DynamoDBCachingService
from .in_memory_cache import InMemoryCachingService


def create_cache_service(config: Dict[str, Any]) -> CacheService:
    cache_type = config.get("type", "in_memory")

    # Common configuration options
    embedding_model_id = config.get(
        "embedding_model_id", "amazon.titan-embed-text-v2:0"
    )
    embedding_size = config.get("embedding_size", 256)
    cache_similarity_threshold = config.get("cache_similarity_threshold", 0.95)
    cache_time_interval = config.get("cache_time_interval", 60 * 5)  # 5 minutes
    region = config.get("region", "us-east-1")
    chat_id = config.get("chat_id", "00000")  # Default chat_id value

    if cache_type == "in_memory":
        cache_size = config.get("cache_size", 1000)
        max_workers = config.get("max_workers", 16)
        return InMemoryCachingService(
            cache_size=cache_size,
            embedding_model_id=embedding_model_id,
            embedding_size=embedding_size,
            cache_similarity_threshold=cache_similarity_threshold,
            cache_time_interval=cache_time_interval,
            max_workers=max_workers,
            region=region,
        )
    elif cache_type == "dynamodb":
        table_name = config.get("table_name")
        max_cache_size = config.get("max_cache_size", 1000)
        delete_existing_table = config.get("delete_existing_table", False)  # New option

        if not table_name:
            raise ValueError("DynamoDB table name must be provided in the config")

        return DynamoDBCachingService(
            table_name=table_name,
            embedding_model_id=embedding_model_id,
            embedding_size=embedding_size,
            cache_similarity_threshold=cache_similarity_threshold,
            cache_time_interval=cache_time_interval,
            max_cache_size=max_cache_size,
            region=region,
            delete_existing_table=delete_existing_table,  # Pass the new option to the service
            chat_id=chat_id,  # Pass the chat_id to the DynamoDBCachingService
        )
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")
