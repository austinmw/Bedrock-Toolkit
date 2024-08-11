"""In-memory caching service for Bedrock Toolkit"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import boto3
import numpy as np
import spacy
from botocore.config import Config
from numpy import linalg

from bedrock_toolkit.cache_services.base import CacheService
from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()


class CacheEntry:
    def __init__(
        self, question: str, embedding: list[float], response: str, timestamp: float
    ):
        self.question = question
        self.embedding = embedding
        self.response = response
        self.timestamp = timestamp


class InMemoryCachingService(CacheService):
    def __init__(
        self,
        cache_size: int = 1000,
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        embedding_size: int = 256,
        cache_similarity_threshold: float = 0.95,
        cache_time_interval: int = 60 * 5,  # 5 minutes
        max_workers: int = 16,
        region: str = "us-east-1",
    ) -> None:
        self.cache: dict[str, CacheEntry] = {}
        self.cache_size = cache_size
        self.cache_similarity_threshold = cache_similarity_threshold
        self.cache_time_interval = cache_time_interval
        self.max_workers = max_workers

        # Botocore config with adaptive retry strategy
        config = Config(retries={"mode": "adaptive", "max_attempts": 10})

        # Initialize Bedrock client with custom config
        self.bedrock_client = boto3.client(
            "bedrock-runtime", region_name=region, config=config
        )
        self.embedding_model_id = embedding_model_id
        self.embedding_size = embedding_size

        # Initialize NER model
        self.nlp = spacy.load("en_core_web_sm")

    def get(self, question: str) -> Optional[str]:
        question_embedding = self.generate_embedding(question)
        similar_question = self.find_similar_question(question, question_embedding)

        if similar_question:
            return self.cache[similar_question].response
        return None

    def put(self, question: str, response: str):
        question_embedding = self.generate_embedding(question)
        self.add_to_cache(
            question, CacheEntry(question, question_embedding, response, time.time())
        )

    def add_to_cache(self, question: str, entry: CacheEntry):
        if len(self.cache) >= self.cache_size:
            self.evict_oldest_entry()
        self.cache[question] = entry

    def evict_oldest_entry(self):
        if self.cache:
            oldest_question = min(self.cache, key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_question]

    def find_similar_question(
        self, question: str, question_embedding: list[float]
    ) -> Optional[str]:
        query_entities = self.extract_entities(question)
        current_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_question = {
                executor.submit(
                    self.calculate_similarity,
                    question_embedding,
                    cached_question,
                    cached_data,
                    query_entities,
                    current_time,
                ): cached_question
                for cached_question, cached_data in self.cache.items()
            }

            max_similarity: float = -1
            most_similar_question: Optional[str] = None

            for future in as_completed(future_to_question):
                cached_question = future_to_question[future]
                try:
                    similarity = future.result()
                    if similarity is not None and similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_question = cached_question
                except Exception as exc:
                    logger.error(
                        f"Question {cached_question} generated an exception: {exc}"
                    )

        if most_similar_question:
            logger.info(f'Found similar question in cache: "{most_similar_question}"')

        return most_similar_question

    def calculate_similarity(
        self,
        question_embedding: list[float],
        cached_question: str,
        cached_data: CacheEntry,
        query_entities: list[Tuple[str, str]],
        current_time: float,
    ) -> Optional[float]:
        cached_embedding = cached_data.embedding
        cached_entities = self.extract_entities(cached_question)

        # Check if the cache entry is within the time interval
        if current_time - cached_data.timestamp > self.cache_time_interval:
            return None

        if self.entities_match(query_entities, cached_entities):
            similarity = self.cosine_similarity(question_embedding, cached_embedding)
            if similarity >= self.cache_similarity_threshold:
                return similarity
        return None

    def cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        return float(
            np.dot(embedding1, embedding2)
            / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        )

    def extract_entities(self, query: str) -> list[Tuple[str, str]]:
        doc = self.nlp(query)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def entities_match(
        self, entities1: list[Tuple[str, str]], entities2: list[Tuple[str, str]]
    ) -> bool:
        # Implement your entity matching logic here
        return set(entities1) == set(entities2)

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

            # Further normalize and reduce dimension if needed
            embedding = self.reduce_emb_dim(
                np.array(embedding), target_dim=self.embedding_size
            )

            return embedding.tolist()
        except boto3.exceptions.Boto3Error as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize the embedding vector.
        """
        return embedding / linalg.norm(embedding, axis=-1, keepdims=True)

    def reduce_emb_dim(
        self, embedding: np.ndarray, target_dim: int, normalize: bool = True
    ) -> np.ndarray:
        """
        Reduce the embedding dimension and optionally normalize it.
        """
        smaller_embedding = embedding[..., :target_dim]
        if normalize:
            smaller_embedding = self.normalize_embedding(smaller_embedding)
        return smaller_embedding
