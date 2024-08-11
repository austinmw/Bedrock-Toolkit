""" This module contains classes that append additional context to user input. """

from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any

import boto3
from botocore.exceptions import ClientError
import numpy as np

from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()

class ContextAppender:
    def __init__(self, additional_context: str) -> None:
        self.additional_context = additional_context
        logger.info("Initialized ContextAppender with additional context.")

    def append_context(self, user_input: str) -> str:
        logger.debug(f"Appending context to user input: {user_input}")
        return f"{user_input}\n\nAdditional Context: {self.additional_context}"

class RAGContextAppender(ContextAppender):
    def __init__(self, retriever: Any) -> None:
        super().__init__(retriever)
        logger.info("Initialized RAGContextAppender with retriever.")

    def append_context(self, user_input: str) -> str:
        logger.debug("Retrieving relevant context for the user input.")
        retrieved_context = self.retriever.get_relevant_context(user_input)
        logger.debug(f"Retrieved context: {retrieved_context}")
        return f"{user_input}\n\nRetrieved context: {retrieved_context}"

class FewShotExampleAppender(ContextAppender):
    def __init__(self, examples: list[str]) -> None:
        super().__init__("Few-shot examples")
        self.examples = examples
        logger.info("Initialized FewShotExampleAppender with examples.")

    def append_context(self, user_input: str) -> str:
        logger.debug("Appending few-shot examples to user input.")
        examples_text = "\n\n".join(f"Example {i+1}:\n{example}" for i, example in enumerate(self.examples))
        logger.debug(f"Appended examples: {examples_text}")
        return f"{user_input}\n\nFew-shot examples:\n{examples_text}"

class DynamicFewShotContextAppender:
    def __init__(
        self,
        few_shot_examples: list[dict[str, str]],
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        n_few_shot_examples: int = 8,
        max_workers: int = 16,
        embedding_size: int = 256,
    ) -> None:
        self.few_shot_examples = few_shot_examples
        self.embedding_model_id = embedding_model_id
        self.n_few_shot_examples = n_few_shot_examples
        self.max_workers = max_workers
        self.embedding_size = embedding_size
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)

        logger.info("Initialized DynamicFewShotContextAppender with configuration.")
        logger.debug(f"Configurations: embedding_model_id={embedding_model_id}, "
                     f"region={region}, n_few_shot_examples={n_few_shot_examples}, "
                     f"max_workers={max_workers}, embedding_size={embedding_size}")

        self.embed_few_shot_examples()

    def embed_few_shot_examples(self) -> None:
        """Embed the few-shot examples using the embedding model."""
        logger.info("Embedding few-shot examples...")

        def embed_example(data: dict[str, str]) -> None:
            question = data["question"]
            logger.debug(f"Embedding question: {question}")
            data["embedding"] = self.embed_text(question)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(embed_example, self.few_shot_examples)
        logger.info("Completed embedding few-shot examples.")

    def embed_text(self, text: str) -> list[float]:
        """Embed the given text using the Bedrock embedding model."""
        logger.debug(f"Embedding text: {text}")
        body = json.dumps({
            "inputText": text,
            "dimensions": self.embedding_size,
            "normalize": True,
        })

        try:
            response = self.bedrock_client.invoke_model(
                body=body, 
                modelId=self.embedding_model_id, 
                accept="application/json", 
                contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding', [])

            if len(embedding) != self.embedding_size:
                logger.warning(f"Expected embedding size {self.embedding_size}, but got {len(embedding)}.")

            logger.debug(f"Generated embedding: {embedding}")
            return embedding
        except ClientError as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate the cosine similarity between two vectors."""
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        logger.debug(f"Calculated cosine similarity: {similarity}")
        return similarity

    def retrieve_few_shot_examples(self, question_embedding: list[float]) -> str:
        """Retrieve the few-shot examples most similar to the input question."""
        logger.info("Retrieving most similar few-shot examples.")

        def similarity(data: dict[str, Any], question_embedding: list[float]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["similarity"] = self.cosine_similarity(question_embedding, data_copy["embedding"])
            return data_copy

        # Create a copy of few_shot_examples and add similarity scores
        with ThreadPoolExecutor() as executor:
            few_shot_examples_with_similarity = list(
                executor.map(lambda x: similarity(x, question_embedding), self.few_shot_examples)
            )

        # Sort the few-shot examples by similarity
        sorted_few_shot_examples = sorted(
            few_shot_examples_with_similarity, key=lambda x: x["similarity"], reverse=True
        )

        # Retrieve the top N most similar few-shot examples
        top_few_shot_examples = sorted_few_shot_examples[: self.n_few_shot_examples]

        # Format the few-shot examples as a string
        few_shot_examples_str = ""
        for i, example in enumerate(top_few_shot_examples):
            few_shot_examples_str += f"Example {i + 1}:\nQuestion: {example['question']}\nSQL Query: {example['sql_query']}\n\n"

        logger.debug(f"Retrieved few-shot examples: {few_shot_examples_str}")
        return few_shot_examples_str

    def append_context(self, user_input: str) -> str:
        """Append the most relevant few-shot examples to the user input."""
        logger.info("Appending context to user input.")
        user_embedding = self.embed_text(user_input)
        few_shot_examples = self.retrieve_few_shot_examples(user_embedding)
        logger.info(f"Appended {self.n_few_shot_examples} relevant few-shot examples to user input.")
        return f"{user_input}\n\n<relevant_examples>\n{few_shot_examples}</relevant_examples>"
