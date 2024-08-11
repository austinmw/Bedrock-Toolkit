""" BedrockClient class for interacting with the Bedrock API. """
import json
from typing import Any, Dict, Generator, List, Optional

import boto3
from botocore.config import Config

from bedrock_toolkit.bedrock_pricing import BEDROCK_PRICING
from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()

class BedrockClient:
    def __init__(self, region: str, config: Optional[Config] = None):
        """
        Initialize the BedrockClient.

        Args:
            region (str): AWS region to use for the Bedrock client.
            config (Optional[Config]): Custom configuration for the Bedrock client.
        """
        if config is None:
            config = Config(
                retries={
                    'max_attempts': 10,
                    'mode': 'adaptive'
                },
                read_timeout=120,
                connect_timeout=30
            )

        self.client = boto3.client(
            service_name='bedrock-runtime',
            config=config,
            region_name=region,
        )

        self.region = region
        self.pricing = self._load_pricing()

    def _load_pricing(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load the pricing information for Bedrock models."""
        # Note: In a production environment, you might want to load this from a file or a database
        return BEDROCK_PRICING

    def converse(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system_prompt: Dict[str, str],
        tool_config: Dict[str, Any],
        inference_config: Dict[str, Any] = {"maxTokens": 4096, "temperature": 0}
    ) -> Dict[str, Any]:
        """
        Handle non-streaming conversations with the Bedrock model.

        Args:
            model_id (str): The ID of the model to use.
            messages (List[Dict[str, Any]]): The conversation history.
            system_prompt (Dict[str, str]): The system prompt to use.
            tool_config (Dict[str, Any]): The tool configuration.
            inference_config (Dict[str, Any], optional): The inference configuration.
                Defaults to {"maxTokens": 4096, "temperature": 0}.

        Returns:
            Dict[str, Any]: The model's response.
        """
        logger.info(f"Generating text with model {model_id}")
        logger.debug(f"Messages: {json.dumps(messages, indent=2)}")

        response = self.client.converse(
            modelId=model_id,
            messages=messages,
            system=[system_prompt],
            inferenceConfig=inference_config,
            toolConfig=tool_config,
        )

        logger.debug(f"Bedrock response:\n{json.dumps(response, indent=4)}")
        return response

    def converse_stream(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system_prompt: Dict[str, str],
        tool_config: Dict[str, Any],
        inference_config: Dict[str, Any] = {"maxTokens": 4096, "temperature": 0},
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Handle streaming conversations with the Bedrock model.

        Args:
            model_id (str): The ID of the model to use.
            messages (List[Dict[str, Any]]): The conversation history.
            system_prompt (Dict[str, str]): The system prompt to use.
            tool_config (Dict[str, Any]): The tool configuration.
            inference_config (Dict[str, Any], optional): The inference configuration.
                Defaults to {"maxTokens": 4096, "temperature": 0}.

        Yields:
            Dict[str, Any]: Chunks of the model's response.
        """
        logger.info(f"Streaming messages with model {model_id}")
        logger.debug(f"Messages: {json.dumps(messages, indent=2)}")

        response = self.client.converse_stream(
            modelId=model_id,
            messages=messages,
            system=[system_prompt],
            inferenceConfig=inference_config,
            toolConfig=tool_config,
        )

        return response

    def calculate_cost(self, usage: Dict[str, int], model_id: str) -> float:
        """
        Calculate the cost of a model invocation.

        Args:
            usage (Dict[str, int]): Token usage information.
            model_id (str): The ID of the model used.

        Returns:
            float: The total cost of the invocation.
        """
        if self.region not in self.pricing:
            raise ValueError(
                f"No pricing information available for region {self.region}. "
                f"Please add {self.region} pricing information to the BEDROCK_PRICING dictionary."
            )

        model_pricing = self.pricing[self.region][model_id]
        input_tokens = usage["inputTokens"]
        output_tokens = usage["outputTokens"]
        total_price = (input_tokens / 1000) * model_pricing["input"] + (output_tokens / 1000) * model_pricing["output"]
        return total_price
