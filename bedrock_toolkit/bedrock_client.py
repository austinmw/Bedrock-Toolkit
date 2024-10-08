"""BedrockClient class for interacting with the Bedrock API."""

import json
from typing import NotRequired, TypedDict

import boto3
from botocore.config import Config
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.type_defs import (
    ConverseResponseTypeDef,
    ConverseStreamResponseTypeDef,
    InferenceConfigurationTypeDef,
    MessageOutputTypeDef,
    MessageTypeDef,
    SystemContentBlockTypeDef,
    ToolConfigurationTypeDef,
)

from bedrock_toolkit.bedrock_pricing import BEDROCK_PRICING
from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()


class ConverseParams(TypedDict):
    modelId: str
    messages: list[MessageTypeDef | MessageOutputTypeDef]
    system: list[SystemContentBlockTypeDef]
    inferenceConfig: InferenceConfigurationTypeDef
    toolConfig: NotRequired[ToolConfigurationTypeDef]


class BedrockClient:
    def __init__(self, region: str, config: Config | None = None):
        """
        Initialize the BedrockClient.

        Args:
            region (str): AWS region to use for the Bedrock client.
            config (Config | None): Custom configuration for the Bedrock client.
        """
        if config is None:
            config = Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                read_timeout=120,
                connect_timeout=30,
            )

        self.client: BedrockRuntimeClient = boto3.client(
            service_name="bedrock-runtime",
            config=config,
            region_name=region,
        )

        self.region = region
        self.pricing = self._load_pricing()

    def _load_pricing(self) -> dict[str, dict[str, dict[str, float]]]:
        """Load the pricing information for Bedrock models."""
        # Note: In a production environment, you might want to load this from a file or a database
        return BEDROCK_PRICING

    def converse(
        self,
        model_id: str,
        messages: list[MessageTypeDef | MessageOutputTypeDef],
        system_prompt: SystemContentBlockTypeDef,
        tool_config: ToolConfigurationTypeDef | None = None,
        inference_config: InferenceConfigurationTypeDef = {
            "maxTokens": 4096,
            "temperature": 0,
        },
    ) -> ConverseResponseTypeDef:
        """
        Handle non-streaming conversations with the Bedrock model.

        Args:
            model_id (str): The ID of the model to use.
            messages (Sequence[MessageTypeDef | MessageOutputTypeDef]): The conversation history.
            system_prompt (SystemContentBlockTypeDef): The system prompt to use.
            tool_config (Optional[ToolConfigurationTypeDef]): The tool configuration.
            inference_config (InferenceConfigurationTypeDef, optional): The inference configuration.
                Defaults to {"maxTokens": 4096, "temperature": 0}.

        Returns:
            ConverseResponseTypeDef: The model's response.
        """
        logger.info(f"Generating text with model {model_id}")
        logger.debug(f"Messages: {json.dumps(messages, indent=2)}")

        # Prepare the arguments for the converse method
        params: ConverseParams = {
            "modelId": model_id,
            "messages": messages,
            "system": [system_prompt],
            "inferenceConfig": inference_config,
        }

        if tool_config is not None:
            params["toolConfig"] = tool_config

        response: ConverseResponseTypeDef = self.client.converse(**params)
        logger.debug(f"Bedrock response:\n{json.dumps(response, indent=4)}")
        return response

    def converse_stream(
        self,
        model_id: str,
        messages: list[MessageTypeDef | MessageOutputTypeDef],
        system_prompt: SystemContentBlockTypeDef,
        tool_config: ToolConfigurationTypeDef | None = None,
        inference_config: InferenceConfigurationTypeDef = {
            "maxTokens": 4096,
            "temperature": 0,
        },
    ) -> ConverseStreamResponseTypeDef:
        """
        Handle streaming conversations with the Bedrock model.

        Args:
            model_id (str): The ID of the model to use.
            messages (list[dict[str, Any]]): The conversation history.
            system_prompt (dict[str, str]): The system prompt to use.
            tool_config (dict[str, Any]): The tool configuration.
            inference_config (dict[str, Any], optional): The inference configuration.
                Defaults to {"maxTokens": 4096, "temperature": 0}.

        Yields:
            dict[str, Any]: Chunks of the model's response.
        """
        logger.info(f"Streaming messages with model {model_id}")
        logger.debug(f"Messages: {json.dumps(messages, indent=2)}")

        # Prepare the arguments for the converse_stream method
        params: ConverseParams = {
            "modelId": model_id,
            "messages": messages,
            "system": [system_prompt],
            "inferenceConfig": inference_config,
        }

        if tool_config is not None:
            params["toolConfig"] = tool_config

        response: ConverseStreamResponseTypeDef = self.client.converse_stream(**params)
        return response

    def calculate_cost(self, usage: dict[str, int], model_id: str) -> float:
        """
        Calculate the cost of a model invocation.

        Args:
            usage (dict[str, int]): Token usage information.
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
        total_price = (input_tokens / 1000) * model_pricing["input"] + (
            output_tokens / 1000
        ) * model_pricing["output"]
        return total_price
