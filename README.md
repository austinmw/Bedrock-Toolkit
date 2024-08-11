# Bedrock LLM Agent Tool Use Framework

## Background

In the rapidly evolving landscape of Large Language Model (LLM) applications, developers often encounter unexpected challenges when integrating these models with custom tools and workflows. A significant issue that has emerged, particularly with Anthropic models, is the lack of server-side schema verification for tool responses. This means that, contrary to what some developers might expect, there's no guarantee that the LLM's responses will strictly adhere to the input tool specifications.

This discrepancy can lead to substantial problems, especially when these potentially non-conforming responses are used as inputs for downstream functions or processes. The issue is further complicated when working with popular LLM orchestration frameworks like LangChain.

LangChain and similar frameworks have introduced features like `bind_tools` to support Pydantic models. Their approach typically involves:

1. Converting Pydantic models to OpenAI-format tool specifications
2. Sending these specifications to the model as input
3. Receiving a JSON response from the model
4. Attempting to parse this response back into a filled-out Pydantic object

However, due to the lack of guaranteed schema conformity in the LLM's output, the final parsing step can fail. Within these frameworks, it's often challenging to implement robust client-side verification and effective retry logic at this critical point of potential failure.

This framework was created to address these specific challenges, offering a more flexible and resilient approach to LLM-tool integration, particularly within the AWS Bedrock ecosystem.

## Overview

This framework provides a robust solution for integrating large language models (LLMs) with custom tools, specifically designed for use with AWS Bedrock. It offers a streamlined approach to handling complex, multi-turn conversations that require external data processing or computation. The framework leverages Pydantic for defining tool interfaces, allowing for flexible output options where Pydantic objects can serve as the final desired output or as validated inputs for downstream tools and APIs.

## Key Features

- **Pydantic-based Tool Definitions**: Leverage the power of Pydantic for type-safe and validated tool interfaces.
- **Automatic ToolSpec Conversion**: Seamlessly converts Pydantic models to Bedrock-compatible toolSpecs.
- **Multi-turn Interaction Support**: Handles complex conversation flows with multiple serial or parallel tool calls.
- **Response Parsing**: Ensures LLM outputs adhere to expected schemas.
- **Sophisticated Error Handling**: Implements retry logic with LLM feedback for improved robustness.
- **Flexible Output Options**: Return Pydantic objects for direct use or further processing, or raw assistant messages.
- **Streaming Support**: Enable real-time interaction where needed.
- **Multi-tool Scenario Handling**: Manage conversations requiring multiple different tools.

## How It Works

1. Define your tools using Pydantic models.
2. The framework converts these models to Bedrock toolSpecs using `pydantic_to_toolspec()`.
3. Engage in multi-turn conversations with the LLM, automatically calling tools as needed.
4. Parse and validate LLM responses to ensure schema compliance.
5. Handle errors gracefully, providing feedback to the LLM for potential self-correction.
6. Process tool outputs, either returning Pydantic objects directly for use as final output or as validated input for downstream processes, or perform additional computations as needed.

## Key Components

- `register_models_from_tools()`: Creates a global registry of all Pydantic models, including nested ones.
- `pydantic_to_toolspec()`: Converts Pydantic models to LLM-compatible tool specifications.
- `generate_text()`: Manages LLM interactions with retry logic.
- `process_tool_use()`: Parses LLM tool responses into Pydantic objects, and passes these as inputs to corresponding tool processor functions, returning the output to the model.

## Usage

```python
# Example usage (simplified)
tool_schemas = [WeatherTool, CityInfoTool]

response_dict = call_model_with_tools(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    # Define an initial input prompt
    user_prompt="What's the weather like in New York?",
    # List the Pydantic models to use
    tool_schemas=tool_schemas,
    # Optional: List of functions to process pydantic outputs
    # If None, the Pydantic object itself will be the tool output
    tool_processors=None,
    use_streaming=True,
    invoke_limit=None,
    max_retries=3,
    log_level="INFO",
)

for tool_output in response_dict["tool_calls"]:
    if isinstance(tool_output, WeatherInfo):
        # Access the properties of the WeatherInfo object directly
        print(f"Temperature: {response.temperature}°C")
```

## Why This Framework?

While similar to other LLM orchestration frameworks like LangChain's `bind_tools`, this solution is tailored specifically for AWS Bedrock. It provides a balance of convenience (through Pydantic usage) and compatibility (with Bedrock's toolSpec requirements), making it an excellent choice for developers working within the AWS ecosystem. The framework's flexibility in handling Pydantic objects as both final outputs and inputs for further processing enhances its utility in complex workflows.

## Advanced Features

- **Nested Schema Handling**: Supports complex tool definitions with nested Pydantic models.
- **Flexible Tool Processing**: Tool functions can return Pydantic model instances for direct use, perform complex operations with external APIs, or serve as validated inputs for downstream processes.
- **Comprehensive Logging**: Detailed logging at each step for debugging and monitoring.

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Define your Pydantic models for tools
4. Set up your AWS credentials for Bedrock access
5. Use the `call_model_with_tools` function to start interacting with your LLM and tools
