"""LLM Agent Weather and City Information Example (Mock APIs)"""

import streamlit as st
from typing import Any, Callable
from pydantic import BaseModel, Field

from bedrock_toolkit.logger_manager import LoggerManager
from bedrock_toolkit.bedrock_client import BedrockClient
from bedrock_toolkit.tool_manager import ToolManager
from bedrock_toolkit.model_runner import ModelRunner
from bedrock_toolkit.streamlit_utils import (
    create_sidebar_options,
    write_stream,
    display_model_response,
)


def main() -> None:
    """LLM Agent Weather and City Information Example (Mock APIs)

    This example demonstrates how the assistant uses two mock API tools:

    1. WeatherRequest: Simulates getting current weather for a city and country.
    2. CityInfoRequest: Simulates retrieving background information about a location.

    The assistant must determine which tools to use based on the user's query and combine the mock data
    into a coherent response. This example uses predefined mock data to simulate API responses.

    The interaction typically occurs in two or three generations:

    1. First Generation (Weather Tool Call):
    The assistant recognizes the need for weather information. It generates a response containing two components:
    a) Text explaining the need to use a weather tool to answer the query.
    b) A tool call in JSON format for the WeatherRequest tool.

    This JSON tool call is then parsed back into a corresponding Pydantic object to validate its structure.
    If valid, it's used as input for a `tool_processor` function. This function may perform additional
    operations (like making an API call) or simply return the structured output if that's sufficient to answer
    the question. The result is then appended as a new user message.

    2. Second Generation (City Info Tool Call):
    After receiving the weather data, the assistant recognizes that it needs additional background
    information about the location. It again generates a response with two components:
    a) Text explaining the need for more information about the city.
    b) A tool call in JSON format for the CityInfoRequest tool.

    Similarly, this JSON is parsed into a Pydantic object, validated, and processed by its
    corresponding `tool_processor` function. The result is appended to the messages list.

    3. Third Generation (Final Answer):
    With both weather and city information now available, the assistant generates a final text response
    that combines all the gathered data into a comprehensive answer to the user's original query.
    This response does not include a tool call.

    Sometimes, the assistant may decide to call each tool in parallel or in serial,
    depending on the complexity and nature of the query. For instance, if the query is straightforward
    and can be addressed by calling both tools simultaneously, the assistant will make parallel calls
    to gather information more efficiently. In other cases, it may call the tools in a sequential manner,
    as described in the three generations above.

    This multi-turn process allows the assistant to gather all required information step-by-step
    before providing a complete answer. Users should observe these two or three separate generations.
    The first two generations each include both explanatory text and a tool call in JSON format
    (which is then parsed and processed), while the final generation is a text-only response.

    The parsing of JSON tool calls back into Pydantic objects serves multiple purposes:
    1. It validates the structure of the LLM's output, ensuring it conforms to the expected schema.
    2. It provides type safety and autocompletion for developers working with the tool inputs.
    3. It allows for easy integration with `tool_processor` functions, which can be designed to work
    with these typed objects.

    The `tool_processor` functions offer flexibility in how tool calls are handled.
    They can perform complex operations like API calls, database queries, or computations.
    Alternatively, if the structured Pydantic object itself contains all necessary information
    to answer the query, the `tool_processor` might simply return this object, allowing the LLM
    to use its structured data in formulating the final response.

    This approach combines the power of LLM-generated queries with the safety and utility of
    strongly-typed data structures, enabling robust and flexible tool use in AI applications.
    """

    st.title("LLM Agent")
    st.subheader("City Weather and Info Example (Mock APIs)")

    # Step 1: Create sidebar options (excluding tool display for now)
    options = create_sidebar_options()

    # Step 2: Configure logger with sidebar option
    logger_manager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # Step 3: Define your Pydantic models
    class WeatherRequest(BaseModel):
        city: str = Field(
            ..., description="Name of the city to get weather information for"
        )
        country: str = Field(..., description="Country where the city is located")

    class CityInfoRequest(BaseModel):
        city: str = Field(..., description="Name of the city to get information for")
        country: str = Field(..., description="Country where the city is located")

    # Step 4: Define your tool processors
    def get_weather_info(request: WeatherRequest) -> dict[str, str | int]:
        logger.debug(
            f"Received weather request: city={request.city}, country={request.country}"
        )
        weather_data: dict[str, str | int] = {
            "temperature": "75 degrees Fahrenheit",
            "condition": "Partly cloudy",
            "humidity": int(65),
        }
        logger.info(f"Returning weather data: {weather_data}")
        return weather_data

    def get_city_info(request: CityInfoRequest) -> dict[str, str | int]:
        logger.debug(
            f"Received city info request: city={request.city}, country={request.country}"
        )
        city_data: dict[str, str | int] = {
            "population": 8_419_000,
            "country": "United States",
            "timezone": "Eastern Time Zone (ET)",
        }
        logger.info(f"Returning city info data: {city_data}")
        return city_data

    # Step 5: Set up your tools
    tool_schemas: list[type[BaseModel]] = [WeatherRequest, CityInfoRequest]
    tool_processors: dict[str, Callable[[Any], dict[str, str | int]]] = {
        "WeatherRequest": get_weather_info,
        "CityInfoRequest": get_city_info,
    }

    # Step 6: Display available tools in the sidebar
    st.sidebar.title("Agent Capabilities")
    for tool in tool_processors.keys():
        st.sidebar.text(f"• {tool}")

    # Step 7: Initialize components
    bedrock_client = BedrockClient(region="us-east-1")
    tool_manager = ToolManager(tool_schemas, tool_processors)
    model_runner = ModelRunner(bedrock_client, tool_manager)

    # Main area for user input with prepopulated question
    default_prompt = "Provide weather and city info for NYC"
    user_prompt: str = st.text_area(
        "Enter your prompt", value=default_prompt, height=100
    )

    if st.button("Run"):
        if user_prompt:
            try:
                with st.spinner("Processing..."):
                    # Prepare initial messages
                    messages = [{"role": "user", "content": [{"text": user_prompt}]}]

                    # Call the model and process the output
                    response_generator = model_runner.generate_text(
                        model_id=options["model_id"],
                        messages=messages,
                        use_streaming=options["use_streaming"],
                        invoke_limit=options["invoke_limit"],
                        max_retries=options["max_retries"],
                    )

                    streamed_text, response_data = write_stream(response_generator)

                # Display the response
                display_model_response(streamed_text, response_data)

                st.success("Processing complete!")

            except Exception as e:
                logger.exception("An unexpected error occurred")
                st.error("An error occurred. Consider increasing the max retries.")
                st.exception(e)

        else:
            st.warning("Please enter a prompt before running.")


if __name__ == "__main__":
    main()
