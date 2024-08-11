""" Example of a conversational agent with DynamoDB message history using the Bedrock Toolkit """

import streamlit as st
from pydantic import BaseModel, Field
from typing import Any, Callable

from bedrock_toolkit.logger_manager import LoggerManager
from bedrock_toolkit.bedrock_client import BedrockClient
from bedrock_toolkit.tool_manager import ToolManager
from bedrock_toolkit.model_runner import ModelRunner
from bedrock_toolkit.conversation_manager import ConversationManager
from bedrock_toolkit.streamlit_utils import (
    create_sidebar_options,
    chat_write_stream,
    display_model_response,
    display_messages,
    clear_chat_history,
)

def main() -> None:
    """ Coversational LLM Agent with Cache Weather and City Information Example (Mock APIs) """

    st.title("Conversational LLM Agent")
    st.subheader("with DynamoDB Chat History and Response Cache")

    aws_region = "us-east-1"

    # Step 1: Create sidebar options (excluding tool display for now)
    options = create_sidebar_options()

    # Step 2: Configure logger with sidebar option
    logger_manager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # Step 3: Define your Pydantic models
    class WeatherRequest(BaseModel):
        city: str = Field(..., description="Name of the city to get weather information for")
        country: str = Field(..., description="Country where the city is located")

    class CityInfoRequest(BaseModel):
        city: str = Field(..., description="Name of the city to get information for")
        country: str = Field(..., description="Country where the city is located")

    # Step 4: Define your tool processors
    def get_weather_info(request: WeatherRequest) -> dict[str, str | int]:
        logger.debug(f"Received weather request: city={request.city}, country={request.country}")
        weather_data: dict[str, str | int] = {
            "temperature": "75 degrees Fahrenheit",
            "condition": "Partly cloudy",
            "humidity": int(65)
        }
        logger.info(f"Returning weather data: {weather_data}")
        return weather_data

    def get_city_info(request: CityInfoRequest) -> dict[str, str | int]:
        logger.debug(f"Received city info request: city={request.city}, country={request.country}")
        city_data: dict[str, str | int] = {
            "population": 8_419_000,
            "country": "United States",
            "timezone": "Eastern Time Zone (ET)"
        }
        logger.info(f"Returning city info data: {city_data}")
        return city_data

    # Step 5: Set up your tools
    tool_schemas = [WeatherRequest, CityInfoRequest]
    tool_processors: dict[str, Callable[..., Any]] | None = {
        "WeatherRequest": get_weather_info,
        "CityInfoRequest": get_city_info,
    }

    # Step 6: Display available tools in the sidebar
    st.sidebar.title("Agent Capabilities")
    if tool_processors:
        for tool in tool_processors.keys():
            st.sidebar.text(f"• {tool}")

    # Step 7: Initialize components, caching resources to maintain statesan
    @st.cache_resource
    def get_bedrock_client() -> BedrockClient:
        return BedrockClient(region=aws_region)

    @st.cache_resource
    def get_tool_manager() -> ToolManager:
        return ToolManager(tool_schemas, tool_processors)

    @st.cache_resource
    def get_model_runner() -> ModelRunner:
        return ModelRunner(
            bedrock_client=get_bedrock_client(),
            tool_manager=get_tool_manager(),
            first_tool_choice="auto",
        )

    @st.cache_resource
    def get_conversation_manager() -> ConversationManager:
        chat_id = "00000"

        return ConversationManager(
            model_runner=get_model_runner(),
            max_turns=10,  # Limit the conversation to N turns, or None for unlimited turns
            chat_history_config={
                "type": "dynamodb",
                "table_name": "MyChatHistoryTable",
                "region": aws_region,
                "chat_id": chat_id,
                "delete_existing_table": True,
            },
            cache_config={
                "type": "dynamodb",  # Specifies the cache type as DynamoDB
                "table_name": "MyCacheTable",
                "region": aws_region,
                "cache_similarity_threshold": 0.70,  # Similarity threshold for caching
                "cache_time_interval": 60 * 5,  # Cache time interval in seconds (5 minutes)
                "max_cache_size": 1000,  # Maximum size of the cache
                "chat_id": chat_id,
                "delete_existing_table": True,
            },
        )

    # Initialize or retrieve session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_manager()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_prompt := st.chat_input("Enter your prompt"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Create a placeholder for the assistant's response
        assistant_placeholder = st.chat_message("assistant")

        # Display assistant response in chat message container
        with assistant_placeholder:
            with st.spinner("Thinking..."):
                try:
                    # Call the model and process the output
                    streamed_text, response_data = st.session_state.conversation.say(
                        text=user_prompt,
                        model_id=options["model_id"],
                        use_streaming=options["use_streaming"],
                        invoke_limit=options["invoke_limit"],
                        max_retries=options["max_retries"],
                        write_stream=chat_write_stream  # Using the chat_write_stream function
                    )

                    # Add response to session state
                    st.session_state.streamed_text = streamed_text
                    st.session_state.response_data = response_data

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": streamed_text})

                except Exception as e:
                    logger.exception("An unexpected error occurred")
                    st.error("An error occurred. Consider increasing the max retries.")
                    st.exception(e)

    # Add clear chat history button if chat history exists
    if "messages" in st.session_state and len(st.session_state.messages):
        if st.button("Clear Chat History", use_container_width=True):
            clear_chat_history()

    # Display the model response or messages if they were successful
    if "streamed_text" in st.session_state and "response_data" in st.session_state:
        display_model_response(st.session_state.streamed_text, st.session_state.response_data)
    # If there was an error, only display the messages
    elif "messages" in st.session_state and len(st.session_state.messages):
        display_messages(st.session_state.messages)

if __name__ == "__main__":
    main()
