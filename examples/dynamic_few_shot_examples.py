"""Example of a conversational agent using the Bedrock Toolkit"""

from typing import Any, Callable

import streamlit as st
from pydantic import BaseModel, Field

from bedrock_toolkit import (
    BedrockClient,
    ConversationManager,
    LoggerManager,
    ModelRunner,
    ToolManager,
)
from bedrock_toolkit.context_appender import (
    ContextAppender,
    DynamicFewShotContextAppender,
)
from bedrock_toolkit.streamlit_utils import (
    chat_write_stream,
    clear_chat_history,
    create_sidebar_options,
    display_messages,
    display_model_response,
)


def main() -> None:
    """Conversational LLM Agent Weather and City Information Example (Mock APIs)"""

    st.title("Conversational LLM Agent")
    st.subheader("City Weather and Info Example (Mock APIs)")

    aws_region = "us-east-1"

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
        return BedrockClient(region="us-east-1")

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
        # Example few-shot examples
        few_shot_examples = [
            {
                "question": "What is the total revenue for 2022?",
                "sql_query": "SELECT SUM(revenue) FROM sales WHERE year = 2022;",
            },
            {
                "question": "How many users signed up in 2023?",
                "sql_query": "SELECT COUNT(user_id) FROM users WHERE signup_date BETWEEN '2023-01-01' AND '2023-12-31';",
            },
            {
                "question": "What are the top 10 products by sales?",
                "sql_query": "SELECT product_name, SUM(sales) FROM sales GROUP BY product_name ORDER BY SUM(sales) DESC LIMIT 10;",
            },
        ]

        # Initialize the context appender
        context_appender: ContextAppender = DynamicFewShotContextAppender(
            region=aws_region,
            few_shot_examples=few_shot_examples,
            n_few_shot_examples=2,
        )

        return ConversationManager(
            model_runner=get_model_runner(),
            context_appender=context_appender,
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
                        write_stream=chat_write_stream,
                    )

                    # Add assistant response to Streamlit chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": streamed_text}
                    )

                except Exception as e:
                    logger.exception("An unexpected error occurred")
                    st.error("An error occurred. Consider increasing the max retries.")
                    st.exception(e)

    # Add clear chat history button if chat history exists
    if "messages" in st.session_state and len(st.session_state.messages):
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            clear_chat_history()

    # Display the model response or messages if they were successful
    if "streamed_text" in st.session_state and "response_data" in st.session_state:
        display_model_response(
            st.session_state.streamed_text, st.session_state.response_data
        )
    # If there was an error, only display the messages
    elif "messages" in st.session_state and len(st.session_state.messages):
        display_messages(st.session_state.messages)


if __name__ == "__main__":
    main()
