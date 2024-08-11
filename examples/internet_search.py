"""Example of a conversational agent using the Bedrock Toolkit with Tavily Search"""

import json
import os
from typing import Any, Callable, Dict

import streamlit as st
from pydantic import BaseModel, Field
from tavily import TavilyClient  # type: ignore

from bedrock_toolkit.bedrock_client import BedrockClient
from bedrock_toolkit.conversation_manager import ConversationManager
from bedrock_toolkit.logger_manager import LoggerManager
from bedrock_toolkit.model_runner import ModelRunner
from bedrock_toolkit.streamlit_utils import (
    chat_write_stream,
    clear_chat_history,
    create_sidebar_options,
    display_messages,
    display_model_response,
)
from bedrock_toolkit.tool_manager import ToolManager

# Set the page layout to wide mode
# st.set_page_config(layout="wide")


def main() -> None:
    """Internet Agent Example using Tavily Search"""

    st.title("Internet Agent Example")
    st.subheader("Powered by Tavily Search")

    aws_region = "us-east-1"

    # Step 1: Create sidebar options
    options = create_sidebar_options(
        default_model="anthropic.claude-3-sonnet-20240229-v1:0"
    )

    # Step 2: Configure logger with sidebar option
    logger_manager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # Step 3: Define your Pydantic model for Tavily search
    class InternetSearch(BaseModel):
        """Request format for Tavily internet search request"""

        query: str = Field(..., description="Search query to look up")
        max_results: int = Field(10, ge=3, description="Max search results to return")
        search_depth: str = Field(
            "advanced", description="The depth of the search, 'basic' or 'advanced'"
        )
        topic: str = Field(
            "general", description="The topic of the search, 'general' or 'news'"
        )
        include_images: bool = Field(
            False, description="Include a list of query-related images"
        )
        # include_answer: bool = Field(False, description="Weather to include a short answer to original query")

    # Step 4: Define your tool processor for Tavily search
    def process_tavily_search(request: InternetSearch) -> Dict[str, Any]:
        logger.info(
            f"Received Tavily search request:\n{json.dumps(request.model_dump(), indent=2)}"
        )

        # Initialize Tavily client (make sure to set TAVILY_API_KEY in your environment)
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        try:
            # Perform the search
            response = client.search(
                query=request.query,
                max_results=request.max_results,
                search_depth=request.search_depth,
                topic=request.topic,
                use_cache=False,
                include_images=request.include_images,
                # include_answer=request.include_answer,
            )

            logger.info(
                "Tavily search completed successfully. Turn on debug logging for more details."
            )
            logger.debug(f"Tavily search response:\n{json.dumps(response, indent=2)}")
            return response
        except Exception as e:
            logger.error(f"Error occurred during Tavily search: {str(e)}")
            return {"error": str(e)}

    # Step 5: Set up your tools
    tool_schemas: list[type[BaseModel]] = [InternetSearch]
    tool_processors: Dict[str, Callable[..., Any]] = {
        "InternetSearch": process_tavily_search,
    }

    # Step 6: Display available tools in the sidebar
    st.sidebar.title("Agent Capabilities")
    if tool_processors:
        for tool in tool_processors.keys():
            st.sidebar.text(f"• {tool}")

    # Step 7: Initialize components, caching resources to maintain state
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
        return ConversationManager(
            model_runner=get_model_runner(),
            max_turns=10,  # Limit the conversation to N turns, or None for unlimited turns
        )

    # Initialize or retrieve session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_manager()

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
                        write_stream=chat_write_stream,  # Using the chat_write_stream function
                    )
                    # Add response to session state
                    st.session_state.streamed_text = streamed_text
                    st.session_state.response_data = response_data

                    # Add assistant response to chat history
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
