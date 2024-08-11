import json
import os
from typing import Any, Callable, Tuple

import faiss  # type: ignore
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer  # type: ignore

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


def main() -> None:
    """Local FAISS Search Agent Example for Python Files"""

    st.title("Local FAISS Search Agent for Bedrock Toolkit Python Files")
    st.subheader("Powered by FAISS and SentenceTransformers")

    aws_region: str = "us-east-1"

    # Step 1: Create sidebar options
    options = create_sidebar_options()

    # Step 2: Configure logger with sidebar option
    logger_manager: LoggerManager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # Step 3: Define your Pydantic model for FAISS search
    class RepositorySearch(BaseModel):
        """Request format for local FAISS search request"""

        query: str = Field(..., description="Search query to look up")
        max_results: int = Field(
            5, ge=1, le=10, description="Max search results to return"
        )

    # Step 4: Define your tool processor for FAISS search
    def process_faiss_search(request: RepositorySearch) -> dict[str, Any]:
        logger.info(
            f"Received FAISS search request:\n{json.dumps(request.model_dump(), indent=2)}"
        )

        try:
            # Perform the search
            query_embedding: np.ndarray = model.encode([request.query])[0]
            distances: np.ndarray
            indices: np.ndarray
            distances, indices = index.search(
                np.array([query_embedding]), request.max_results
            )

            results: list[dict[str, str | float]] = []
            for i, idx in enumerate(indices[0]):
                results.append(
                    {
                        "file": file_paths[idx],
                        "content": documents[idx],
                        "relevance_score": float(
                            1 - distances[0][i]
                        ),  # Convert distance to similarity score
                    }
                )

            response: dict[str, str | list[dict[str, str | float]]] = {
                "query": request.query,
                "results": results,
            }

            logger.info(
                "FAISS search completed successfully. Turn on debug logging for more details."
            )
            logger.debug(f"FAISS search response:\n{json.dumps(response, indent=2)}")
            return response
        except Exception as e:
            logger.error(f"Error occurred during FAISS search: {str(e)}")
            return {"error": str(e)}

    # Step 5: Set up your tools
    tool_schemas: list[type[BaseModel]] = [RepositorySearch]
    tool_processors: dict[str, Callable[..., Any]] = {
        "RepositorySearch": process_faiss_search,
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

    @st.cache_resource
    def load_python_files() -> Tuple[list[str], list[str]]:
        documents: list[str] = []
        file_paths: list[str] = []
        base_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Read Python files from examples directory
        examples_dir: str = os.path.join(base_path, "examples")
        for filename in os.listdir(examples_dir):
            if filename.endswith(".py"):
                file_path: str = os.path.join(examples_dir, filename)
                with open(file_path, "r") as file:
                    documents.append(file.read())
                    file_paths.append(os.path.relpath(file_path, base_path))

        # Read Python files from bedrock_toolkit directory
        toolkit_dir: str = os.path.join(base_path, "bedrock_toolkit")
        for root, _, files in os.walk(toolkit_dir):
            for filename in files:
                if filename.endswith(".py"):
                    file_path: str = os.path.join(root, filename)  # type: ignore
                    with open(file_path, "r") as file:
                        documents.append(file.read())
                        file_paths.append(os.path.relpath(file_path, base_path))

        return documents, file_paths

    @st.cache_resource
    def setup_faiss_index(
        documents: list[str],
    ) -> Tuple[SentenceTransformer, faiss.IndexFlatL2]:
        model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings: np.ndarray = model.encode(documents)

        index: faiss.IndexFlatL2 = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        return model, index

    # Load documents and set up FAISS index
    documents, file_paths = load_python_files()
    model, index = setup_faiss_index(documents)

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
