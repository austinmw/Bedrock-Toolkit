"""LLM Agent Weather and City Information Example (Mock APIs)"""

import streamlit as st

from bedrock_toolkit.bedrock_client import BedrockClient
from bedrock_toolkit.logger_manager import LoggerManager
from bedrock_toolkit.model_runner import ModelRunner
from bedrock_toolkit.streamlit_utils import (
    create_sidebar_options,
    display_model_response,
    write_stream,
)


def main() -> None:
    st.title("LLM Agent")
    st.subheader("Basic Example")

    # Step 1: Create sidebar options (excluding tool display for now)
    options = create_sidebar_options()

    # Step 2: Configure logger with sidebar option
    logger_manager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # Step 7: Initialize components
    bedrock_client = BedrockClient(region="us-east-1")
    model_runner = ModelRunner(bedrock_client)

    # Main area for user input with prepopulated question
    default_prompt = "What's your name?"
    user_prompt: str = st.text_area(
        "Enter your prompt", value=default_prompt, height=100
    )

    if st.button("Run"):
        if user_prompt:
            try:
                with st.spinner("Thinking..."):
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
