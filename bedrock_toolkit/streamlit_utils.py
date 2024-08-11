""" Streamlit utility functions for interacting with Bedrock models.

Bedrock model IDs and supported features:
https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features
"""

import json
from typing import Any, Callable, Generator, Literal, Tuple, TypedDict

import streamlit as st

from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()

log_level_options = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

def write_stream(stream: Generator[str | dict[str, Any], None, None]) -> Tuple[str, dict[str, Any] | None]:
    """
    Handle streaming output in Streamlit.

    Args:
        stream (Generator[str | dict[str, Any], None, None]): Generator yielding text chunks or final response data.

    Returns:
        Tuple[str, dict[str, Any] | None]: The complete streamed text and the final response data (if any).
    """
    result = ""
    container = st.empty()
    response_data = None

    for item in stream:
        if isinstance(item, str):
            result += item
            container.markdown(result)
        elif isinstance(item, dict):
            response_data = item

    return result, response_data

def chat_write_stream(stream: Generator[str | dict[str, Any], None, None]) -> tuple[str, dict[str, Any] | None]:
    """
    Handle streaming output in Streamlit chat.

    Args:
        stream (Generator[str | dict[str, Any], None, None]): Generator yielding text chunks or final response data.

    Returns:
        Tuple[str, dict[str, Any] | None]: The complete streamed text and the final response data (if any).
    """
    result = ""
    response_data: dict[str, Any] | None = None

    # Create a placeholder for the assistant's message
    assistant_message_placeholder = st.empty()

    for item in stream:
        if isinstance(item, str):
            result += item
            assistant_message_placeholder.markdown(result)  # Update the placeholder with new text
        elif isinstance(item, dict):
            response_data = item

    return result, response_data

def print_response_data(
    response_data: dict[str, Any],
    print_func: Callable[[str], None] = st.write,
    json_indent: int = 4
) -> None:
    """
    Print the details of the response data in Streamlit.

    Args:
        response_data (dict[str, Any]): The dictionary containing response data including tool calls and usage statistics.
        print_func (Callable[[str], None]): The function to use for printing (default is st.write).
        json_indent (int): The indentation level for JSON output (default is 4 spaces).

    Returns:
        None
    """
    if not response_data:
        print_func("No response data received.")
        return

    print_func("--- Response Data Details ---")

    if "tool_calls" in response_data:
        print_func("Tool Calls:")
        for tool_call in response_data["tool_calls"]:
            print_func(f"{tool_call.__class__.__name__}: {tool_call.model_dump_json(indent=json_indent)}")

    if "usage" in response_data:
        print_func("Usage Statistics:")
        print_func(json.dumps(response_data["usage"], indent=json_indent))

    if "messages" in response_data:
        print_func("Messages:")
        for message in response_data["messages"]:
            print_func(f"Role: {message['role']}")
            for content in message['content']:
                if 'text' in content:
                    print_func(f"Text: {content['text']}")
                elif 'toolUse' in content:
                    print_func(f"Tool Use: {json.dumps(content['toolUse'], indent=json_indent)}")
                elif 'toolResult' in content:
                    print_func(f"Tool Result: {json.dumps(content['toolResult'], indent=json_indent)}")
            print_func("")  # Add a blank line between messages for readability

def create_expandable_section(title: str, content: Any) -> None:
    """
    Create an expandable section in Streamlit.

    Args:
        title (str): The title of the expandable section.
        content (Any): The content to display in the expandable section.

    Returns:
        None
    """
    with st.expander(title, expanded=False):
        if isinstance(content, dict):
            st.json(content)
        elif isinstance(content, str):
            st.text(content)
        else:
            st.write(content)

def count_message_types(messages):
    user_text_messages = 0
    assistant_tool_calls = 0
    user_tool_outputs = 0
    assistant_no_tool_calls = 0

    for message in messages:
        if message["role"] == "user":
            if any("toolResult" in content for content in message["content"] if isinstance(content, dict)):
                user_tool_outputs += 1
            else:
                user_text_messages += 1

        elif message["role"] == "assistant":
            if any("toolUse" in content for content in message["content"] if isinstance(content, dict)):
                assistant_tool_calls += 1
            else:
                assistant_no_tool_calls += 1


    return {
        "User Text Messages": user_text_messages,
        "Assistant Messages with Tool Calls": assistant_tool_calls,
        "User Messages with Tool Outputs": user_tool_outputs,
        "Assistant Messages without Tool Calls": assistant_no_tool_calls
    }

def display_model_response(streamed_text: str, response_data: dict[str, Any] | None) -> None:
    """
    Display the model's response and additional data in Streamlit.

    Args:
        streamed_text (str): The complete streamed text from the model.
        response_data (dict[str, Any] | None): Additional response data, if any.

    Returns:
        None
    """

    if response_data:
        st.markdown("#### Diagnostic Information")

        if "messages" in response_data:
            create_expandable_section("Message counts", count_message_types(response_data["messages"]))
            create_expandable_section("Messages", response_data["messages"])

        if "tool_calls" in response_data:
            create_expandable_section("Tool calls in last response", response_data["tool_calls"])

        if "usage" in response_data:
            create_expandable_section("Token usage in last response", response_data["usage"])

def display_messages(messages: list[dict[str, Any]]) -> None:
    create_expandable_section("Messages", messages)

class Options(TypedDict):
    log_level: log_level_options
    model_id: str
    inference_config: dict
    use_streaming: bool
    max_retries: int
    invoke_limit: int | None

def create_sidebar_options() -> Options:
    """
    Create sidebar options for the Streamlit app.

    Returns:
        dict[str, Any]: A dictionary containing the selected options.
    """
    st.sidebar.title("Model Options")

    options: Options = {
        "log_level": "INFO",
        "model_id": "",
        "inference_config": {"maxTokens": 4096, "temperature": 0},
        "use_streaming": False,
        "max_retries": 3,
        "invoke_limit": None,
    }

    log_level = st.sidebar.selectbox(
        "Log Level",
        ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        index=2  # Set default to "INFO"
    )
    options["log_level"] = log_level  # type: ignore

    options["model_id"] = str(st.sidebar.selectbox(
        "Select Model",
        [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "mistral.mistral-large-2402-v1:0",
            "mistral.mistral-small-2402-v1:0",
            "cohere.command-r-plus-v1:0",
            "cohere.command-r-v1:0",
        ],
        index=2,
    ))

    options["use_streaming"] = bool(st.sidebar.checkbox("Use Streaming", value=True))

    options["max_retries"] = int(st.sidebar.number_input("Max Retries", min_value=1, max_value=10, value=3))

    invoke_limit_input = st.sidebar.number_input("Invoke Limit", min_value=1, max_value=20, value=None)
    options["invoke_limit"] = int(invoke_limit_input) if invoke_limit_input is not None else None

    return options

def clear_chat_history():
    st.session_state.messages = []
    st.session_state.conversation.clear_messages()
    st.experimental_rerun()
