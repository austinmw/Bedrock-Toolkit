"""ConversationManager class for managing conversational interactions."""

import time
from typing import Any, Callable, Generator

from bedrock_toolkit.cache_services import CacheService, create_cache_service
from bedrock_toolkit.chat_history import ChatHistoryStorage, create_chat_history_storage
from bedrock_toolkit.logger_manager import LoggerManager
from bedrock_toolkit.model_runner import ModelRunner
from bedrock_toolkit.context_appender import ContextAppender

logger = LoggerManager.get_logger()


class ConversationManager:
    def __init__(
        self,
        model_runner: ModelRunner,
        context_appender: ContextAppender | None = None,
        max_turns: int | None = None,
        chat_history_config: dict[str, Any] = {"type": "local"},
        cache_config: dict[str, Any] | None = None,
    ) -> None:
        self.model_runner = model_runner
        self.context_appender = context_appender
        self.max_turns = max_turns
        self.turn_count = 0
        self.chat_history: ChatHistoryStorage = create_chat_history_storage(
            chat_history_config
        )
        self.cache_service: CacheService | None = (
            create_cache_service(cache_config) if cache_config else None
        )

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.chat_history.get_messages()

    def add_user_message(self, text: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            text: The content of the user message.
        """
        messages = self.messages
        messages.append({"role": "user", "content": [{"text": text}]})
        self.chat_history.save_messages(messages)
        self.turn_count += 1

    def clear_messages(self) -> None:
        """Clear all messages in the conversation."""
        self.chat_history.clear_messages()
        self.turn_count = 0

    def reset_messages(
        self,
        write_stream: Callable[
            [Generator[str | dict[str, Any], None, None]],
            tuple[str, dict[str, Any] | None],
        ]
        | None = None,
    ) -> bool:
        """Reset the conversation if max_turns is reached."""
        user_message_count = sum(
            1
            for message in self.messages
            if message["role"] == "user"
            and not any("toolResult" in content for content in message["content"])
        )

        if self.max_turns and user_message_count > self.max_turns:
            self.clear_messages()

            if write_stream:
                reset_message = "Conversation history reset due to max turns exceeded."
                response_generator = (chunk for chunk in [reset_message])
                write_stream(response_generator)

            return True
        return False

    def say(
        self,
        text: str,
        model_id: str,
        use_streaming: bool,
        invoke_limit: int | None = None,
        max_retries: int = 3,
        inference_config: dict[str, Any] | None = {"maxTokens": 4096, "temperature": 0},
        write_stream: Callable[
            [Generator[str | dict[str, Any], None, None]],
            tuple[str, dict[str, Any] | None],
        ]
        | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Generate a response from the model.

        Args:
            text: The user's input text.
            model_id: The ID of the model to use.
            use_streaming: Whether to use streaming or not.
            invoke_limit: Maximum number of model invocations (optional).
            max_retries: Maximum number of retry attempts (optional).
            inference_config: Configuration for model inference (optional).
            write_stream: Function to handle streaming output (optional).

        Returns:
            The model's response and response data.
        """
        start_time = time.time()

        if self.context_appender:
            text = self.context_appender.append_context(text)

        self.add_user_message(text)

        if self.reset_messages(write_stream=write_stream):
            return "Conversation history reset due to max turns exceeded.", None

        streamed_text, response_data = self._get_response(
            text,
            model_id,
            use_streaming,
            invoke_limit,
            max_retries,
            inference_config,
            write_stream,
        )

        self._update_conversation(streamed_text, response_data)

        if self.cache_service:
            self.cache_service.put(text, streamed_text)

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        return streamed_text, response_data

    def _get_response(
        self,
        text: str,
        model_id: str,
        use_streaming: bool,
        invoke_limit: int | None,
        max_retries: int,
        inference_config: dict[str, Any] | None,
        write_stream: Callable | None,
    ) -> tuple[str, dict[str, Any] | None]:
        if self.cache_service:
            cached_response = self.cache_service.get(text)
            if cached_response:
                return self._handle_cached_response(cached_response, write_stream)

        generate_text_args = self._build_generate_text_args(
            model_id, use_streaming, invoke_limit, max_retries, inference_config
        )
        response_generator = self.model_runner.generate_text(**generate_text_args)

        return self._process_response(response_generator, write_stream)

    def _handle_cached_response(
        self, cached_response: str, write_stream: Callable | None
    ) -> tuple[str, dict[str, Any] | None]:
        def cached_generator():
            yield cached_response
            message = {"role": "assistant", "content": [{"text": cached_response}]}
            messages = self.messages
            messages.append(message)
            self.chat_history.save_messages(messages)
            yield {"messages": messages}

        return self._process_response(cached_generator(), write_stream)

    def _build_generate_text_args(
        self,
        model_id: str,
        use_streaming: bool,
        invoke_limit: int | None,
        max_retries: int,
        inference_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        args = {
            "model_id": model_id,
            "messages": self.messages,
            "use_streaming": use_streaming,
            "invoke_limit": invoke_limit,
            "max_retries": max_retries,
        }
        if inference_config is not None:
            args["inference_config"] = inference_config
        return args

    def _process_response(
        self, response_generator: Generator, write_stream: Callable | None
    ) -> tuple[str, dict[str, Any] | None]:
        if write_stream:
            return write_stream(response_generator)
        return self.default_write_stream(response_generator)

    def _update_conversation(
        self, streamed_text: str, response_data: dict[str, Any] | None
    ) -> None:
        messages = self.messages
        if response_data and "messages" in response_data:
            messages = response_data["messages"]
        else:
            messages.append({"role": "assistant", "content": [{"text": streamed_text}]})
        self.chat_history.save_messages(messages)
        self.turn_count += 1

    def default_write_stream(
        self, stream: Generator[str | dict[str, Any], None, None]
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Default handler for streaming output.

        Args:
            stream: Generator yielding text chunks or final response data.

        Returns:
            The complete streamed text and the final response data (if any).
        """
        result = ""
        response_data: dict[str, Any] | None = None

        for item in stream:
            if isinstance(item, str):
                result += item
                print(item, end="", flush=True)  # Print each chunk as it arrives
            elif isinstance(item, dict):
                response_data = item

        print()  # New line after all chunks are printed
        return result, response_data
