""" ModelRunner class for running a model and processing the output. """
import json
from collections import Counter
from typing import Any, Generator, Literal

from bedrock_toolkit.logger_manager import LoggerManager
from bedrock_toolkit.prompts import ERROR_PROMPT_SUFFIX, SYSTEM_PROMPT


logger = LoggerManager.get_logger()

class ModelRunner:
    def __init__(
        self,
        bedrock_client,
        tool_manager,
        system_prompt: dict[str, str] = SYSTEM_PROMPT,
        first_tool_choice: Literal["any", "auto"] = "any",
    ) -> None:
        """
        Initialize the ModelRunner.

        Args:
            bedrock_client: An instance of the BedrockClient class.
            tool_manager: An instance of the ToolManager class.
            first_tool_choice:
                - "any": LLM assistant must use a tool on the first turn.
                - "auto": LLM assistant can choose whether to use a tool on the first turn.
                Note: After the first turn, this setting is ignored and "auto" is used.
        """
        self.bedrock_client = bedrock_client
        self.tool_manager = tool_manager
        self.system_prompt = system_prompt
        self.first_tool_choice = first_tool_choice

    def generate_text(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        use_streaming: bool,
        inference_config: dict[str, Any] = {"maxTokens": 4096, "temperature": 0},
        invoke_limit: int | None = None,
        max_retries: int = 3
    ) -> Generator[str | dict[str, Any], None, None]:
        """
        Generate text using the supplied Amazon Bedrock model with built-in retry logic.

        Args:
            model_id (str): The ID of the model to use.
            messages (list[dict[str, Any]]): The conversation history.
            use_streaming (bool): Whether to use streaming or not.
            invoke_limit (int | None): Maximum number of model invocations.
            max_retries (int): Maximum number of retry attempts.

        Yields:
            str | dict[str, Any]: Generated text chunks or final response data.
        """

        self.inference_config = inference_config

        original_messages = messages.copy()
        original_user_message = original_messages[-1]['content'][0]['text']
        error_message_pairs = []

        for attempt in range(max_retries):
            try:
                yield from self._generate_text_core(model_id, messages, use_streaming, invoke_limit)
                return

            except Exception as e:
                # Yield a newline to separate the error message from subsequent messages
                yield "\n\n"

                logger.warning(
                    f"Attempt {attempt + 1} failed. Exception:\n{str(e)}\n\n"
                    "Note: If you see failures and retry attempts often, you should improve the Pydantic model definitions and docstring."
                )
                if attempt == max_retries - 1:
                    raise

                # Extract the assistant's last failed response's 'input' parts
                last_assistant_message = messages[-1] if messages[-1]['role'] == 'assistant' else None

                tool_inputs = []

                if last_assistant_message:
                    for content in last_assistant_message['content']:
                        if 'toolUse' in content:
                            tool_inputs.append(json.dumps(content['toolUse']['input'], indent=4))

                # Add the current error-message pair to the list, including only the 'input' parts
                error_message_pairs.append((str(e), tool_inputs))

                # Reset messages to original state
                messages = original_messages.copy()

                # Create error_context with all previous error-message pairs
                error_context = "\n\nNote: Previous attempts resulted in errors. Here's a summary:\n"
                for i, (err, inputs) in enumerate(error_message_pairs, 1):
                    error_context += f"\nAttempt {i} error:\n{err}\n"
                    for input_msg in inputs:
                        error_context += f"The tool call(s) that caused this error was:\n{input_msg}\n"

                error_context += ERROR_PROMPT_SUFFIX

                # Update the last user message with cumulative error information
                messages[-1]['content'][0]['text'] = original_user_message + error_context


        raise Exception(f"All {max_retries} attempts failed")

    def _stream_messages(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        system_prompt: dict,
        tool_config: dict[str, Any]
    ) -> Generator[str | tuple[dict, str, dict[str, Any], Counter], None, None]:
        """Streams messages to a model and processes the response."""
        response = self.bedrock_client.converse_stream(model_id, messages, system_prompt, tool_config, self.inference_config)

        stop_reason = ""
        message: dict = {"content": []}
        text = ''
        tool_use = {}
        stream_usage: Counter = Counter()

        try:
            for chunk in response['stream']:
                if 'metadata' in chunk:
                    stream_usage.update(chunk['metadata']['usage'])
                if 'messageStart' in chunk:
                    message['role'] = chunk['messageStart']['role']
                elif 'contentBlockStart' in chunk:
                    tool = chunk['contentBlockStart']['start']['toolUse']
                    tool_use = {'toolUseId': tool['toolUseId'], 'name': tool['name']}
                elif 'contentBlockDelta' in chunk:
                    delta = chunk['contentBlockDelta']['delta']
                    if 'toolUse' in delta:
                        tool_use['input'] = tool_use.get('input', '') + delta['toolUse']['input']
                    elif 'text' in delta:
                        text += delta['text']
                        yield delta['text']  # Yield the text chunk
                elif 'contentBlockStop' in chunk:
                    if 'input' in tool_use:
                        tool_use['input'] = json.loads(tool_use['input'])
                        message['content'].append({'toolUse': tool_use})
                        tool_use = {}
                    else:
                        message['content'].append({'text': text})
                        text = ''
                elif 'messageStop' in chunk:
                    stop_reason = chunk['messageStop']['stopReason']
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
        finally:
            # Ensure we always return the values, even if an exception occurred
            yield response, stop_reason, message, stream_usage


    def _generate_text_core(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        use_streaming: bool,
        invoke_limit: int | None,
    ) -> Generator[str | dict[str, Any], None, None]:
        """
        Core logic for generating text using the Amazon Bedrock model.

        Args:
            model_id (str): The ID of the model to use.
            messages (list[dict[str, Any]]): The conversation history.
            use_streaming (bool): Whether to use streaming or not.
            invoke_limit (int | None): Maximum number of model invocations.

        Yields:
            str | dict[str, Any]: Generated text chunks or final response data.
        """
        stop_reason = 'tool_use'
        first_tool_use = True
        tool_calls = []
        invoke_count = 0
        usage: Counter = Counter()

        tool_config = {
            "tools": self.tool_manager.format_tools(),
            "toolChoice": {self.first_tool_choice: {}}
        }

        while stop_reason == 'tool_use':
            # Set the toolChoice to auto after the first tool use to allow the model to respond without tools
            if not first_tool_use:
                tool_config["toolChoice"] = {"auto": {}}
            first_tool_use = False

            logger.debug("Messages:")
            for message in messages:
                role = message["role"]
                context = message["content"]
                logger.debug(f"{role}:")
                for part in context:
                    if 'text' in part:
                        logger.debug(f"text:\n{part['text']}")
                    elif 'toolUse' in part:
                        logger.debug(f"toolUse:\n{json.dumps(part['toolUse'], indent=4)}")
                    elif 'toolResult' in part:
                        logger.debug(f"toolResult:\n{json.dumps(part['toolResult'], indent=4)}")
                    else:
                        raise ValueError(f"Unexpected message part: {part}")

            logger.debug("Sending messages to model...")

            if use_streaming:
                generator = self._stream_messages(
                    model_id,
                    messages,
                    self.system_prompt,
                    tool_config,
                )

                if invoke_count > 0:
                    yield "\n\n"  # Add newlines  between streamed responses

                for item in generator:
                    if isinstance(item, str):
                        yield item  # Yield the text chunk
                    else:
                        # This is the final yield with the return values
                        response, stop_reason, message, stream_usage = item

                logger.debug(f"Bedrock response (streamed):\n{response}")
                usage.update(stream_usage)

            else:
                response = self.bedrock_client.converse(
                    model_id,
                    messages,
                    self.system_prompt,
                    tool_config,
                    self.inference_config
                )

                message = response['output']['message']
                stop_reason = response['stopReason']

                for content in message['content']:
                    if 'text' in content:
                        yield content['text']

                logger.debug(f"Bedrock response:\n{json.dumps(response, indent=4)}")
                usage.update(response["usage"])

            messages.append(message)

            if stop_reason == 'tool_use':
                tool_results = []
                for content in message['content']:
                    if 'toolUse' in content:
                        tool = content['toolUse']
                        logger.debug(f"Requesting tool {tool['name']}. Request: {tool['toolUseId']}. Inputs: {tool['input']}")
                        parsed_tool_call, tool_result = self.tool_manager.process_tool_use(tool)
                        tool_calls.append(parsed_tool_call)
                        tool_results.append(tool_result)

                if tool_results:
                    tool_result_message = {
                        "role": "user",
                        "content": [{"toolResult": result} for result in tool_results]
                    }
                    messages.append(tool_result_message)

            invoke_count += 1

            if invoke_limit is not None and invoke_count >= invoke_limit:
                logger.info(f"Reached invoke limit of {invoke_limit}. Stopping generation.")
                break

        logger.debug(f"Total token usage:\n{json.dumps(usage, indent=2)}")

        total_price = self.bedrock_client.calculate_cost(usage, model_id)
        total_price_str = f"${total_price:.3f} USD"
        logger.info(f"Last reply cost: {total_price_str}")
 
        response_dict = {
            "messages": messages,
            "tool_calls": tool_calls,
            "usage": {**usage, "totalPrice": total_price_str}
        }

        yield response_dict
