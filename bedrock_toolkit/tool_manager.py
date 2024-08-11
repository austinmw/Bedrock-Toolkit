""" ToolManager class for managing tools and tool processing. """
import inspect
import json
import types
from typing import Any, Callable, Tuple, Type

from pydantic import BaseModel

from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()

class ToolManager:
    def __init__(
        self,
        tool_schemas: list[type[BaseModel]],
        tool_processors: dict[str, Callable[..., Any]] | None = None) -> None:
        """
        Initialize the ToolManager.

        Args:
            tool_schemas (list[Type[BaseModel]]): list of Pydantic model classes representing tool schemas.
            tool_processors (dict[str, Callable] | None): dictionary of tool processors. Keys are tool names, values are processor functions.
        """
        self.tool_schemas = tool_schemas
        self.tool_processors = tool_processors or {}
        self.pydantic_model_registry: dict[str, Type[BaseModel]] = {}
        self.register_models()

    def register_models(self) -> None:
        """
        Register all Pydantic models from the tool schemas, including nested models.

        This function populates the `pydantic_model_registry` dictionary with all the Pydantic models
        defined in `tool_schemas`, as well as any nested models. It ensures that all relevant models
        are registered and available for processing.
        """
        def register_pydantic_model(model: Type[BaseModel]) -> None:
            """Register a Pydantic model in the model registry."""
            self.pydantic_model_registry[model.__name__] = model

        def register_nested_models(model: Type[BaseModel], visited: set) -> None:
            """
            Recursively register nested models.

            Args:
                model (Type[BaseModel]): The Pydantic model to register.
                visited (set): A set of already visited models to avoid infinite recursion.
            """
            if model in visited:
                return
            visited.add(model)
            register_pydantic_model(model)
            for field in model.model_fields.values():
                field_type = field.annotation
                # Check if the field type is a subclass of BaseModel (Pydantic model)
                if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                    register_nested_models(field_type, visited)
                # Check if the field type is a generic list containing a subclass of BaseModel
                elif (isinstance(field_type, types.GenericAlias) and
                      field_type.__origin__ is list and
                      inspect.isclass(field_type.__args__[0]) and 
                      issubclass(field_type.__args__[0], BaseModel)):
                    register_nested_models(field_type.__args__[0], visited)

        visited_models: set = set()
        for model in self.tool_schemas:
            register_nested_models(model, visited_models)

    def format_tools(self) -> list[dict[str, Any]]:
        """Format the tools for the model input."""
        formatted_tools = [self._pydantic_to_toolspec(model) for model in self.tool_schemas]
        for n, formatted_tool in enumerate(formatted_tools):
            logger.debug(f"Tool {n + 1}:\n{json.dumps(formatted_tool, indent=4)}")
        return formatted_tools

    def _pydantic_to_toolspec(self, model: Type[BaseModel]) -> dict[str, Any]:
        """
        Convert a Pydantic model to a toolSpec dictionary, including additional field arguments.

        Args:
            model (Type[BaseModel]): A Pydantic model class.

        Returns:
            dict[str, Any]: A dictionary representing the toolSpec for the input model.
        """
        def convert_schema(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
            if "$ref" in schema:
                ref_key = schema["$ref"].split("/")[-1]
                return convert_schema(defs[ref_key], defs)

            result: dict[str, Any] = {}
            for key in ["type", "description"]:
                if key in schema:
                    result[key] = schema[key]

            for constraint in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems"]:
                if constraint in schema:
                    result[constraint] = schema[constraint]

            if "properties" in schema:
                result["properties"] = {
                    k: convert_schema(v, defs) for k, v in schema["properties"].items()
                }

            if "items" in schema:
                result["items"] = convert_schema(schema["items"], defs)

            if "required" in schema:
                result["required"] = schema["required"]

            return result

        schema = model.model_json_schema()
        defs = schema.get("$defs", {})

        description = model.__doc__.strip() if model.__doc__ else schema.get("description", f"Model for {schema['title']}")

        return {
            "toolSpec": {
                "name": schema["title"],
                "description": description,
                "inputSchema": {
                    "json": convert_schema(schema, defs)
                }
            }
        }

    def parse_model(self, tool: dict[str, Any]) -> BaseModel:
        """
        Parse a tool dictionary into the appropriate Pydantic model.

        Args:
            tool (dict[str, Any]): The tool dictionary to parse.

        Returns:
            BaseModel: The parsed Pydantic model instance.

        Raises:
            ValueError: If the input is invalid or the model is not found.
        """
        if not isinstance(tool, dict):
            raise ValueError("Input tool must be a dictionary.")

        model_name = tool.get('name')
        input_data = tool.get('input', {})

        if model_name is None or model_name not in self.pydantic_model_registry:
            raise ValueError("Invalid or missing 'name' in input tool.")

        model_class = self.pydantic_model_registry[model_name]
        return model_class.model_validate(input_data)

    def process_tool_use(self, tool: dict[str, Any]) -> Tuple[BaseModel, dict[str, Any]]:
        """
        Process the tool use request.

        Args:
            tool (dict[str, Any]): The tool use request.

        Returns:
            tuple: The processed tool call as a Pydantic model and the tool result as a dictionary.

        Raises:
            Exception: If there's an error in processing the tool use.
        """
        try:
            parsed_tool_call = self.parse_model(tool)
            tool_function = self.tool_processors.get(tool["name"], self._default_process_function)
            tool_output = tool_function(parsed_tool_call)

            tool_result = {
                "toolUseId": tool['toolUseId'],
                "content": [{"json": tool_output}]
            }

            logger.debug(f"Tool result: {json.dumps(tool_result, indent=4)}")

            return parsed_tool_call, tool_result

        except Exception as err:
            error_message = (
                "Pydantic model validation failed. The LLM failed to return a valid response.\n"
                f"Validation error details:\n{str(err)}\n\n"
                f"Tool response that caused this error:\n{json.dumps(tool['input'], indent=4)}\n\n"
            )
            logger.error(error_message)
            raise

    @staticmethod
    def _default_process_function(parsed_model: BaseModel) -> dict[str, Any]:
        """Default process function for tools."""
        return parsed_model.model_dump()
