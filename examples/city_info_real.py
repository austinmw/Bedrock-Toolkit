"""LLM Agent Weather and City Information Example (Real APIs)"""

import streamlit as st
from typing import Any, Callable
from pydantic import BaseModel, Field
import requests
import wikipediaapi  # type: ignore

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
    """LLM Agent Weather and City Information Example (Real APIs)

    This example shows the assistant using two real API tools:

    1. WeatherRequest:
        Gets actual current weather data for a city and country using a weather API.
    2. CityInfoRequest:
        Retrieves real background information about a location using Wikipedia's API.

    The assistant needs to decide which tools to use based on the user's question
    and integrate real-time data from both APIs into a single, informative response.
    This example interacts with live data sources.
    """

    st.title("LLM Agent")
    st.subheader("City Weather and Info Example (Real APIs)")

    # Step 1: Create sidebar options (excluding tool display for now)
    options = create_sidebar_options(
        default_model="anthropic.claude-3-sonnet-20240229-v1:0"
    )

    # Step 2: Configure logger with sidebar option
    logger_manager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # Step 3: Define your Pydantic models
    class WeatherRequest(BaseModel):
        """Model for requesting the current weather from a location."""

        location: str = Field(
            ...,
            description="The name of the location for which you want the current weather. Example locations are New York and London.",
        )

    class LocationInfoRequest(BaseModel):
        """Model for requesting background information about a location."""

        location: str = Field(
            ...,
            description="The name of the location for which you want the background information. Example locations are New York and London.",
        )
        char_limit: int = Field(
            2000,
            description="The maximum number of characters to return from the Wikipedia page. Default is 2000.",
        )

    # Step 4: Define your tool processors
    class LocationNotFoundError(Exception):
        """Raised when a location isn't found."""

        pass

    def get_current_weather(location: str) -> dict[str, str] | None:
        """Returns the current weather for the requested location using wttr.in."""
        logger.info(f"Getting weather information for {location}...")
        try:
            response = requests.get(f"https://wttr.in/{location}?format=%t+%C&u")
            if response.status_code == 200:
                weather_data = response.text.strip().split()
                return {
                    "temperature": weather_data[0],
                    "condition": " ".join(weather_data[1:]),
                }
            else:
                raise LocationNotFoundError(f"Location {location} not found.")
        except requests.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            raise LocationNotFoundError(f"Location {location} not found.")

    def get_location_info(
        location: str, char_limit: int = 2000
    ) -> dict[str, str] | None:
        """Returns background information for the requested location using Wikipedia."""
        logger.info(f"Getting background information for {location}...")
        headers = {
            "User-Agent": "LocationInfoBot/1.0 (https://example.org/locationbot/; locationbot@example.org)"
        }

        wiki_wiki = wikipediaapi.Wikipedia(
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=headers["User-Agent"],
        )
        page = wiki_wiki.page(location)

        if page.exists():
            full_text = page.text[:char_limit]
            return {"full_text": full_text}
        else:
            raise LocationNotFoundError(f"Location {location} not found on Wikipedia.")

    def process_weather_request(request: WeatherRequest) -> dict[str, Any]:
        try:
            weather_data = get_current_weather(request.location)
            return {"weather": weather_data}
        except LocationNotFoundError as e:
            return {"error": str(e)}

    def process_location_info_request(request: LocationInfoRequest) -> dict[str, Any]:
        try:
            location_info = get_location_info(request.location, request.char_limit)
            return {"info": location_info}
        except LocationNotFoundError as e:
            return {"error": str(e)}

    # Step 5: Set up your tools
    tool_schemas: list[type[BaseModel]] = [WeatherRequest, LocationInfoRequest]
    tool_processors: dict[str, Callable[[Any], dict[str, Any]]] = {
        "WeatherRequest": process_weather_request,
        "LocationInfoRequest": process_location_info_request,
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
