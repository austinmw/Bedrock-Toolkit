"""Streamlit application for AI-powered trip planning."""

import streamlit as st
from typing import Any, Callable
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

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
    """
    Main function for the AI Trip Planner Streamlit application.

    This function sets up a Streamlit interface for an AI-powered trip planner. It demonstrates
    how an AI agent can use multiple APIs to gather information and plan a trip. The
    application uses three simulated APIs:

    1. WeatherForecast: Provides weather information for the destination.
    2. TouristAttractions: Retrieves top attractions for the destination.
    3. FlightSearch: Searches for available flights from the origin to the destination.

    These APIs are not real and use hardcoded responses. However, the example illustrates how
    the AI agent needs to understand the user's query, determine which APIs to call, and
    combine the information from all three APIs to generate a comprehensive trip plan.

    The AI agent's decision-making process in calling these APIs and synthesizing the
    information demonstrates its ability to break down complex tasks and gather relevant data
    from multiple sources to provide a complete response to the user's request.
    """
    st.title("AI Trip Planner")
    st.subheader("Plan Your Next Adventure with AI")

    # Step 1: Create sidebar options
    options = create_sidebar_options()

    # Step 2: Configure logger
    logger_manager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # Step 3: Define Pydantic models for tool inputs
    class WeatherForecast(BaseModel):
        city: str = Field(..., description="Name of the city")
        country: str = Field(..., description="Country where the city is located")
        start_date: str = Field(
            ..., description="Start date of the forecast (YYYY-MM-DD)"
        )
        end_date: str = Field(..., description="End date of the forecast (YYYY-MM-DD)")

    class TouristAttractions(BaseModel):
        city: str = Field(..., description="Name of the city")
        country: str = Field(..., description="Country where the city is located")
        num_attractions: int = Field(
            5, description="Number of top attractions to return"
        )

    class FlightSearch(BaseModel):
        origin: str = Field(..., description="Origin city")
        destination: str = Field(..., description="Destination city")
        date: str = Field(..., description="Date of travel (YYYY-MM-DD)")

    # Step 4: Define tool processors
    def get_weather_forecast(request: WeatherForecast) -> dict[str, Any]:
        logger.info(
            f"Getting weather forecast for {request.city}, {request.country}..."
        )
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        forecast = []
        for i in range(3):
            date = start_date + timedelta(days=i)
            forecast.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "temperature": 20 + i,
                    "condition": "Partly cloudy" if i % 2 == 0 else "Sunny",
                }
            )
        return {"forecast": forecast}

    def get_tourist_attractions(request: TouristAttractions) -> dict[str, Any]:
        logger.info(f"Getting top attractions for {request.city}, {request.country}...")
        attractions = [
            "Eiffel Tower",
            "Louvre Museum",
            "Notre-Dame Cathedral",
            "Arc de Triomphe",
            "Champs-Élysées",
        ]
        return {"attractions": attractions[: request.num_attractions]}

    def search_flights(request: FlightSearch) -> dict[str, Any]:
        logger.info(
            f"Searching flights from {request.origin} to {request.destination}..."
        )
        return {
            "flights": [
                {
                    "airline": "Air France",
                    "departure": "08:00",
                    "arrival": "21:00",
                    "price": "$800",
                },
                {
                    "airline": "Delta",
                    "departure": "10:30",
                    "arrival": "23:30",
                    "price": "$750",
                },
                {
                    "airline": "United",
                    "departure": "14:00",
                    "arrival": "03:00",
                    "price": "$700",
                },
            ]
        }

    # Step 5: Set up tools
    tool_schemas: list[type[BaseModel]] = [
        WeatherForecast,
        TouristAttractions,
        FlightSearch,
    ]
    tool_processors: dict[str, Callable[[Any], dict[str, Any]]] = {
        "WeatherForecast": get_weather_forecast,
        "TouristAttractions": get_tourist_attractions,
        "FlightSearch": search_flights,
    }

    # Step 6: Display available tools in the sidebar
    st.sidebar.title("Agent Capabilities")
    for tool in tool_processors.keys():
        st.sidebar.text(f"• {tool}")

    # Step 7: Initialize components
    bedrock_client = BedrockClient(region="us-east-1")
    tool_manager = ToolManager(tool_schemas, tool_processors)
    model_runner = ModelRunner(bedrock_client, tool_manager)

    # Main area for user input
    st.write("Enter your trip details:")
    city = st.text_input("Destination City", value="Paris")
    country = st.text_input("Destination Country", value="France")
    origin_city = st.text_input("Origin City", value="New York")

    if st.button("Plan My Trip"):
        user_prompt = f"""Plan a 3-day trip to {city}, {country} for next month.
        Provide information about the weather, top attractions, and available flights from {origin_city}.
        """

        try:
            with st.spinner("Planning your trip..."):
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

            st.success("Trip planning complete!")

        except Exception as e:
            logger.exception("An unexpected error occurred")
            st.error("An error occurred while planning your trip. Please try again.")
            st.exception(e)


if __name__ == "__main__":
    main()
