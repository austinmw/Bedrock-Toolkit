import streamlit as st
import pandas as pd
from typing import Any, Callable, Dict, List
from pydantic import BaseModel, Field
from typing import Annotated
from annotated_types import Len

from bedrock_toolkit.logger_manager import LoggerManager
from bedrock_toolkit.bedrock_client import BedrockClient
from bedrock_toolkit.tool_manager import ToolManager
from bedrock_toolkit.model_runner import ModelRunner
from bedrock_toolkit.conversation_manager import ConversationManager
from bedrock_toolkit.streamlit_utils import (
    clear_chat_history,
    create_sidebar_options,
    chat_write_stream,
    display_model_response,
    display_messages,
)

st.set_page_config(layout="wide")

# Define your Pydantic model
@st.cache_resource
def get_workout_plan_class(num_days: int = 30):
    class WorkoutDay(BaseModel):
        description: str = Field(description="Description of the workout for this day.")
        duration: int = Field(ge=0, description="Duration of the workout in minutes. Every day requires a duration, even rest days (use 0).")

    class WorkoutPlan(BaseModel):
        days: Annotated[List[WorkoutDay], Len(num_days)] = Field(description=f"List of exactly {num_days} workout days.")

        @property
        def total_duration(self) -> int:
            """Calculate the total duration of the workout plan in minutes."""
            return sum(day.duration for day in self.days)

        @property
        def rest_days(self) -> int:
            """Calculate the number of rest days (days with duration 0) in the plan."""
            return sum(1 for day in self.days if day.duration == 0)

        def get_day(self, day_number: int) -> WorkoutDay:
            """Get a specific day's workout plan."""
            if 1 <= day_number <= len(self.days):
                return self.days[day_number - 1]
            raise ValueError(f"Day number must be between 1 and {len(self.days)}.")

    return WorkoutPlan


def display_workout_plan(workout_plan):

    # Custom CSS to make the table full-width
    st.markdown("""
    <style>
    .stDataFrame {
        width: 100%;
    }
    .stDataFrame > div {
        width: 100%;
    }
    .stDataFrame table {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Your Workout Plan")

    # Create a list to hold all the data
    data = []

    # Iterate through each day
    for day_num, day in enumerate(workout_plan.days, start=1):
        data.append({
            'Day': day_num,
            'Description': day.description,
            'Duration (minutes)': day.duration
        })

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Set index
    df.set_index('Day', inplace=True)

    # Display the DataFrame
    def highlight_zero(val):
        return ['background-color: lightgreen' if v == 0 else '' for v in val]

    # Use a container to make the table full-width
    with st.container():
        st.dataframe(df.style.apply(highlight_zero, subset=['Duration (minutes)']), use_container_width=True)

    # Access properties of the workout plan
    st.write(f"Total duration: {workout_plan.total_duration} minutes")
    st.write(f"Number of rest days: {workout_plan.rest_days}")


def main() -> None:
    """ Workout Planner App """

    st.title("Workout Planner")
    st.subheader("Personalized Cardio and Weight Loss Program")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "workout_plan" not in st.session_state:
        st.session_state.workout_plan = None

    # Step 1: Create sidebar options
    options = create_sidebar_options(
        default_model="anthropic.claude-3-sonnet-20240229-v1:0",
        default_invoke_limit=2,
    )

    # Step 2: Configure logger with sidebar option
    logger_manager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # User input form
    with st.form("user_input"):
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        weight = st.number_input("Weight (lbs)", min_value=80, max_value=400, value=175)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height_ft = st.number_input("Height (feet)", min_value=4, max_value=7, value=6)
        height_in = st.number_input("Height (inches)", min_value=0, max_value=11, value=0)
        num_days = int(st.number_input("Number of days", min_value=1, max_value=90, value=30))

        submit_button = st.form_submit_button("Generate Workout Plan")

    if submit_button:
        # Step 3: Define your Pydantic models
        WorkoutPlan = get_workout_plan_class(num_days)

        # Step 4: Define your tool processors
        def process_workout_plan(workout_plan: BaseModel) -> Dict[str, Any]:
            logger.info("Processing workout plan...")
            workout_plan_dict: dict[str, Any] = workout_plan.model_dump()
            logger.info(f"Workout plan:\n{workout_plan_dict}")
            return workout_plan_dict

        # Step 5: Set up your tools
        tool_schemas: List[type[BaseModel]] = [WorkoutPlan]
        tool_processors: Dict[str, Callable[[BaseModel], Dict[str, Any]]] = {
            "WorkoutPlan": process_workout_plan,
        }

        # Step 6: Initialize components
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
            return ConversationManager(
                model_runner=get_model_runner(),
                max_turns=3,
            )

        # Initialize or retrieve session state variables
        if "conversation" not in st.session_state:
            st.session_state.conversation = get_conversation_manager()

        # Construct the user prompt
        height = f"{height_ft}'{height_in}\""
        user_prompt = (
            "Create a workout program for the below person:\n\n"
            f"Age: {age}\n"
            f"Weight: {weight} lbs\n"
            f"Gender: {gender}\n"
            f"Height: {height}\n"
            f"Number of days: {num_days}\n"
            "\nI want the program to focus on cardio and weight loss."
        )

        # Add user message to chat history
        st.session_state.messages = [{"role": "user", "content": user_prompt}]

        # Display user message in chat message container
        with st.chat_message("user"):
            st.text(user_prompt)

        # Create a placeholder for the assistant's response
        assistant_placeholder = st.chat_message("assistant")

        # Display assistant response in chat message container
        with assistant_placeholder:
            with st.spinner("Generating your personalized workout plan..."):
                try:
                    # Clear messages
                    st.session_state.conversation.clear_messages()

                    # Call the model and process the output
                    streamed_text, response_data = st.session_state.conversation.say(
                        text=user_prompt,
                        model_id=options["model_id"],
                        use_streaming=options["use_streaming"],
                        invoke_limit=options["invoke_limit"],
                        max_retries=options["max_retries"],
                        write_stream=chat_write_stream
                    )

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": streamed_text})

                    # Store the workout plan
                    if response_data:
                        for item in response_data["tool_calls"]:
                            if isinstance(item, WorkoutPlan):
                                st.session_state.workout_plan = item

                    # Display the workout plan
                    if st.session_state.workout_plan:
                        display_workout_plan(st.session_state.workout_plan)

                    # Display the response
                    display_model_response(streamed_text, response_data)

                except Exception as e:
                    logger.exception("An unexpected error occurred")
                    st.error("An error occurred. Consider increasing the max retries.")
                    st.exception(e)
                    display_messages(st.session_state.conversation.messages)

if __name__ == "__main__":
    main()