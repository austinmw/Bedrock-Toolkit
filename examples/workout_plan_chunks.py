import streamlit as st
import pandas as pd
from typing import Any, List
from pydantic import BaseModel, Field
from typing import Annotated
from annotated_types import Len
from textwrap import dedent
import math
import time

from bedrock_toolkit.logger_manager import LoggerManager
from bedrock_toolkit.bedrock_client import BedrockClient
from bedrock_toolkit.tool_manager import ToolManager
from bedrock_toolkit.model_runner import ModelRunner
from bedrock_toolkit.conversation_manager import ConversationManager
from bedrock_toolkit.streamlit_utils import (
    create_sidebar_options,
    chat_write_stream,
    display_model_response,
    display_messages,
)

st.set_page_config(layout="wide")

SYSTEM_PROMPT = {
    "text": dedent(
        """\
        You are an intelligent conversational assistant capable of calling functions to gather all necessary information to answer a user's question comprehensively. 
        Note that all of your standard text responses will be displayed using markdown, so you can use markdown formatting as needed. 
        Avoid using markdown titles unless they are appropriate for structuring a detailed response—remember, you are having a conversation. 
        Each response should build upon the previous one, combining all gathered information into a final, standalone answer. 
        Ensure the final response fully addresses the user's question, presenting it as if you are answering from scratch, without assuming the user remembers any information from previous responses. 
        Finally, you should not include a top-level 'properties' or 'json' key in tool call requests.

        Additionally, you are an expert fitness coach capable of creating workout plans perfectly suited to individuals of all levels of fitness and experience. 
        The workout plans you create should demonstrate slight progression in difficulty every few weeks.
        Keep explanations for the workout plan clear and very concise; a max of 4 sentences.
        Don't detail every day in the explanatons, just talk about the high level philosophy of the plan.
        Make sure to highlight the user's goals and how the plan is tailored to achieve them.
        """
    )
}

USER_PROMPT_TEMLATE = dedent(
    """\
    At Echelon Fitness, our training philosophy is rooted in making fitness accessible, engaging, and community-driven for all individuals, regardless of their starting point. 
    As an expert personal trainer, your role is to create workout plans that foster consistency, variety, and motivation, ensuring each session is inclusive and adaptable to an indivual's fitness level. 
    We emphasize the importance of data-driven progress and expert guidance, encouraging you to utilize performance metrics to personalize the experience and help each member reach their goals. 
    Above all, cultivate a sense of community in your sessions, where members feel supported and inspired to push their limits while enjoying the journey together.

    Create a workout program for the below person:\n
    Age: {age}\n
    Weight: {weight} lbs\n
    Gender: {gender}\n
    Height: {height_ft}'{height_in}\"\n
    Body Fat Percentage: {body_fat}%\n
    Fitness Level: {fitness_level}\n
    Preferences: {preferences}\n
    Health Considerations: {health_considerations}\n

    Goals: {goals}\n

    Total days: {total_days}\n
    """
)

# Set a maximum number of days to generate a workout plan for at a time
# This helps the model to generate a plan for a large number of days without running into errors
MAX_DAYS_PER_CHUNK = 14

@st.cache_resource
def get_workout_plan_class(num_days: int):
    """ Dynamically generate a Pydantic model for a workout plan with a specific number of days. """

    class WorkoutDay(BaseModel):
        description: str = Field(description="General description of the workout for this day.")
        duration: int = Field(ge=0, description="Duration of the workout in minutes. Every day requires a duration. Rest days should have a duration of 0.")
        focus_area: str = Field(description="Primary focus area or activity type (e.g., upper body, running).")
        exercises: List[str] = Field(default_factory=list, description="List of key exercises for the day in the format 'Exercise Name: Sets x Reps', 'Exercise Name: Time', or 'Exercise Name: Time x Distance', etc.")

    class WorkoutPlan(BaseModel):
        """ A workout plan for a specific number of days. """
        days: Annotated[List[WorkoutDay], Len(num_days)] = Field(description=f"List of exactly {num_days} workout days.")

        @property
        def total_duration(self) -> int:
            return sum(day.duration for day in self.days)

        @property
        def rest_days(self) -> int:
            return sum(1 for day in self.days if day.duration == 0)

        def get_day(self, day_number: int) -> WorkoutDay:
            if 1 <= day_number <= len(self.days):
                return self.days[day_number - 1]
            raise ValueError(f"Day number must be between 1 and {len(self.days)}.")

    return WorkoutPlan

def update_workout_dataframe(workout_plan, start_day, existing_df=None) -> pd.DataFrame:
    """ Update the workout plan dataframe with the new days. """

    data = []
    for day_num, day in enumerate(workout_plan.days, start=start_day):
        data.append({
            'Day': day_num,  # Day numbers will start from `start_day`
            'Description': day.description,
            'Duration (minutes)': day.duration,
            'Focus Area': day.focus_area,
            'Exercises': ', '.join(day.exercises),
        })

    new_df = pd.DataFrame(data)
    new_df.set_index('Day', inplace=True)

    if existing_df is not None:
        # Append new days to the existing dataframe
        return pd.concat([existing_df, new_df])
    else:
        return new_df

def display_workout_plan(df) -> None:
    """ Display the workout plan in a DataFrame format. """

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

    st.subheader(f"Your Workout Plan (Days 1-{len(df)})")

    def highlight_zero(val):
        return ['background-color: lightgreen' if v == 0 else '' for v in val]

    st.dataframe(df.style.apply(highlight_zero, subset=['Duration (minutes)']), use_container_width=True)

    st.write(f"Total duration so far: {df['Duration (minutes)'].sum()} minutes")
    st.write(f"Number of rest days so far: {(df['Duration (minutes)'] == 0).sum()}")


def main() -> None:
    st.title("Echelon Fitness AI Workout Planner")
    st.subheader("Personalized Fitness Plans for All Levels")
    st.markdown("⚡️ Powered by Amazon Bedrock")

    # Step 1: Create sidebar options
    options = create_sidebar_options(
        default_model="anthropic.claude-3-sonnet-20240229-v1:0",
        default_invoke_limit=2,
    )

    # Step 2: Configure logger with sidebar option
    logger_manager = LoggerManager()
    logger_manager.configure_logger(log_level=options["log_level"])
    logger = logger_manager.get_logger()

    # Step 3: Define your Pydantic models (done in get_workout_plan_class function)

    # Step 4: Define your tool processors
    def process_workout_plan(workout_plan: BaseModel) -> dict[str, Any]:
        logger.info("Processing workout plan...")
        workout_plan_dict: dict[str, Any] = workout_plan.model_dump()
        logger.debug(f"Workout plan:\n{workout_plan_dict}")
        return workout_plan_dict

    # Step 5: Set up your tools (will be done dynamically for each chunk)

    # Step 6: Display available tools in the sidebar
    st.sidebar.title("Agent Capabilities")
    st.sidebar.text("• WorkoutPlan generation")

    # Step 7: Initialize components, caching resources to maintain state
    @st.cache_resource
    def get_bedrock_client() -> BedrockClient:
        return BedrockClient(region="us-east-1")

    @st.cache_resource
    def get_tool_manager() -> ToolManager:
        return ToolManager([], {})  # Start with empty tools

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
            max_turns=100,
        )

    # Initialize or retrieve session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_manager()

    if "workout_dataframe" not in st.session_state:
        st.session_state.workout_dataframe = None

    # # Display chat messages from history on app rerun
    # for message in st.session_state.conversation.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # Accept user input
    with st.form("user_input"):
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        weight = st.number_input("Weight (lbs)", min_value=80, max_value=400, value=167)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height_ft = st.number_input("Height (feet)", min_value=4, max_value=7, value=5)
        height_in = st.number_input("Height (inches)", min_value=0, max_value=11, value=11)
        body_fat = st.number_input("Body Fat Percentage", min_value=5, max_value=70, value=17)
        fitness_level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"], index=1)
        goals = st.text_input("Fitness goals", value="Get to 12% bodyfat without dropping below 162 lbs")
        preferences = st.text_input("Workout preferences", value="None")
        health_considerations = st.text_input("Health considerations", value="None")
        total_days = int(st.number_input("Total number of days for the workout plan", min_value=1, max_value=365, value=30))

        # Submit button to generate the workout plan
        submit_button = st.form_submit_button("Generate Workout Plan")

    if submit_button:
        # Clear messages
        st.session_state.conversation.clear_messages()

        start_time = time.time()
        success = True

        # Set the initial workout plan dataframe to None
        if "workout_dataframe" not in st.session_state:
            st.session_state.workout_dataframe = None

        # Base prompt for the workout plan
        base_prompt = USER_PROMPT_TEMLATE.format(
            age=age,
            weight=weight,
            gender=gender,
            height_ft=height_ft,
            height_in=height_in,
            body_fat=body_fat,
            fitness_level=fitness_level,
            preferences=preferences,
            health_considerations=health_considerations,
            goals=goals,
            total_days=total_days,
        )

        # Calculate the number of chunks based on the total number of days
        chunks = math.ceil(total_days / MAX_DAYS_PER_CHUNK)

        # Loop through each chunk of days
        for chunk in range(chunks):
            start_day = chunk * MAX_DAYS_PER_CHUNK + 1
            end_day = min(start_day + MAX_DAYS_PER_CHUNK - 1, total_days)
            days_in_chunk = end_day - start_day + 1

            # Dynamically get the workout plan class based on the number of days in the chunk
            WorkoutPlan = get_workout_plan_class(days_in_chunk)

            # Update tool manager for this chunk
            tool_schemas = [WorkoutPlan]
            tool_processors = {"WorkoutPlan": process_workout_plan}

            # Update tool manager for this chunk
            st.session_state.conversation.model_runner.tool_manager = ToolManager(tool_schemas, tool_processors)

            # Update the user prompt based on the chunk
            # The first chunk will have a different prompt than the subsequent chunks
            if chunk == 0:
                user_prompt = dedent(
                    f"""\
                    {base_prompt}\nCreate the plan for days {start_day}-{end_day}.
                    """
                )
            # For subsequent chunks, mention the previous days and slight progression
            else:
                user_prompt = dedent(
                    f"""\
                    Create the plan for days {start_day}-{end_day}, reflecting on the previous {start_day - 1} 
                    days, while adhering to the overall goals and progression of the plan.
                    """
                )

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Create a placeholder for the assistant's response
            assistant_placeholder = st.chat_message("assistant")

            # Use a context manager to display the assistant's response in the chat message container
            with assistant_placeholder:
                with st.spinner(f"Generating your personalized workout plan for days {start_day}-{end_day}..."):
                    try:

                        # Call the model and process the output
                        streamed_text, response_data = st.session_state.conversation.say(
                            text=user_prompt,
                            model_id=options["model_id"],
                            use_streaming=options["use_streaming"],
                            invoke_limit=options["invoke_limit"],
                            max_retries=options["max_retries"],
                            write_stream=chat_write_stream
                        )

                        # Store the workout plan in a dataframe
                        for item in response_data["tool_calls"]:
                            if isinstance(item, WorkoutPlan):
                                st.session_state.workout_dataframe = update_workout_dataframe(
                                    item, start_day, st.session_state.workout_dataframe
                                )
                                # Display the workout plan
                                display_workout_plan(st.session_state.workout_dataframe)

                        # Add assistant response to chat history
                        display_model_response(streamed_text, response_data)

                    except Exception as e:
                        logger.exception("An unexpected error occurred")
                        st.error("An error occurred. Consider increasing the max retries.")
                        st.exception(e)

                        display_messages(st.session_state.conversation.messages)
                        success = False

        if success:
            elapsed_time = time.time() - start_time
            st.success(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
