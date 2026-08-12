import streamlit as st
from groq import Groq
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="AI Travel Assistant",
    page_icon="✈️",
    layout="wide"
)

# -------------------------------------------------
# Groq API Key
# -------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -------------------------------------------------
# Title and Description
# -------------------------------------------------

st.title("✈️ AI Travel Assistant")

st.write(
    "Your intelligent travel companion for destinations, "
    "attractions, food, hotels and travel tips."
)

# -------------------------------------------------
# Initialize Conversation History
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header("🌍 Travel Settings")

destination = st.sidebar.text_input(
    "📍 Travel Destination",
    placeholder="Example: Mumbai"
)

st.sidebar.metric(
    "Conversation Messages",
    len(st.session_state.messages)
)

# -------------------------------------------------
# Clear Chat
# -------------------------------------------------

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -------------------------------------------------
# Display Previous Messages
# -------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# Chat Input
# -------------------------------------------------

user_input = st.chat_input(
    "Ask me anything about your trip..."
)

# -------------------------------------------------
# User Query
# -------------------------------------------------

if user_input:

    if not destination.strip():
        st.warning(
            "Please enter a travel destination in the sidebar first."
        )
        st.stop()

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------------------------
    # System Prompt
    # -------------------------------------------------

    system_prompt = f"""
You are an intelligent AI Travel Assistant.

The user's selected destination is: {destination}

Provide helpful and personalized travel recommendations
about:

- Tourist attractions
- Places to visit
- Food
- Hotels
- Transportation
- Activities
- Travel tips
- Itinerary planning
- Budget travel

Use the previous conversation to maintain context.

Give clear, useful and organized answers.
"""

    # -------------------------------------------------
    # Groq API Call
    # -------------------------------------------------

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                *st.session_state.messages
            ],
            temperature=0.7
        )

        assistant_response = response.choices[0].message.content

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Save AI response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

    except Exception as e:

        st.error(
            f"Error while connecting to Groq: {e}"
        )

# -------------------------------------------------
# Download Conversation
# -------------------------------------------------

conversation_text = ""

for message in st.session_state.messages:

    if message["role"] == "user":
        role = "You"
    else:
        role = "AI Travel Assistant"

    conversation_text += (
        f"{role}:\n"
        f"{message['content']}\n\n"
    )

st.sidebar.download_button(
    label="⬇️ Download Conversation",
    data=conversation_text,
    file_name="travel_conversation.txt",
    mime="text/plain"
)