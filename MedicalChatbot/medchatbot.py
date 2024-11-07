import os
import google.generativeai as genai
import streamlit as st

# Set up environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/langchain endpoints/langchain-439407-c27f03a83218.json"
GEMINI_API_KEY = "AIzaSyCjgg1kdbGp1ze7UKRwb6ig6aaUzQForiQ"
genai.configure(api_key=GEMINI_API_KEY)

# Load the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")

# Function to generate follow-up question or diagnosis based on chat history
def generate_response(chat_history, is_collecting_info, question_count):
    if is_collecting_info:
        # Collect basic information in sequence: Name, Age, and Sex
        if not st.session_state.get("name"):
            return "Can I have your name, please?", False
        elif not st.session_state.get("age"):
            return "Thank you. Can I have your age?", False
        elif not st.session_state.get("sex"):
            return "Thank you. What is your sex (Male/Female/Other)?", False
        else:
            # All basic info collected; transition to symptom questions
            st.session_state['collecting_info'] = False
            return "Thank you for providing your information. Let's discuss your symptoms now.", False
    elif question_count >= 8:  # Threshold for enough diagnostic questions
        # Generate diagnosis and treatment
        conversation = "\n".join([f"{role}: {message}" for role, message in chat_history])
        prompt = f"{conversation}\n\nBased on the provided information, give a diagnosis and suggest possible medications and treatments."

        diagnosis_chat = model.start_chat(history=[{"role": "user", "parts": [{"text": prompt}]}])
        diagnosis_response = diagnosis_chat.send_message(prompt)

        diagnosis = ""
        for chunk in diagnosis_response:
            if hasattr(chunk, 'text') and chunk.text:
                diagnosis = chunk.text.strip()
                break

        # Follow-up prompt after diagnosis
        follow_up_prompt = "Is there any additional information you’d like to provide for a more detailed diagnosis, or type 'stop' to end the session."
        return f"{diagnosis}\n\n{follow_up_prompt}", True  # True indicates that a diagnosis has been given
    else:
        # Continue with follow-up questions based on current context
        conversation = "\n".join([f"{role}: {message}" for role, message in chat_history])
        prompt = f"{conversation}\n\nAs a medical assistant, please ask a relevant follow-up question based on the user's responses so far."

        follow_up_chat = model.start_chat(history=[{"role": "user", "parts": [{"text": prompt}]}])
        follow_up_response = follow_up_chat.send_message(prompt)

        follow_up_question = ""
        for chunk in follow_up_response:
            if hasattr(chunk, 'text') and chunk.text:
                follow_up_question = chunk.text.strip()
                break

        return follow_up_question, False  # False indicates more questions are needed

# Function to handle question submission and follow-up flow
def handle_query():
    query = st.session_state['input_text'].strip().lower()

    # If user says "stop," end the conversation
    if query == "stop":
        st.session_state['chat_history'].append(("Bot", "Thank you for chatting. Take care!"))
        st.session_state['is_active'] = False
    else:
        # Add user input to chat history
        st.session_state['chat_history'].append(("You", query))

        # Store responses for basic information
        if st.session_state['collecting_info']:
            if not st.session_state.get("name"):
                st.session_state["name"] = query.capitalize()
            elif not st.session_state.get("age"):
                st.session_state["age"] = query
            elif not st.session_state.get("sex"):
                st.session_state["sex"] = query.capitalize()

        # Generate a follow-up question, diagnosis, or treatment suggestion
        response, is_diagnosis = generate_response(
            st.session_state['chat_history'],
            st.session_state['collecting_info'],
            st.session_state['question_count']
        )

        # Increment question count if in diagnostic mode
        if not st.session_state['collecting_info'] and not is_diagnosis:
            st.session_state['question_count'] += 1

        # Add response to chat history and handle stopping condition after diagnosis
        if is_diagnosis:
            st.session_state['chat_history'].append(("Bot", response))
            st.session_state['waiting_for_more_info'] = True  # Wait for user input for additional information
        else:
            st.session_state['chat_history'].append(("Bot", response))

    # Clear the input text after handling the query
    st.session_state['input_text'] = ""

# Initialize Streamlit app
st.set_page_config(page_title="Medical Assistance Chatbot", layout="wide")
st.header("Medical Assistance Chatbot")

# Sidebar for settings and actions
with st.sidebar:
    st.title("Settings")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state['chat_history'] = [("Bot", "Hi there! How can I assist you today?")]
        st.session_state['is_active'] = True
        st.session_state['question_count'] = 0
        st.session_state['waiting_for_more_info'] = False  # Reset flag for additional information
        st.session_state['collecting_info'] = True  # Reset to start collecting info
        st.session_state['name'] = None
        st.session_state['age'] = None
        st.session_state['sex'] = None

# Custom CSS for styling
st.markdown("""
    <style>
    .user-message {
        text-align: right;
        background-color: #d1ecf1;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 60%;
        float: right;
        color: black;
    }
    .bot-message {
        text-align: left;
        background-color: #fff9c8;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 80%;
        float: left;
        color: black;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        max-height: 80vh;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history and conversation flow
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [("Bot", "Hi there! How can I assist you today?")]
if 'is_active' not in st.session_state:
    st.session_state['is_active'] = True
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""
if 'question_count' not in st.session_state:
    st.session_state['question_count'] = 0  # Initialize question counter
if 'waiting_for_more_info' not in st.session_state:
    st.session_state['waiting_for_more_info'] = False  # Flag to check if additional information is needed
if 'collecting_info' not in st.session_state:
    st.session_state['collecting_info'] = True  # Flag for initial info collection
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'age' not in st.session_state:
    st.session_state['age'] = None
if 'sex' not in st.session_state:
    st.session_state['sex'] = None

# Function to render chat messages
def render_chat():
    for role, message in st.session_state['chat_history']:
        if role == "You":
            st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{message}</div>", unsafe_allow_html=True)

# Display chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
render_chat()
st.markdown("</div>", unsafe_allow_html=True)

# Input field for user question
if st.session_state['is_active']:
    st.text_input(
        "Your response (type 'stop' to end):",
        key="input_text",
        max_chars=500,
        on_change=handle_query
    )
