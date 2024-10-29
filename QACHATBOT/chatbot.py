import streamlit as st
import os
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image

# Set up environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/langchain endpoints/langchain-439407-c27f03a83218.json"
GEMINI_API_KEY = "AIzaSyCjgg1kdbGp1ze7UKRwb6ig6aaUzQForiQ"
genai.configure(api_key=GEMINI_API_KEY)

# Load the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")

# Function to check if the user query is requesting an image
def is_image_request(query):
    image_keywords = ["image of", "picture of", "show me", "photo of", "images of"]
    return any(keyword in query.lower() for keyword in image_keywords)

# Efficiently scrape images based on a refined query
def scrape_images(query, num_images=2):
    search_url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(search_url, headers=headers, verify=False)  # Disable SSL verification
        image_urls = []

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            img_tags = soup.find_all("img", {"class": "mimg"}, limit=num_images)
            for img in img_tags:
                img_url = img.get("src")
                if img_url:
                    image_urls.append(img_url)

        return image_urls
    except requests.exceptions.SSLError:
        st.error("An SSL error occurred while trying to scrape images. Please check your network settings.")
        return []

def get_gemini_response(question, chat_history):
    latest_history = [{"role": "user", "parts": [{"text": chat_history[-1][1]}]}] if chat_history else []
    
    chat = model.start_chat(history=latest_history)
    response = chat.send_message(question, stream=True)
    
    bot_response = ""
    finish_reason = getattr(response, "finish_reason", None)

    if finish_reason == 4:
        bot_response = "I'm sorry, but I can't provide an answer to that question due to content restrictions."
        follow_up_suggestions = []
    else:
        try:
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    bot_response += chunk.text
            follow_up_suggestions = generate_dynamic_follow_ups(bot_response)
        
        except ValueError:
            bot_response = "There was an issue retrieving a response. Please try rephrasing your question."
            follow_up_suggestions = []

    return bot_response, follow_up_suggestions

def generate_dynamic_follow_ups(response_text):
    prompt = f"Based on the response: '{response_text}', suggest three brief follow-up questions that are meaningful and complete, each ideally under 10 words."
    follow_up_chat = model.start_chat(history=[{"role": "user", "parts": [{"text": prompt}]}])
    follow_up_response = follow_up_chat.send_message(prompt)

    follow_up_suggestions = []
    for chunk in follow_up_response:
        if hasattr(chunk, 'text') and chunk.text:
            follow_up_suggestions.extend(chunk.text.strip().splitlines())

    return follow_up_suggestions[:3]

# Initialize Streamlit app
st.set_page_config(page_title="Interactive Q&A Demo", layout="wide")

st.header("Gemini LLM Interactive Chat")

# Sidebar for settings and actions
with st.sidebar:
    st.title("Settings")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state['chat_history'] = [("Bot", "Hi there! How can I assist you today?", [])]
        st.session_state['suggestions'] = []

    # Download chat button
    if st.button("Download Chat"):
        conversation = "\n".join([f"{role}: {message}" for role, message, _ in st.session_state['chat_history']])
        st.download_button("Download Chat as Text", data=conversation, file_name="chat_history.txt", mime="text/plain")

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
    .suggestion-container {
        text-align: left;
        margin-top: 10px;
        max-height: 200px;
        overflow-y: auto;
        padding: 5px;
    }
    .suggestion-button {
        font-size: 0.85rem;
        padding: 5px 10px;
        margin: 3px 0;
        width: 90%;
        color: #333;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history, suggestions, and feedback
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [("Bot", "Hi there! How can I assist you today?", [])]
if 'suggestions' not in st.session_state:
    st.session_state['suggestions'] = []
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""
if 'include_images' not in st.session_state:
    st.session_state['include_images'] = False  # Initialize checkbox state for each question

# Function to render chat messages with images below responses
def render_chat():
    for role, message, images in st.session_state['chat_history']:
        if role == "You":
            st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
        else:
            # Display bot message and images based on the inclusion setting
            if images:
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.markdown(f"<div class='bot-message'>{message}</div>", unsafe_allow_html=True)
                with col2:
                    for img_url in images:
                        response_image = requests.get(img_url, verify=False)
                        img = Image.open(BytesIO(response_image.content))
                        st.image(img, use_column_width=True, width=350)  # Larger image size for visibility
            else:
                st.markdown(f"<div class='bot-message'>{message}</div>", unsafe_allow_html=True)

# Function to handle suggestion submission
def submit_suggestion(suggestion):
    st.session_state['suggestions'] = []
    st.session_state['chat_history'].append(("You", suggestion, []))

    # Include images based on the checkbox state
    if st.session_state['include_images'] and is_image_request(suggestion):
        image_urls = scrape_images(suggestion)
        st.session_state['chat_history'].append(("Bot", "", image_urls))  # Only images, no text response
    else:
        response, new_suggestions = get_gemini_response(suggestion, st.session_state['chat_history'])
        image_urls = scrape_images(suggestion) if st.session_state['include_images'] else []
        st.session_state['chat_history'].append(("Bot", response, image_urls))
        st.session_state['suggestions'] = new_suggestions

    # Reset the checkbox state after processing the response
    st.session_state['include_images'] = False

# Render chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
render_chat()
st.markdown("</div>", unsafe_allow_html=True)

# Updated suggestion display section with compact styling and truncation
if st.session_state['suggestions']:
    with st.expander("Suggested Follow-ups"):
        st.markdown("<div class='suggestion-container'>", unsafe_allow_html=True)
        for i, suggestion in enumerate(st.session_state['suggestions']):
            if st.button(suggestion, key=f"{suggestion}_{i}", help="Click to submit this follow-up question", 
                         on_click=lambda s=suggestion: submit_suggestion(s), use_container_width=True):
                pass
        st.markdown("</div>", unsafe_allow_html=True)

# Callback function to handle text input submission
def submit_question():
    question = st.session_state['input_text']
    if question:
        st.session_state['suggestions'] = []
        st.session_state['chat_history'].append(("You", question, []))

        # Include images based on the checkbox state
        if st.session_state['include_images'] and is_image_request(question):
            image_urls = scrape_images(question)
            st.session_state['chat_history'].append(("Bot", "", image_urls))  # Only images, no text response
        else:
            response, suggestions = get_gemini_response(question, st.session_state['chat_history'])
            image_urls = scrape_images(question) if st.session_state['include_images'] else []
            st.session_state['chat_history'].append(("Bot", response, image_urls))
            st.session_state['suggestions'] = suggestions

        # Reset the input and checkbox state after processing the response
        st.session_state['input_text'] = ""
        st.session_state['include_images'] = False

# Input field at the bottom of the page with a checkbox
col1, col2 = st.columns([4, 1])
with col1:
    st.text_input(
        "Type your question here (max 500 chars):",
        key="input_text",
        max_chars=500,
        on_change=submit_question,
    )
with col2:
    st.session_state['include_images'] = st.checkbox("Include images", value=False)
