import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for image extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\langchain endpoints\langchain-439407-c27f03a83218.json"
GEMINI_API_KEY = "AIzaSyCjgg1kdbGp1ze7UKRwb6ig6aaUzQForiQ"
genai.configure(api_key=GEMINI_API_KEY)

# Function to extract text and images from PDF files
def get_pdf_text_and_images(pdf_docs):
    text = ""
    text_image_mapping = []

    os.makedirs("extracted_images", exist_ok=True)

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        page_texts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                page_texts.append(page_text)

        with fitz.open(pdf) as doc:
            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
                image_list = page.get_images(full=True)

                for image_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_extension = base_image["ext"]
                    image_name = f"pdf_page_{page_number + 1}_image_{image_index + 1}.{image_extension}"
                    image_path = os.path.join("extracted_images", image_name)
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    # Associate images with corresponding page text for better matching
                    if page_number < len(page_texts):
                        text_image_mapping.append({
                            "text": page_texts[page_number][:500],  # Shorten text for matching
                            "image_path": image_path
                        })
    
    return text, text_image_mapping

# Function to extract text and images from valid URLs
def get_text_and_images_from_urls(urls):
    all_text = ""
    url_text_image_mapping = []

    for url in urls:
        if len(url) > 2048:
            st.warning(f"Skipping long URL: {url}")
            continue

        try:
            result = urlparse(url)
            if all([result.scheme, result.netloc]):
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")

                # Extract text from <p> tags
                paragraphs = soup.find_all("p")
                page_text = "\n".join([para.get_text() for para in paragraphs])
                all_text += page_text

                # Extract images with valid src attributes
                images = soup.find_all("img")
                image_paths = []
                for img in images:
                    src = img.get("src")
                    if src:
                        if not src.startswith("http"):
                            src = f"{result.scheme}://{result.netloc}{src}"
                        try:
                            img_response = requests.get(src)
                            image = Image.open(BytesIO(img_response.content))
                            image_path = os.path.join("extracted_images", os.path.basename(src))
                            image.save(image_path)
                            image_paths.append(image_path)
                        except Exception as e:
                            st.warning(f"Unable to retrieve image from {src}: {e}")

                if page_text:
                    url_text_image_mapping.append({
                        "text": page_text,
                        "image_paths": image_paths
                    })
            else:
                st.warning(f"Invalid URL skipped: {url}")
        except Exception as e:
            st.warning(f"Could not retrieve data from {url}: {e}")
    return all_text, url_text_image_mapping

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text available to create embeddings.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, text_image_mapping):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

    # Calculate embeddings for question and PDF text segments to find best match
    question_embedding = embeddings.embed_query(user_question)
    best_match_image_paths = []

    # Calculate the similarity between the question and each text-image mapping
    max_similarity = -1
    for item in text_image_mapping:
        text_embedding = embeddings.embed_query(item["text"])
        similarity = cosine_similarity([question_embedding], [text_embedding])[0][0]

        # Update best match if higher similarity is found
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_image_paths = [item["image_path"]]

    # Display the best matching images
    for img_path in best_match_image_paths:
        st.image(img_path)

def main():
    st.set_page_config(page_title="Chat PDF and News")
    st.header("Chat with PDFs and News Articles")

    user_question = st.text_input("Ask a Question based on PDF files or News Articles")

    if 'text_image_mapping' not in st.session_state:
        st.session_state['text_image_mapping'] = None

    with st.sidebar:
        st.title("Upload PDFs or Enter URLs")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])
        urls = st.text_area("Enter URLs of news articles (one per line)")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text, text_image_mapping = "", []

                # Process PDFs if uploaded
                if pdf_docs:
                    pdf_text, pdf_text_image_mapping = get_pdf_text_and_images(pdf_docs)
                    raw_text += pdf_text
                    text_image_mapping.extend(pdf_text_image_mapping)

                # Process URLs if provided
                if urls:
                    url_list = urls.splitlines()
                    url_text, url_text_image_mapping = get_text_and_images_from_urls(url_list)
                    raw_text += url_text
                    text_image_mapping.extend(url_text_image_mapping)

                # Create text chunks and vector store if there is any text to process
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    get_vector_store(text_chunks)
                    st.session_state['text_image_mapping'] = text_image_mapping
                    st.success("Done")
                else:
                    st.error("No valid text found in PDFs or URLs.")

    if user_question and st.session_state['text_image_mapping'] is not None:
        user_input(user_question, st.session_state['text_image_mapping'])
    elif user_question:
        st.warning("Please upload and process the PDFs or enter URLs before asking a question.")

if __name__ == "__main__":
    main()
