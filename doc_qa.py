import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
from docx import Document
import sentencepiece as spm  # Ensure SentencePiece is imported
import os
import json
import uuid
from streamlit_chat import message

# Initialize the FLAN-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from TXT
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Function to handle file upload and text extraction
def handle_file_upload(uploaded_files):
    context = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            context += extract_text_from_pdf(uploaded_file) + "\n"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            context += extract_text_from_docx(uploaded_file) + "\n"
        elif uploaded_file.type == "text/plain":
            context += extract_text_from_txt(uploaded_file) + "\n"
        else:
            st.error("Unsupported file type")
    return context

# Function to generate response from FLAN-T5 model
def generate_response(context, question):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=512,  # Adjust this value as needed
        min_length=100,  # Ensure a minimum length for the output
        num_beams=4,  # Use beam search for better quality
        early_stopping=True,
        repetition_penalty=2.0  # Adjust to reduce repetition
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to save chat history
def save_chat_history(session_id, session_name, chat_history):
    with open(f"chat_history_{session_id}_{session_name}.json", "w") as f:
        json.dump(chat_history, f)

# Function to load chat history
def load_chat_history(session_id, session_name):
    try:
        with open(f"chat_history_{session_id}_{session_name}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Streamlit app layout
st.set_page_config(layout="wide")
st.title("Document Question Answering with FLAN-T5")

# Initialize session state for chat history
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state["chat_history"] = []
    st.session_state["greeting"] = "Hello, How can I assist you today?"
    st.session_state["session_name"] = "Session 1"
    st.session_state["session_count"] = 1

# Load previous chat history
previous_sessions = [f for f in os.listdir(".") if f.startswith("chat_history_")]
with st.sidebar:
    st.title("Chat Sessions")
    for session_file in previous_sessions:
        session_id = session_file.split("_")[2]
        session_name = session_file.split("_")[3].split(".")[0]
        if st.button(f"{session_name}"):
            st.session_state["session_id"] = session_id
            st.session_state["chat_history"] = load_chat_history(session_id, session_name)
            st.session_state["greeting"] = None
            st.session_state["session_name"] = session_name
    if st.button("+ New Session"):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["chat_history"] = []
        st.session_state["greeting"] = "Hello, How can I assist you today?"
        st.session_state["session_count"] += 1
        st.session_state["session_name"] = f"Session {st.session_state['session_count']}"

col1, col2, col3 = st.columns([0.5, 3, 2])

with col3:
    uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        context = handle_file_upload(uploaded_files)
        st.session_state["context"] = context
        st.text_area("Document Content", value=context, height=400)

with col2:
    if st.session_state["greeting"]:
        st.session_state["chat_history"].append({"role": "assistant", "content": st.session_state["greeting"]})
        st.session_state["greeting"] = None

    for i, chat in enumerate(st.session_state["chat_history"]):
        message(chat["content"], is_user=(chat["role"] == "user"), key=f"chat_{i}")

    question = st.chat_input("Type your message here...")

    if question:
        if "context" in st.session_state:
            answer = generate_response(st.session_state["context"], question)
            st.session_state["chat_history"].append({"role": "user", "content": question})
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            save_chat_history(st.session_state["session_id"], st.session_state["session_name"], st.session_state["chat_history"])
        else:
            st.error("Please upload a document and ask a question")

