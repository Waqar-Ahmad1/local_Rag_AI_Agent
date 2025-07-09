import streamlit as st
import os
from document_processor import process_uploaded_files
from vector import VectorDB
from interface import QAChain
import tempfile
import sys
import shutil
import io
from fpdf import FPDF
from pathlib import Path
import base64
from gtts import gTTS
import pygame

# Fix for PyTorch + Streamlit watcher bug
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# App configuration
st.set_page_config(page_title="Local RAG AI Agent", layout="wide")

def initialize_environment():
    """Set up environment variables for SSL and telemetry"""
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    telemetry_vars = {
        'LANGCHAIN_TRACING_V2': 'false',
        'LANGCHAIN_TELEMETRY': 'false',
        'LANGSMITH_TRACING': 'false',
        'LANGCHAIN_ANALYTICS': 'false'
    }
    for k, v in telemetry_vars.items():
        os.environ[k] = v

def export_chat_history(history, format_type="txt"):
    """Export chat history to text or PDF"""
    if format_type == "txt":
        return "\n\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in history]).encode()
    elif format_type == "pdf":
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for item in history:
            pdf.multi_cell(0, 10, f"Q: {item['question']}\nA: {item['answer']}\n\n")
        pdf_output = pdf.output(dest='S').encode('latin1')
        return pdf_output

def text_to_speech(text):
    """Convert text to speech and play it"""
    tts = gTTS(text=text, lang='en')
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

def display_sidebar():
    """Sidebar with model selection and system controls"""
    with st.sidebar:
        st.header("Application Controls")
        st.session_state.model_name = st.selectbox(
            "Select Ollama Model",
            ["mxbai-embed-large", "llama3", "phi", "mistral"],
            index=0
        )

def process_files(uploaded_files):
    """Handle file processing with proper error handling"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    try:
        for file in uploaded_files:
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(path)

        progress = st.progress(0, text="Loading and processing documents...")
        documents = process_uploaded_files(file_paths)
        progress.progress(50, text="Creating vector database...")

        if not documents:
            st.error("No valid documents could be processed")
            return None

        vector_db = VectorDB(documents, model_name=st.session_state.model_name)
        progress.progress(100, text="Done!")
        return vector_db

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None

    finally:
        try:
            for path in file_paths:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            st.warning(f"Cleanup warning: {str(cleanup_err)}")

def main():
    initialize_environment()
    display_sidebar()

    st.title("Local RAG AI Agent")
    st.write("Upload documents and ask questions about their content")

    # Initialize session state variables
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # File upload section
    with st.expander("Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose files (CSV, TXT, PDF)",
            type=["csv", "txt", "pdf"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing files..."):
                vector_db = process_files(uploaded_files)
                if vector_db:
                    st.session_state.vector_db = vector_db
                    st.session_state.processed = True
                    st.success("Files processed successfully!")

    # Chat interface
    if st.session_state.processed:
        st.divider()
        st.subheader("Ask a question")
        question = st.text_input("Enter your question:")
        
        col1, col2 = st.columns(2)
        with col1:
            if question and st.button("Get Answer"):
                with st.spinner("Searching for answer..."):
                    try:
                        qa_chain = QAChain(st.session_state.vector_db)
                        answer = qa_chain.ask_question(question)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer
                        })
                        
                        st.subheader("Answer")
                        st.write(answer)

                        with st.expander("Relevant Sources"):
                            for doc in qa_chain.last_retrieved_docs:
                                st.write(doc.page_content)
                                st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                                st.divider()
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
        
        with col2:
            if st.session_state.chat_history and st.button("Read Last Answer Aloud"):
                text_to_speech(st.session_state.chat_history[-1]["answer"])

        # Chat history display
        if st.session_state.chat_history:
            st.divider()
            st.subheader("Chat History")
            for i, chat in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {chat['question']}")
                st.markdown(f"**A{i+1}:** {chat['answer']}")
                st.divider()

            # Export buttons
            st.download_button(
                "Download Chat History (TXT)",
                export_chat_history(st.session_state.chat_history, "txt"),
                file_name="chat_history.txt",
                mime="text/plain"
            )
            
            st.download_button(
                "Download Chat History (PDF)",
                export_chat_history(st.session_state.chat_history, "pdf"),
                file_name="chat_history.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()