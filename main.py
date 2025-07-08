import streamlit as st
import os
from document_processor import process_uploaded_files, summarize_content
from vector import VectorDB
from interface import QAChain
import tempfile
import sys
import shutil
import io
import pandas as pd
from fpdf import FPDF
from pathlib import Path

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

def export_csv(summaries):
    df = pd.DataFrame({"Summary": summaries})
    return df.to_csv(index=False).encode()

def export_pdf(summaries):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for i, text in enumerate(summaries):
        pdf.multi_cell(0, 10, f"Document {i+1} Summary:\n{text}\n\n")
    pdf_output = pdf.output(dest='S').encode('latin1')
    buffer = io.BytesIO(pdf_output)
    return buffer.getvalue()

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
        progress.progress(40, text="Creating vector database...")

        if not documents:
            st.error("No valid documents could be processed")
            return None, None

        vector_db = VectorDB(documents, model_name=st.session_state.model_name)
        progress.progress(80, text="Generating summaries...")
        summaries = summarize_content(documents)
        progress.progress(100, text="Done!")

        return vector_db, summaries

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None, None

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

    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []

    with st.expander("Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose files (CSV, TXT, PDF)",
            type=["csv", "txt", "pdf"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing files..."):
                vector_db, summaries = process_files(uploaded_files)

                if vector_db and summaries:
                    st.session_state.vector_db = vector_db
                    st.session_state.summaries = summaries
                    st.session_state.processed = True
                    st.success("Files processed successfully!")

    if st.session_state.processed and st.session_state.summaries:
        with st.expander("Document Summaries"):
            for i, summary in enumerate(st.session_state.summaries):
                st.subheader(f"Document {i+1} Summary")
                st.write(summary)

            csv_data = export_csv(st.session_state.summaries)
            st.download_button("Download CSV", csv_data, file_name="summaries.csv", mime="text/csv")

            pdf_data = export_pdf(st.session_state.summaries)
            st.download_button("Download PDF", pdf_data, file_name="summaries.pdf", mime="application/pdf")

        st.divider()
        st.subheader("Ask a question based on summaries")
        question_summary = st.text_input("Query summaries only:")
        if question_summary and st.button("Get Answer from Summary"):
            with st.spinner("Querying summaries..."):
                from langchain_core.documents import Document
                summary_docs = [Document(page_content=s) for s in st.session_state.summaries]
                summary_vector_db = VectorDB(summary_docs, model_name=st.session_state.model_name)
                qa_chain = QAChain(summary_vector_db)
                answer = qa_chain.ask_question(question_summary)
                st.write(answer)

    if st.session_state.processed:
        st.divider()
        st.subheader("Ask a question based on full documents")
        question = st.text_input("Ask a question:")
        if question and st.button("Get Answer"):
            with st.spinner("Searching for answer..."):
                try:
                    qa_chain = QAChain(st.session_state.vector_db)
                    answer = qa_chain.ask_question(question)
                    st.subheader("Answer")
                    st.write(answer)

                    with st.expander("Relevant Sources"):
                        for doc in qa_chain.last_retrieved_docs:
                            st.write(doc.page_content)
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                            st.divider()
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
