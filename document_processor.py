import os
import sys
import pandas as pd
import logging
from typing import List, Optional
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains.summarize import load_summarize_chain
    from langchain_ollama import OllamaLLM
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Required packages: langchain-community pypdf langchain-text-splitters langchain-ollama")
    raise

def process_csv_file(file_path: str) -> Optional[List[Document]]:
    """Enhanced CSV processing with multiple fallback methods"""
    try:
        # Method 1: Try pandas with flexible parsing
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip', encoding_errors='replace')
            docs = []
            for i, row in df.iterrows():
                try:
                    content = "\n".join(
                        f"{col}: {val}" 
                        for col, val in row.items() 
                        if pd.notna(val) and str(val).strip()
                    )
                    if content:
                        docs.append(Document(
                            page_content=content,
                            metadata={
                                "source": os.path.basename(file_path),
                                "row_index": i,
                                "file_type": "csv"
                            }
                        ))
                except Exception as row_error:
                    logger.warning(f"Error processing row {i} in {file_path}: {row_error}")
            
            if docs:
                logger.info(f"Processed {len(docs)} rows from {file_path} using pandas")
                return docs
            
        except Exception as pandas_error:
            logger.warning(f"Pandas CSV processing failed, trying CSVLoader: {pandas_error}")

        # Method 2: Fallback to CSVLoader
        try:
            loader = CSVLoader(file_path=file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source": os.path.basename(file_path),
                    "file_type": "csv"
                })
            logger.info(f"Processed {len(docs)} documents from {file_path} using CSVLoader")
            return docs
        except Exception as loader_error:
            logger.error(f"CSVLoader failed for {file_path}: {loader_error}")
            return None

    except Exception as e:
        logger.error(f"All CSV processing methods failed for {file_path}: {e}")
        return None

def process_uploaded_files(file_paths: List[str]) -> List[Document]:
    """Robust file processor with support for CSV, TXT, and PDF"""
    documents = []
    
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue

            logger.info(f"Processing file: {file_path}")
            
            if file_path.endswith('.csv'):
                docs = process_csv_file(file_path)
            elif file_path.endswith('.txt'):
                try:
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            "source": os.path.basename(file_path),
                            "file_type": "text"
                        })
                except Exception as e:
                    logger.error(f"Error processing text file {file_path}: {e}")
                    continue
            elif file_path.endswith('.pdf'):
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            "source": os.path.basename(file_path),
                            "file_type": "pdf",
                            "page": doc.metadata.get('page', 'unknown')
                        })
                except Exception as e:
                    logger.error(f"Error processing PDF {file_path}: {e}")
                    continue
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                continue

            if docs:
                documents.extend(docs)
                logger.info(f"Successfully processed {len(docs)} documents from {file_path}")

        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
            continue

    if documents:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
                keep_separator=True
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Document splitting failed, returning unsplit documents: {e}")
            return documents
    
    logger.warning("No valid documents were processed")
    return []

def summarize_content(documents: List[Document]) -> List[str]:
    """Generate summaries for documents in batches with robust error handling"""
    if not documents:
        logger.warning("No documents provided for summarization")
        return []

    try:
        llm = OllamaLLM(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.3
        )
        
        summary_chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            verbose=False
        )
        
        summaries = []
        batch_size = 3
        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            try:
                debug_document_batches(batch, i)  # <-- debug log
                result = summary_chain.run(batch)
                
                if not result or not result.strip():
                    logger.warning(f"Empty summary for batch {i+1}-{i+batch_size}")
                    summaries.extend(["Summary unavailable"] * len(batch))
                else:
                    logger.info(f"Summarized documents {i+1}-{min(i+batch_size, total_docs)}/{total_docs}")
                    summaries.extend([result] * len(batch))
                    
            except Exception as batch_error:
                logger.error(f"Error summarizing batch {i//batch_size + 1}: {batch_error}")
                summaries.extend(["Summary unavailable"] * len(batch))
        
        return summaries

    except Exception as e:
        logger.error(f"Summarization setup failed: {e}")
        return ["Summary generation failed"] * len(documents)

def debug_document_batches(batch: List[Document], batch_index: int):
    """Prints first few characters of documents in the current batch for debugging."""
    logger.debug(f"--- Batch {batch_index//3 + 1} Preview ---")
    for j, doc in enumerate(batch):
        preview = doc.page_content.strip()[:200].replace('\n', ' ')
        logger.debug(f"Doc {j+1} Preview: {preview}")
