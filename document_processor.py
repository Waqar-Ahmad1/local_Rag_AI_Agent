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
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Required packages: langchain-community pypdf langchain-text-splitters")
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