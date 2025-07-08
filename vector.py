from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import logging
import shutil
import time
import tempfile
import gc
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def shutdown_chroma(db):
    """Gracefully shutdown Chroma to release file locks."""
    try:
        if hasattr(db, "_client") and hasattr(db._client, "reset"):
            db._client.reset()
        elif hasattr(db, "_client") and hasattr(db._client, "close"):
            db._client.close()
        del db
        gc.collect()
        logger.info("Chroma vector store shutdown completed.")
    except Exception as e:
        logger.warning(f"Warning: Failed to shutdown Chroma properly: {e}")

class VectorDB:
    def __init__(self, documents: List[Document], model_name: str = "mxbai-embed-large"):
        """
        Initialize vector database with model switching and retry logic
        """
        if not documents:
            logger.error("No documents provided for vector storage")
            raise ValueError("Cannot initialize with empty documents list")

        try:
            self._validate_documents(documents)
            logger.info(f"Initializing Ollama embeddings with model: {model_name}")
            self.embeddings = OllamaEmbeddings(
                model=model_name,
                base_url="http://localhost:11434"
            )

            self.persist_dir = tempfile.mkdtemp(prefix="chroma_")
            logger.info(f"Using temporary Chroma directory: {self.persist_dir}")

            self.db = self._initialize_chroma_with_retry(documents)
            logger.info(f"Successfully stored {len(documents)} documents in vector store")

        except Exception as e:
            logger.error("Vector database initialization failed", exc_info=True)
            raise RuntimeError(f"Failed to initialize vector database: {str(e)}") from e

    def _initialize_chroma_with_retry(self, documents: List[Document], max_retries: int = 3) -> Chroma:
        """Initialize Chroma with retry logic for file locks"""
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                self._safe_cleanup()
                return Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            except (PermissionError, OSError) as e:
                last_exception = e
                wait_time = attempt * 2
                logger.warning(
                    f"File access conflict (attempt {attempt}/{max_retries}). Retrying in {wait_time} seconds... Error: {str(e)}"
                )
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to initialize after {max_retries} attempts. Last error: {str(last_exception)}"
        )

    def _safe_cleanup(self):
        """Safely remove existing database if needed"""
        if not os.path.exists(self.persist_dir):
            return

        logger.info("Cleaning up existing database...")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                shutil.rmtree(self.persist_dir)
                logger.info("Successfully cleaned up old database")
                return
            except PermissionError as e:
                if attempt == max_attempts:
                    logger.error(f"Failed to clean up database after {max_attempts} attempts")
                    raise
                logger.warning(
                    f"File lock detected (attempt {attempt}/{max_attempts}). Retrying after 1 second..."
                )
                time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error during cleanup: {str(e)}")
                raise

    def _validate_documents(self, documents: List[Document]):
        """Validate documents before processing"""
        invalid_docs = [
            i for i, doc in enumerate(documents)
            if not doc.page_content or not doc.page_content.strip()
        ]

        if invalid_docs:
            error_msg = f"Found {len(invalid_docs)} empty/invalid documents (indices: {invalid_docs})"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_retriever(self, k: int = 5):
        """
        Get a configured retriever with robust defaults
        """
        if not hasattr(self, 'db'):
            raise RuntimeError("Vector database not initialized")

        try:
            k = max(1, min(k, 20))
            logger.info(f"Creating retriever with k={k}")

            return self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )

        except Exception as e:
            logger.error("Failed to create retriever", exc_info=True)
            raise RuntimeError(f"Retriever creation failed: {str(e)}") from e

    def __del__(self):
        """Clean up resources safely on object deletion"""
        try:
            if hasattr(self, 'db'):
                logger.info("Releasing vector database resources...")
                shutdown_chroma(self.db)

            if hasattr(self, 'persist_dir') and os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir, ignore_errors=True)
                logger.info("Temporary Chroma directory cleaned up")

        except Exception as e:
            logger.warning(f"Cleanup warning: {str(e)}")
