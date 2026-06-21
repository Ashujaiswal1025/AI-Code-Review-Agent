import os
import shutil
import logging
import re

os.environ["ANONYMIZED_TELEMETRY"] = "false"

from git import Repo

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ Replaced: OllamaEmbeddings → GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def sanitize_collection_name(repo_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", repo_name)


def clone_repository(repo_url: str, dest_dir: str) -> str:
    """Clone a GitHub repository locally."""

    if os.path.exists(dest_dir):
        logger.info(
            f"Directory {dest_dir} already exists. Removing it."
        )
        shutil.rmtree(dest_dir)

    logger.info(f"Cloning {repo_url} into {dest_dir}...")
    Repo.clone_from(repo_url, dest_dir)

    return dest_dir


def load_and_split_codebase(repo_path: str):
    """Load repository files and split them into chunks."""

    logger.info("Loading documents...")

    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".go",
            ".rs",
            ".md",
        ],
        exclude=[
            "**/node_modules/**",
            "**/venv/**",
            "**/.git/**",
            "**/dist/**",
            "**/build/**",
        ],
        parser=LanguageParser(),
    )

    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")

    logger.info("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    texts = text_splitter.split_documents(docs)

    logger.info(f"Split into {len(texts)} chunks.")

    return texts


def ingest_repository(repo_url: str):
    """
    Full pipeline:
    1. Clone repo
    2. Load & chunk files
    3. Generate embeddings with Gemini
    4. Store in ChromaDB
    """

    settings = get_settings()

    repo_name = repo_url.rstrip("/").split("/")[-1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_path = os.path.join(
        settings.REPO_TEMP_DIR,
        repo_name,
    )

    # Step 1: Clone Repository
    clone_repository(repo_url, repo_path)

    # Step 2: Load & Split Documents
    docs = load_and_split_codebase(repo_path)

    if not docs:
        logger.warning("No documents found to ingest.")
        return False

    # Step 3: Gemini Embeddings
    # ✅ Replaced: OllamaEmbeddings(model="nomic-embed-text")
    #    → GoogleGenerativeAIEmbeddings(model="models/embedding-004")
    #    No local Ollama required — uses your GOOGLE_API_KEY env var.
    #    embedding-004 is the latest model (embedding-001 is outdated).
    logger.info("Generating embeddings using Gemini...")

    embeddings = GoogleGenerativeAIEmbeddings(
         model="gemini-embedding-2-preview",
        google_api_key=settings.GOOGLE_API_KEY,
    )

    # Step 4: Store in ChromaDB
    logger.info("Writing embeddings to ChromaDB...")

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DB_DIR,
        collection_name=sanitize_collection_name(repo_name),
    )

    logger.info(
        f"Ingestion completed successfully for {repo_name}"
    )

    return repo_name