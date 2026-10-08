#!/usr/bin/env python
import logging
import argparse
from pathlib import Path
from app.rag.document_loader import DocumentLoader
from app.rag.retriever import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Load documents into RAG system")
    parser.add_argument("directory", help="Directory containing documents")
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="File pattern to match (e.g., *.txt, *.pdf, *.md)",
    )
    parser.add_argument(
        "--output",
        default="./rag_index",
        help="Output directory for RAG index",
    )
    args = parser.parse_args()

    # Load documents
    logger.info(f"Loading documents from {args.directory} with pattern {args.pattern}")
    retriever = RAGRetriever()
    retriever.load_from_directory(args.directory, args.pattern)

    # Save index
    logger.info(f"Saving RAG index to {args.output}")
    retriever.embedding_manager.save_index(args.output)
    logger.info("RAG index saved successfully")


if __name__ == "__main__":
    main()
