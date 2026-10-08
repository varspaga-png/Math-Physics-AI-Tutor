from pathlib import Path
from typing import Iterator
from PyPDF2 import PdfReader
import re


class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str | Path) -> list[str]:
        """Load and extract text from PDF files."""
        chunks = []
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        # Split by paragraphs
                        paragraphs = text.split("\n\n")
                        for para in paragraphs:
                            cleaned = DocumentLoader._clean_text(para)
                            if len(cleaned) > 50:  # Only keep substantial chunks
                                chunks.append(cleaned)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return chunks

    @staticmethod
    def load_text(file_path: str | Path) -> list[str]:
        """Load and chunk text files."""
        chunks = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Split by paragraphs
                paragraphs = content.split("\n\n")
                for para in paragraphs:
                    cleaned = DocumentLoader._clean_text(para)
                    if len(cleaned) > 50:
                        chunks.append(cleaned)
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
        return chunks

    @staticmethod
    def load_directory(directory: str | Path, pattern: str = "*.txt") -> list[str]:
        """Load all documents from a directory."""
        chunks = []
        directory = Path(directory)
        for file in directory.glob(pattern):
            if file.suffix == ".pdf":
                chunks.extend(DocumentLoader.load_pdf(file))
            elif file.suffix in [".txt", ".md"]:
                chunks.extend(DocumentLoader.load_text(file))
        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep mathematical notation
        text = text.strip()
        return text
