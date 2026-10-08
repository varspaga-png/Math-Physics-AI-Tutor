from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Generate embeddings for a list of documents."""
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        return embeddings

    def build_index(self, documents: list[str]):
        """Build FAISS index from documents."""
        self.documents = documents
        embeddings = self.embed_documents(documents)
        self.embeddings = embeddings

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for similar documents."""
        if self.index is None:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "distance": float(distance),
                    "index": int(idx),
                })
        return results

    def save_index(self, path: str | Path):
        """Save FAISS index and documents to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss_index.bin"))
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "documents.txt", "w") as f:
            for doc in self.documents:
                f.write(doc + "\n\n---\n\n")

    def load_index(self, path: str | Path):
        """Load FAISS index from disk."""
        path = Path(path)
        self.index = faiss.read_index(str(path / "faiss_index.bin"))
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "documents.txt", "r") as f:
            content = f.read()
            self.documents = [doc.strip() for doc in content.split("---") if doc.strip()]
