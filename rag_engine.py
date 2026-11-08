"""RAG engine for the Cred capstone."""

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
KB_FILE = ROOT / "kb" / "policies.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAMES = {
    "fixed": "policies_fixed",
    "sentence": "policies_sentence",
}


def load_policies() -> list[dict]:
    with KB_FILE.open(encoding="utf-8") as handle:
        documents = json.load(handle)

    required = {"id", "title", "text"}
    if not isinstance(documents, list):
        raise ValueError("policies.json must contain a list")

    for document in documents:
        if not required.issubset(document):
            raise ValueError("Each policy needs id, title and text fields")

    return documents


def fixed_chunks(text: str, size: int = 240, overlap: int = 45) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid fixed chunk settings")

    chunks = []
    step = size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break

    return chunks


def sentence_chunks(text: str, sentences_per_chunk: int = 2) -> list[str]:
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return [
        " ".join(sentences[index:index + sentences_per_chunk])
        for index in range(0, len(sentences), sentences_per_chunk)
    ]


class PolicySearch:
    def __init__(self, auto_index: bool = True):
        settings = Settings(
            anonymized_telemetry=False,
            chroma_api_impl="chromadb.api.segment.SegmentAPI",
        )

        self.client = chromadb.Client(settings)
        self.collections = {
            strategy: self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
            for strategy, collection_name in COLLECTION_NAMES.items()
        }

        print("Loading local embedding model...", flush=True)
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("Embedding model ready.", flush=True)

        self.cache = {}
        self.generation_calls = 0

        if auto_index:
            self.build_index()

    def _embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            device="cpu",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.astype("float32").tolist()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def build_index(self) -> None:
        documents = load_policies()
        print(f"Loaded {len(documents)} policy documents.", flush=True)

        for strategy, collection in self.collections.items():
            ids = []
            texts = []
            embedding_inputs = []
            metadatas = []

            for document in documents:
                chunks = (
                    fixed_chunks(document["text"])
                    if strategy == "fixed"
                    else sentence_chunks(document["text"])
                )

                for chunk_number, chunk in enumerate(chunks):
                    ids.append(f"{document['id']}_{chunk_number}")
                    texts.append(chunk)

                    # The title is included only in the embedding input. It improves
                    # retrieval for short questions such as "What documents are needed
                    # for KYC?" while keeping the returned policy text clean.
                    embedding_inputs.append(f"{document['title']}. {chunk}")

                    metadatas.append(
                        {
                            "doc_id": document["id"],
                            "title": document["title"],
                            "strategy": strategy,
                        }
                    )

            print(
                f"Embedding {len(texts)} {strategy} chunks one at a time...",
                flush=True,
            )
            embeddings = self._embed_texts(embedding_inputs)
            print(f"Embeddings ready for {strategy} chunks.", flush=True)

            batch_size = 8
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                collection.upsert(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=embeddings[start:end],
                )

            print(
                f"{strategy}: indexed {collection.count()} chunks "
                f"from {len(documents)} policies",
                flush=True,
            )

    def search(self, query: str, strategy: str = "sentence", k: int = 3) -> list[dict]:
        if strategy not in self.collections:
            raise ValueError(f"Unknown strategy: {strategy}")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        collection = self.collections[strategy]
        if collection.count() == 0:
            raise RuntimeError("The RAG index is empty")

        query_embedding = self._embed_text(query)

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            similarity = 1.0 - float(distance)
            hits.append(
                {
                    "text": text,
                    "doc_id": metadata["doc_id"],
                    "title": metadata["title"],
                    "similarity": round(similarity, 4),
                }
            )

        return hits

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def answer(
        self,
        query: str,
        threshold: float,
        strategy: str = "sentence",
        k: int = 3,
    ) -> dict:
        cache_key = (
            self._normalize_query(query),
            round(float(threshold), 4),
            strategy,
            k,
        )

        if cache_key in self.cache:
            cached = dict(self.cache[cache_key])
            cached["cache_hit"] = True
            return cached

        matches = self.search(query, strategy=strategy, k=k)
        self.generation_calls += 1

        if not matches or matches[0]["similarity"] < threshold:
            response = {
                "answer": "I don't know based on the available policy documents.",
                "sources": [],
                "supported": False,
                "top_similarity": matches[0]["similarity"] if matches else None,
            }
        else:
            # MOCK_LLM mode deliberately returns retrieved policy text only.
            # This makes it impossible for the generator to invent unsupported facts.
            response = {
                "answer": matches[0]["text"],
                "sources": [matches[0]["doc_id"]],
                "supported": True,
                "top_similarity": matches[0]["similarity"],
            }

        self.cache[cache_key] = dict(response)
        response["cache_hit"] = False
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Cred policy RAG demo")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--query")
    parser.add_argument(
        "--strategy",
        choices=list(COLLECTION_NAMES),
        default="sentence",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not args.index and not args.query:
        parser.print_help()
        return

    engine = PolicySearch(auto_index=True)

    if args.index:
        print("Index build completed.", flush=True)
        print(f"fixed count: {engine.collections['fixed'].count()}", flush=True)
        print(f"sentence count: {engine.collections['sentence'].count()}", flush=True)

    if args.query:
        hits = engine.search(args.query, strategy=args.strategy, k=args.top_k)
        print(json.dumps(hits, indent=2))

        if args.threshold is not None:
            response = engine.answer(
                args.query,
                threshold=args.threshold,
                strategy=args.strategy,
                k=args.top_k,
            )
            print("\nGrounded answer")
            print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
