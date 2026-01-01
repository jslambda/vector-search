import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")


class VectorSearchIndex:
    def __init__(self) -> None:
        # each entry: dict with keys "id", "vector" (np.ndarray), "norm" (float)
        self.docs: List[dict] = []

    def add(self, doc: dict, vector: np.ndarray) -> None:
        doc_id = str(uuid.uuid4())
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        self.docs.append({
            "id": doc_id,
            "vector": vec,
            "norm": norm,
            **{k: v for k, v in doc.items() if k not in ("id", "vector", "norm")}
        })

    def search(self, query_vec: np.ndarray, k: int = 10, caption_key='header') -> List[dict]:
        q = query_vec.astype(np.float32)
        q_norm = np.linalg.norm(q)
        scores: List[dict] = []
        # compute cosine similarity for each doc
        for doc in self.docs:
            dot = float(np.dot(q, doc["vector"]))
            sim = dot / (doc["norm"] *
                         q_norm) if doc["norm"] and q_norm else 0.0
            scores.append(
                {"id": doc["id"], 'caption': doc[caption_key], "score": sim})
        return sorted(scores, key=lambda x: x["score"], reverse=True)[:k]

    def load_from_list(self, docs: List[dict]) -> None:
        for d in docs:
            self.docs.append({
                'id': d['id'],
                'vector': np.array(d['vector'], dtype=np.float32),
                'norm': np.float64(d['norm']),
                **{k: v for k, v in d.items() if k not in ("id", "vector", "norm")}
            })

    def to_list(self) -> list[dict]:
        """
        Convert the index into a JSON-serializable dictionary.
        """
        return [
            {
                "id": doc["id"],
                "vector": doc["vector"].tolist(),
                "norm": float(doc["norm"]),
                **{k: v for k, v in doc.items() if k not in ("id", "vector", "norm")}
            }
            for doc in self.docs
        ]

    def to_json(self, path: Optional[Path] = None) -> str:
        """
        Serialize the index to a JSON string. If 'path' is provided, write it to that file.
        Returns the JSON string.
        """
        data = self.to_list()
        json_str = json.dumps(data)
        if path:
            path.write_text(json_str, encoding='utf-8')
        return json_str


def embed_batch(texts: List[str], model_name: None | str = None) -> List[np.ndarray]:
    """
    Encode a batch of strings into normalized float32 vectors.
    """
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [emb.astype(np.float32) for emb in embeddings]


def vectorize_docs(
        docs,
        batch_size: int,
        output: Optional[Path],
        text_extractor:Callable[[Dict[str, Any]], str] = lambda doc: doc["text_block"],
        verbose=False) -> VectorSearchIndex:
    # batch-index the documents
    index = VectorSearchIndex()
    total = len(docs)
    counter = 0

    for i in range(0, total, batch_size):
        chunk = docs[i: i + batch_size]
        texts = [text_extractor(d) for d in chunk]
        embeddings = embed_batch(texts)

        for doc, emb in zip(chunk, embeddings):
            index.add(doc, emb)
            counter += 1
            if verbose:
                print(f"Indexed: {doc['header']}")

        if verbose:
            print(f"⏳ Progress: {counter}/{total} docs indexed so far…")

    print(f"✅ Done. Total indexed: {len(index.docs)}")

    # optionally serialize index
    if output:
        index.to_json(output)
        print(f"Serialized index written to {output}")

    return index


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Vectorize a JSON file of docs, or load an existing JSON file "
            "that already includes vectorized data."
        )
    )
    p.add_argument(
        "data_path",
        type=Path,
        help=(
            "Path to a JSON file containing either raw docs (array of objects with "
            "'header' & 'text_block') or a previously vectorized index."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of docs to embed per batch (default: 32).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to write the serialized index as JSON.",
    )
    p.add_argument(
        "--query",
        "-q",
        type=str,
        default=None
    )

    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output."
    )

    args = p.parse_args()

    # load your JSON docs
    if not args.data_path.is_file():
        p.error(f"No such file: {args.data_path}")
    docs = json.loads(args.data_path.read_text(encoding="utf-8"))

    # The entries in the (initial) input file should not have 'vector' attribute.
    # Is there a better heuristic?
    if docs[0].get('vector'):
        index = VectorSearchIndex()
        index.load_from_list(docs)
    else:
        text_extractor = lambda doc: doc['text_block']
        text_blocks_text_extractor = lambda doc: ' '.join(doc.get('text_blocks', []))
        index = vectorize_docs(
            docs=docs,
            batch_size=args.batch_size,
            output=args.output,
            text_extractor=text_extractor if docs[0].get('text_block') else text_blocks_text_extractor,
            verbose=args.verbose
        )

    if args.query:
        search_term = args.query
        q_vec_arr = embed_batch([search_term])
        results = index.search(q_vec_arr[0], k=10)
        print(f"Top-10 results for “{search_term}”:")
        for rank, entry in enumerate(results, start=1):
            print(
                f"{rank:2d}. {entry['caption']} {entry['id']} (score={entry['score']:.4f})")


if __name__ == "__main__":
    main()
