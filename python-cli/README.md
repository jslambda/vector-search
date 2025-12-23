# Python CLI Vector Search

This directory contains a small Python command-line tool that builds and queries a local vector search index from a JSON file of documents. It embeds text using the [`sentence-transformers`](https://www.sbert.net/) model `all-MiniLM-L6-v2`, stores vectors in memory, and can optionally serialize the index as JSON.

## Features
- Converts JSON documents into normalized embeddings in batches.
- Computes cosine similarity search over the in-memory index.
- Supports loading a previously serialized index (with stored vectors and norms).
- Optional query flag to immediately search the index after creation.

## Installation
1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

> **Note:** Downloading the embedding model occurs on the first run of the tool.

## Input format
The CLI expects a JSON file containing an array of objects. Each object should include either:
- `header` (a short caption/title) and `text_block` (string), **or**
- `header` and `text_blocks` (list of strings) if you want multiple blocks concatenated.

If your JSON already contains `vector` and `norm` fields for each entry, the CLI will load them instead of re-embedding.

## Usage
Run the CLI from this directory:

```bash
python app.py <path-to-docs.json> [--batch-size 32] [--output index.json] [--query "search text"] [--verbose]
```

- `data_path` (positional): Path to the JSON file described above.
- `--batch-size`: Number of documents to embed per batch (default: 32).
- `--output` / `-o`: Optional path to write the serialized index as JSON.
- `--query` / `-q`: Optional search string to run immediately after indexing. Prints the top 10 results with scores.
- `--verbose` / `-v`: Print progress as documents are embedded.

### Example
Embed documents and save the index:

```bash
python app.py data/docs.json --batch-size 64 --output index.json --verbose
```

Search an existing serialized index without re-embedding:

```bash
python app.py data/index.json --query "vector databases"  # loads vectors instead of recomputing
```

## How it works
- `VectorSearchIndex` keeps a list of entries containing an `id` (UUID), vector, norm, and metadata fields from the source document.
- `vectorize_docs` embeds documents in batches using `SentenceTransformer.encode`, normalizes them, and adds them to the index. If `--output` is provided, it writes the serialized index to disk.
- When `--query` is supplied, the CLI embeds the query string, calculates cosine similarity against the stored vectors, and prints the top 10 matches along with their scores and headers.

## Notes
- All embeddings are stored in memory; very large datasets may not fit comfortably.
- The embedding model download requires network access on the first run.
