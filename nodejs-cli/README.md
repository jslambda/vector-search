# Node.js Vector Search CLI

A command-line tool that builds a local vector index from JSON documents and runs semantic search queries using the Hugging Face `@huggingface/transformers` pipeline (JS/WASM). It batches text embedding requests to speed up indexing and reuses cached vectors when they are already present in the input file.

## Prerequisites
- Node.js 18+ (ESM is enabled via `"type": "module"`).
- Internet access on first run so the default `Xenova/all-MiniLM-L6-v2` model can be downloaded by `@huggingface/transformers`.

## Installation
From the repository root:

```bash
cd nodejs-cli
npm install
```

## Input data format
Provide a JSON file containing an array of documents. Each document must include a `header` and a `text_block`. If you already have precomputed embeddings, include them in a `vector` property to skip recomputation. Example:

```json
[
  {
    "id": "doc-1",
    "header": "Intro",
    "text_block": "This is the opening section.",
    "vector": [0.1, 0.2, 0.3]
  },
  {
    "id": "doc-2",
    "header": "Details",
    "text_block": "More detailed explanation goes here."
  }
]
```

- If `vector` is present on the first document, the CLI assumes every document already has an embedding and loads them directly via `VectorSearch.loadDocs`.
- Otherwise, the tool batches calls to the feature-extraction pipeline (default batch size: 32) and uses mean pooling with L2 normalization to create vectors.

## Usage
Run the CLI by pointing to your data file and supplying a query string:

```bash
node app.js <path/to/data.json> --query "what you want to find" [--verbose]
```

Options:
- `--query` (required): The natural-language query to embed and search for.
- `--verbose`: Logs indexing progress and each document header as it is added.

### What it does
1. Reads and parses the JSON file provided as the first positional argument; exits with an error message if the file is missing or malformed.
2. Initializes a `VectorSearch` instance from `../vector_search.js`.
3. Either loads existing vectors from the file or embeds `text_block` values in batches using the shared feature-extraction pipeline.
4. Adds each embedding to the vector index (calling `index.add` with the document header as the label).
5. Embeds the query text, performs a similarity search for the top 10 matches (`index.search(qVec, 10)`), and prints the ranked results with scores.

## Tips
- Adjust the `BATCH_SIZE` constant in `app.js` if you want to trade off memory versus throughput during embedding.
- You can change the embedding model by passing a model ID to `embedBatch`/`getExtractor` (e.g., `getExtractor("sentence-transformers/all-MiniLM-L12-v2")`).
- To disable verbose logging, omit the `--verbose` flag.
