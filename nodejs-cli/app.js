import fs from 'fs';
let { pipeline } = await import('@huggingface/transformers');
import { VectorSearch } from '../vector_search.js';


// ────────────────────────────────────────
//  Embedder: load once, then batch-call it
// ────────────────────────────────────────
let _extractor = null;

/**
 * Returns the "feature-extraction" pipeline instance (singleton).
 * You can pass options like { device: "webgpu" } if supported.
 */
async function getExtractor(model = "Xenova/all-MiniLM-L6-v2") {
    if (!_extractor) {
        _extractor = await pipeline("feature-extraction", model, {
            // e.g. you could specify { device: "webgpu" } or { device: "cuda" } if Node has CUDA support
        });
    }
    return _extractor;
}

/**
 * Batch-embed an array of strings (textList). Returns an array of Float32Array.
 *
 */
async function embedBatch(textList, model) {
    const extractor = await getExtractor(model);
    // When you call extractor([...], { pooling:"mean", normalize:true }),
    // you get back an array of tensors (one per input string).
    const results = await extractor(textList, { pooling: "mean", normalize: true });

    // Each result is a Tensor; `tensor.data` is a Float32Array of shape [1, dim].
    // We want just the inner Float32Array (i.e. shape [dim]).
    // Depending on HF-JS version, results might be:
    //   • an array of { data: Float32Array, shape: [...], ... }
    //   • or directly an array of Float32Array
    return results;
}

function readCmdArgs() {
    function getFlag(name) {
        const idx = args.indexOf(name);
        if (idx === -1) return null;
        if (idx + 1 >= args.length) throw new Error(`Flag ${name} needs a value`);
        return args[idx + 1];
    }

    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.error('Usage: node app.js <path to data file> [--query "your query text"] [--verbose]');
        process.exit(1);
    }

    const dataFile = args.find(arg => !arg.startsWith('--'));
    const queryText = getFlag('--query');
    const verbose = args.find(arg => arg === '--verbose');

    if (!dataFile) {
        console.error('Error: Missing path to data file');
        process.exit(1);
    }
    if (!queryText) {
        console.error("Error: Missing query text");
        process.exit(1);
    }
    return { dataFile, queryText, verbose };
}
// ──────────────────────────────────────────
//  Main indexing logic 
// ──────────────────────────────────────────
(async () => {

    const { dataFile, queryText, verbose } = readCmdArgs();
    let raw;
    try {
        raw = fs.readFileSync(dataFile, 'utf-8');
    } catch (err) {
        console.error(`Failed to read file "${dataFile}":`, err.message);
        process.exit(1);
    }
    const docs = JSON.parse(raw);

    const index = new VectorSearch();

    if (docs[0]["vector"]) {
        index.loadDocs(docs);
    } else {
        const BATCH_SIZE = 32; // tweak this up or down based on memory/CPU!
        const total = docs.length;
        let counter = 0;

        for (let i = 0; i < total; i += BATCH_SIZE) {
            // Slice out a chunk of up to BATCH_SIZE docs
            const chunk = docs.slice(i, i + BATCH_SIZE);
            const texts = chunk.map((d) => d.text_block);

            // Get a batch of embeddings in one HF call
            const embeddings = await embedBatch(texts);
            // embeddings is now an array of Float32Array, one per doc

            // Add each embedded vector to your index
            for (let j = 0; j < chunk.length; j++) {
                index.add(chunk[j].header, embeddings[j]);
                if (verbose) { console.log(`Indexed: ${chunk[j].header}`); }
                counter++;
            }

            if (verbose) { console.log(`⏳ Progress: ${counter}/${total} docs indexed so far…`); }
        }

        console.log(`✅ Done. Total indexed: ${index.docs.length}`);

    }

    // Finally, run a sample query
    const qVecArr = await embedBatch([queryText]);
    const qVec = qVecArr[0];
    const results = index.search(qVec, 10);
    console.log(`Top-10 results for “${queryText}”:`, results.map(({ header, id, score }, index) => `${index + 1}. ${header}  ${id} (score=${score})`));
})();
