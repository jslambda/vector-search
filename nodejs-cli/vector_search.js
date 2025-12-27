/**
 * Simple in-memory cosine-similarity index.
 */
export class VectorSearch {
  constructor() {
    this.docs = [];  // each entry: { id, vector: Float32Array, norm: number }
  }

  add(id, vector) {
    const vec = Float32Array.from(vector);
    const norm = Math.hypot(...vec);
    this.docs.push({ id, vector: vec, norm });
  }

  loadDocs(docs) {
    if (!Array.isArray(docs)) {
      throw new TypeError("VectorSearch.loadDocs: expected an array of {id,vector,norm}");
    }
    docs.forEach(({ id, vector, norm, ...rest }) => {
      const vec = Float32Array.from(vector);
      this.docs.push({ id, vector: vec, norm, ...rest });
    });
  }

  search(queryVec, k = 10) {
    const q = Float32Array.from(queryVec);
    const qNorm = Math.hypot(...q);

    return this.docs
      .map(({ id, vector, norm, ...rest }) => {
        let dot = 0;
        for (let i = 0; i < q.length; i++) {
          dot += q[i] * vector[i];
        }
        return { id, score: dot / (norm * qNorm), ...rest };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  /**
   * Finds the first doc whose [attr] matches text exactly (case-insensitive).
   * @param {string} text – The search term.
   * @param {string} [attr='header'] – The document property to compare.
   * @returns {object|null}
   */
  textSearch(text, attr = 'header') {
    if (typeof text !== 'string') {
      return null;
    }
    const loweredText = text.trim().toLowerCase();

    for (const doc of this.docs) {
      const value = doc?.[attr];
      if (typeof value === 'string' && value.trim().toLowerCase() === loweredText) {
        return doc;
      }
    }

    return null;

  }

}
