//! SIMD-accelerated conversation recall using byte-histogram embeddings.
//!
//! Each text is embedded as a 256-dim vector (one dimension per byte value),
//! normalized via the search kernel. Recall uses SIMD cosine similarity + top-k.

use crate::kernels::search;

/// Embedding dimension — one per byte value.
const DIM: usize = 256;

/// A recalled conversation entry.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub index: usize,
    pub score: f32,
    pub text: String,
}

/// Vector store for conversation recall.
/// Stores normalized byte-histogram embeddings in a flat f32 buffer
/// for SIMD-friendly access patterns.
pub struct VectorStore {
    /// Flat embedding buffer: vecs[i*DIM .. (i+1)*DIM] = embedding for entry i
    vecs: Vec<f32>,
    /// Original text for each entry
    texts: Vec<String>,
    /// Scratch buffer for query embedding
    query_buf: Vec<f32>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            vecs: Vec::new(),
            texts: Vec::new(),
            query_buf: vec![0.0f32; DIM],
        }
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            vecs: Vec::with_capacity(max_entries * DIM),
            texts: Vec::with_capacity(max_entries),
            query_buf: vec![0.0f32; DIM],
        }
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Index a text entry. Computes byte-histogram embedding and normalizes it.
    pub fn insert(&mut self, text: &str) {
        embed_bytes(text.as_bytes(), &mut self.query_buf);
        // Normalize in-place via SIMD
        search::normalize_vectors(&mut self.query_buf, DIM, 1);
        self.vecs.extend_from_slice(&self.query_buf);
        self.texts.push(text.to_string());
    }

    /// Find the top-k most similar entries to the query.
    /// Returns results sorted by descending similarity score.
    pub fn recall(&mut self, query: &str, k: usize) -> Vec<RecallResult> {
        if self.texts.is_empty() || k == 0 {
            return Vec::new();
        }

        // Embed query
        embed_bytes(query.as_bytes(), &mut self.query_buf);
        search::normalize_vectors(&mut self.query_buf, DIM, 1);

        let query_norm = l2_norm(&self.query_buf);
        if query_norm < 1e-9 {
            return Vec::new();
        }

        // SIMD cosine similarity against all stored vectors
        let n = self.texts.len();
        let scores = search::batch_cosine(&self.query_buf, query_norm, &self.vecs, DIM, n);

        // SIMD top-k
        let (indices, top_scores) = search::top_k(&scores, k);

        let mut results: Vec<RecallResult> = indices
            .into_iter()
            .zip(top_scores)
            .filter(|(_, score)| *score > 0.01) // filter noise
            .map(|(idx, score)| RecallResult {
                index: idx as usize,
                score,
                text: self.texts[idx as usize].clone(),
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Search and format results as a display string.
    pub fn recall_formatted(&mut self, query: &str, k: usize) -> String {
        if query.is_empty() {
            return format!("usage: /recall <query> ({} entries indexed)", self.len());
        }
        let results = self.recall(query, k);
        if results.is_empty() {
            return "No matching entries found.".to_string();
        }
        let mut out = format!("Recall ({} results):\n", results.len());
        for (i, r) in results.iter().enumerate() {
            let preview: String = r.text.chars().take(120).collect();
            out.push_str(&format!(
                "  {}. [{:.2}] {}{}\n",
                i + 1,
                r.score,
                preview,
                if r.text.len() > 120 { "..." } else { "" }
            ));
        }
        out
    }

    /// Clear all stored entries.
    pub fn clear(&mut self) {
        self.vecs.clear();
        self.texts.clear();
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute byte-histogram embedding: count frequency of each byte value.
fn embed_bytes(input: &[u8], out: &mut [f32]) {
    assert!(out.len() >= DIM);
    // Zero the buffer
    for v in out.iter_mut().take(DIM) {
        *v = 0.0;
    }
    // Count byte frequencies
    for &b in input {
        out[b as usize] += 1.0;
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_bytes_basic() {
        let mut buf = vec![0.0f32; DIM];
        embed_bytes(b"aab", &mut buf);
        assert_eq!(buf[b'a' as usize], 2.0);
        assert_eq!(buf[b'b' as usize], 1.0);
        assert_eq!(buf[b'c' as usize], 0.0);
    }

    #[test]
    fn test_insert_and_recall() {
        let mut store = VectorStore::new();
        store.insert("hello world");
        store.insert("cargo build test");
        store.insert("hello there friend");

        let results = store.recall("hello", 2);
        assert!(!results.is_empty());
        // "hello world" and "hello there friend" should rank highest
        assert!(results[0].text.contains("hello"), "top result: {}", results[0].text);
    }

    #[test]
    fn test_recall_empty() {
        let mut store = VectorStore::new();
        let results = store.recall("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_recall_exact_match() {
        let mut store = VectorStore::new();
        store.insert("the quick brown fox");
        store.insert("jumps over the lazy dog");
        store.insert("the quick brown fox jumps");

        let results = store.recall("the quick brown fox", 1);
        assert_eq!(results.len(), 1);
        // Should match the exact or near-exact entry
        assert!(results[0].score > 0.9, "score: {}", results[0].score);
    }

    #[test]
    fn test_recall_different_content() {
        let mut store = VectorStore::new();
        store.insert("rust programming language");
        store.insert("12345 numbers only");
        store.insert("rust cargo build system");

        let results = store.recall("rust compiler", 2);
        // Both "rust" entries should rank above the numbers entry
        for r in &results {
            assert!(
                r.text.contains("rust"),
                "expected rust entries, got: {}",
                r.text
            );
        }
    }

    #[test]
    fn test_clear() {
        let mut store = VectorStore::new();
        store.insert("test");
        assert_eq!(store.len(), 1);
        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.recall("test", 1).is_empty());
    }

    #[test]
    fn test_many_entries() {
        let mut store = VectorStore::with_capacity(100);
        for i in 0..100 {
            store.insert(&format!("entry number {i} with some text"));
        }
        assert_eq!(store.len(), 100);
        let results = store.recall("entry number 42", 3);
        assert!(!results.is_empty());
    }
}
