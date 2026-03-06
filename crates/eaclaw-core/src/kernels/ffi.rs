//! FFI declarations for Eä kernels.
//!
//! These match the C ABI exports from the compiled .so files.
//! In production, `ea bind --rust` generates these automatically.

#[link(name = "byte_classifier")]
extern "C" {
    pub fn classify_bytes(text: *const u8, flags: *mut u8, len: i32);
}

#[link(name = "json_scanner")]
extern "C" {
    pub fn count_json_structural(text: *const u8, len: i32, out_count: *mut i32);

    pub fn extract_json_structural(
        text: *const u8,
        len: i32,
        out_pos: *mut i32,
        out_types: *mut u8,
        out_count: *mut i32,
    );
}

#[link(name = "leak_scanner")]
extern "C" {
    pub fn scan_leak_prefixes(
        text: *const u8,
        len: i32,
        out_masks: *mut i32,
        out_n_blocks: *mut i32,
    );
}

#[link(name = "sanitizer")]
extern "C" {
    pub fn scan_injection_prefixes(
        text: *const u8,
        len: i32,
        out_masks: *mut i32,
        out_n_blocks: *mut i32,
    );
}

#[link(name = "command_router")]
extern "C" {
    pub fn match_command(text: *const u8, len: i32, out_match: *mut i32);
}

#[link(name = "search")]
extern "C" {
    pub fn batch_dot(
        query: *const f32,
        vecs: *const f32,
        dim: i32,
        n_vecs: i32,
        out_scores: *mut f32,
    );

    pub fn batch_cosine(
        query: *const f32,
        query_norm: f32,
        vecs: *const f32,
        dim: i32,
        n_vecs: i32,
        out_scores: *mut f32,
    );

    pub fn batch_l2(
        query: *const f32,
        vecs: *const f32,
        dim: i32,
        n_vecs: i32,
        out_scores: *mut f32,
    );

    pub fn normalize_vectors(vecs: *mut f32, dim: i32, n_vecs: i32);

    pub fn threshold_filter(
        scores: *const f32,
        n: i32,
        threshold: f32,
        out_indices: *mut i32,
        out_count: *mut i32,
    );

    pub fn top_k(
        scores: *const f32,
        n: i32,
        k: i32,
        out_indices: *mut i32,
        out_scores: *mut f32,
    );
}

#[link(name = "fused_safety")]
extern "C" {
    pub fn scan_safety_fused(
        text: *const u8,
        len: i32,
        out_inject_masks: *mut i32,
        out_leak_masks: *mut i32,
        out_n_blocks: *mut i32,
    );
}
