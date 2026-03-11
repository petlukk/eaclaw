//! Raw FFI bindings to llama.cpp.
//!
//! Minimal surface: model loading, context creation, tokenization,
//! decode (prefill/generate), sampling, and state export.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_float};
use std::os::raw::c_uchar;

// Opaque types
pub enum llama_model {}
pub enum llama_context {}
pub enum llama_sampler {}

// Token type
pub type llama_token = c_int;

/// Model parameters
#[repr(C)]
pub struct llama_model_params {
    pub n_gpu_layers: c_int,
    pub use_mmap: bool,
    pub use_mlock: bool,
    // Pad remaining fields — we only set a few.
    // The actual struct has more fields; we rely on default_params().
    _padding: [u8; 128],
}

/// Context parameters
#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    _padding: [u8; 128],
}

/// Batch for decode
#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut c_float,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

extern "C" {
    // Default params
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;

    // Model
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);

    // Context
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);

    // Tokenization
    pub fn llama_tokenize(
        model: *const llama_model,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn llama_token_to_piece(
        model: *const llama_model,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    // Special tokens
    pub fn llama_token_eos(model: *const llama_model) -> llama_token;
    pub fn llama_token_bos(model: *const llama_model) -> llama_token;

    // Vocab
    pub fn llama_n_vocab(model: *const llama_model) -> i32;

    // Decode
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> c_int;

    // Batch helpers
    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);

    // Logits
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;

    // KV cache
    pub fn llama_kv_cache_clear(ctx: *mut llama_context);
    pub fn llama_kv_cache_seq_rm(ctx: *mut llama_context, seq_id: i32, p0: i32, p1: i32) -> bool;

    // State (for Approach A — eakv bridge)
    pub fn llama_state_seq_get_size(ctx: *mut llama_context, seq_id: i32) -> usize;
    pub fn llama_state_seq_get_data(
        ctx: *mut llama_context,
        dst: *mut c_uchar,
        size: usize,
        seq_id: i32,
    ) -> usize;

    // Sampler
    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, sampler: *mut llama_sampler);
    pub fn llama_sampler_free(sampler: *mut llama_sampler);
    pub fn llama_sampler_sample(sampler: *mut llama_sampler, ctx: *mut llama_context, idx: i32) -> llama_token;
    pub fn llama_sampler_reset(sampler: *mut llama_sampler);

    // Built-in samplers
    pub fn llama_sampler_init_temp(temp: c_float) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;

    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;
}

#[repr(C)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

/// Safe wrapper around llama.cpp model + context.
pub struct LlamaEngine {
    model: *mut llama_model,
    ctx: *mut llama_context,
    sampler: *mut llama_sampler,
    n_ctx: u32,
}

// Safety: LlamaEngine is used behind a Mutex in LocalLlmProvider,
// ensuring single-threaded access to the raw pointers.
unsafe impl Send for LlamaEngine {}

impl LlamaEngine {
    /// Load model and create context.
    pub fn new(model_path: &str, n_ctx: u32, n_threads: u32) -> Result<Self, String> {
        use std::ffi::CString;

        let c_path = CString::new(model_path)
            .map_err(|e| format!("invalid model path: {e}"))?;

        unsafe {
            let mut model_params = llama_model_default_params();
            model_params.n_gpu_layers = 0; // CPU only

            let model = llama_model_load_from_file(c_path.as_ptr(), model_params);
            if model.is_null() {
                return Err(format!("failed to load model: {model_path}"));
            }

            let mut ctx_params = llama_context_default_params();
            ctx_params.n_ctx = n_ctx;
            ctx_params.n_batch = 512;
            ctx_params.n_threads = n_threads;
            ctx_params.n_threads_batch = n_threads;

            let ctx = llama_init_from_model(model, ctx_params);
            if ctx.is_null() {
                llama_model_free(model);
                return Err("failed to create llama context".into());
            }

            // Set up sampler chain: top-k → top-p → temperature
            let chain_params = llama_sampler_chain_default_params();
            let sampler = llama_sampler_chain_init(chain_params);
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));

            Ok(Self { model, ctx, sampler, n_ctx })
        }
    }

    /// Tokenize text into token IDs.
    pub fn tokenize(&self, text: &str, add_special: bool) -> Vec<llama_token> {
        let c_text = std::ffi::CString::new(text).unwrap_or_default();
        let mut tokens = vec![0i32; text.len() + 16];
        let n = unsafe {
            llama_tokenize(
                self.model, c_text.as_ptr(), text.len() as i32,
                tokens.as_mut_ptr(), tokens.len() as i32,
                add_special, true,
            )
        };
        if n < 0 {
            // Buffer too small, resize
            tokens.resize((-n) as usize, 0);
            let n2 = unsafe {
                llama_tokenize(
                    self.model, c_text.as_ptr(), text.len() as i32,
                    tokens.as_mut_ptr(), tokens.len() as i32,
                    add_special, true,
                )
            };
            tokens.truncate(n2.max(0) as usize);
        } else {
            tokens.truncate(n as usize);
        }
        tokens
    }

    /// Convert a single token to text.
    pub fn token_to_str(&self, token: llama_token) -> String {
        let mut buf = vec![0u8; 64];
        let n = unsafe {
            llama_token_to_piece(
                self.model, token,
                buf.as_mut_ptr() as *mut c_char, buf.len() as i32,
                0, false,
            )
        };
        if n > 0 {
            buf.truncate(n as usize);
            String::from_utf8_lossy(&buf).into_owned()
        } else {
            String::new()
        }
    }

    /// Detokenize a slice of tokens.
    pub fn detokenize(&self, tokens: &[llama_token]) -> String {
        tokens.iter().map(|&t| self.token_to_str(t)).collect()
    }

    /// EOS token ID.
    pub fn eos_token(&self) -> llama_token {
        unsafe { llama_token_eos(self.model) }
    }

    /// Decode a batch of tokens (prefill or single-token generate).
    pub fn decode(&mut self, tokens: &[llama_token], start_pos: i32) -> Result<(), String> {
        unsafe {
            let mut batch = llama_batch_init(tokens.len() as i32, 0, 1);
            batch.n_tokens = tokens.len() as i32;

            for (i, &tok) in tokens.iter().enumerate() {
                *batch.token.add(i) = tok;
                *batch.pos.add(i) = start_pos + i as i32;
                *batch.n_seq_id.add(i) = 1;
                let seq_ids = std::slice::from_raw_parts_mut(*batch.seq_id.add(i), 1);
                seq_ids[0] = 0;
                // Only compute logits for the last token
                *batch.logits.add(i) = if i == tokens.len() - 1 { 1 } else { 0 };
            }

            let rc = llama_decode(self.ctx, batch);
            llama_batch_free(batch);

            if rc != 0 {
                return Err(format!("llama_decode failed: {rc}"));
            }
            Ok(())
        }
    }

    /// Sample next token from logits of the last decoded position.
    pub fn sample(&mut self) -> llama_token {
        unsafe {
            let token = llama_sampler_sample(self.sampler, self.ctx, -1);
            llama_sampler_reset(self.sampler);
            token
        }
    }

    /// Clear the KV cache.
    pub fn kv_cache_clear(&mut self) {
        unsafe { llama_kv_cache_clear(self.ctx) }
    }

    /// Remove KV cache entries from position p0 to end.
    pub fn kv_cache_truncate(&mut self, p0: i32) {
        unsafe { llama_kv_cache_seq_rm(self.ctx, 0, p0, -1); }
    }

    /// Export KV state for eakv bridge (Approach A).
    pub fn export_kv_state(&mut self) -> Vec<u8> {
        unsafe {
            let size = llama_state_seq_get_size(self.ctx, 0);
            let mut buf = vec![0u8; size];
            llama_state_seq_get_data(self.ctx, buf.as_mut_ptr(), size, 0);
            buf
        }
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        unsafe {
            llama_sampler_free(self.sampler);
            llama_free(self.ctx);
            llama_model_free(self.model);
        }
    }
}
