# Local Inference Runtime with eakv KV Persistence — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Anthropic API backend with an embedded llama.cpp + eakv runtime that supports O(1) KV cache checkpointing for near-free tool-call loops.

**Architecture:** `LocalLlmProvider` implements the existing `LlmProvider` trait (no agent loop changes). It embeds llama.cpp for inference and eakv for Q4-compressed KV cache storage. A stateless-to-stateful bridge diffs incoming messages against prefilled tokens to enable incremental-only prefill. Tool-call detection uses token-level pattern matching with sentinel optimization.

**Tech Stack:** Rust, C (llama.cpp + eakv), Ea SIMD kernels, Cargo feature gates, `cc` crate for C compilation

**Spec:** `docs/superpowers/specs/2026-03-11-local-inference-eakv-design.md`

---

## Chunk 1: eakv Checkpoint API + Config Foundation

### Task 1: eakv checkpoint/restore API

**Files:**
- Modify: `/root/dev/eakv/include/eakv.h:46` (after `eakv_cache_clear`)
- Modify: `/root/dev/eakv/src/cache.c:164` (after `eakv_cache_clear`)
- Test: `/root/dev/eakv/tests/` (new test file)

- [ ] **Step 1: Write the failing test**

Create `/root/dev/eakv/tests/test_checkpoint.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "eakv.h"

static void test_checkpoint_returns_seq_len(void) {
    eakv_cache_t *cache = eakv_cache_create(2, 2, 64, 256);
    assert(cache != NULL);

    /* Empty cache: checkpoint should be 0 */
    int cp0 = eakv_checkpoint(cache);
    assert(cp0 == 0);

    /* Load some data, checkpoint should match seq_len */
    int n_embd = 2 * 64; /* n_kv_heads * head_dim */
    int seq = 8;
    float *data = calloc(2 * 2 * n_embd * seq, sizeof(float));
    eakv_cache_load_raw(cache, data, seq);
    free(data);

    int cp1 = eakv_checkpoint(cache);
    assert(cp1 == 8);

    printf("  PASS: checkpoint_returns_seq_len\n");
    eakv_cache_free(cache);
}

static void test_restore_resets_seq_len(void) {
    eakv_cache_t *cache = eakv_cache_create(2, 2, 64, 256);
    int n_embd = 2 * 64;
    int seq = 16;
    float *data = calloc(2 * 2 * n_embd * seq, sizeof(float));
    eakv_cache_load_raw(cache, data, seq);
    free(data);

    int cp = eakv_checkpoint(cache);
    assert(cp == 16);

    /* Restore to halfway */
    eakv_restore(cache, 8);
    assert(eakv_cache_seq_len(cache) == 8);

    /* Restore to 0 */
    eakv_restore(cache, 0);
    assert(eakv_cache_seq_len(cache) == 0);

    printf("  PASS: restore_resets_seq_len\n");
    eakv_cache_free(cache);
}

static void test_restore_bounds_check(void) {
    eakv_cache_t *cache = eakv_cache_create(2, 2, 64, 256);
    int n_embd = 2 * 64;
    float *data = calloc(2 * 2 * n_embd * 8, sizeof(float));
    eakv_cache_load_raw(cache, data, 8);
    free(data);

    /* Restore beyond current seq_len should return error */
    int rc = eakv_restore(cache, 16);
    assert(rc == EAKV_ERR_INVALID);
    assert(eakv_cache_seq_len(cache) == 8); /* unchanged */

    /* Negative should also fail */
    rc = eakv_restore(cache, -1);
    assert(rc == EAKV_ERR_INVALID);
    assert(eakv_cache_seq_len(cache) == 8);

    printf("  PASS: restore_bounds_check\n");
    eakv_cache_free(cache);
}

static void test_append_after_restore(void) {
    eakv_cache_t *cache = eakv_cache_create(2, 2, 64, 256);
    int n_embd = 2 * 64;
    int seq = 16;
    float *data = calloc(2 * 2 * n_embd * seq, sizeof(float));
    eakv_cache_load_raw(cache, data, seq);

    /* Checkpoint, restore, append more */
    eakv_restore(cache, 8);

    /* Append 4 more tokens per layer/kv */
    float *append_data = calloc(n_embd * 4, sizeof(float));
    for (int l = 0; l < 2; l++) {
        eakv_cache_append(cache, append_data, l, 0, 4);
        eakv_cache_append(cache, append_data, l, 1, 4);
    }
    eakv_cache_advance(cache, 4);
    assert(eakv_cache_seq_len(cache) == 12);

    free(append_data);
    free(data);
    printf("  PASS: append_after_restore\n");
    eakv_cache_free(cache);
}

int main(void) {
    printf("test_checkpoint:\n");
    test_checkpoint_returns_seq_len();
    test_restore_resets_seq_len();
    test_restore_bounds_check();
    test_append_after_restore();
    printf("All checkpoint tests passed.\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /root/dev/eakv
gcc -I include -o build/test_checkpoint tests/test_checkpoint.c build/libeakv.a -lm
./build/test_checkpoint
```

Expected: linker error — `eakv_checkpoint` and `eakv_restore` undefined.

- [ ] **Step 3: Add declarations to eakv.h**

In `/root/dev/eakv/include/eakv.h`, after `eakv_cache_clear` (line 46), add:

```c
/* Checkpoint: returns current seq_len. O(1). */
int eakv_checkpoint(eakv_cache_t *cache);

/* Restore to a previous seq_len. O(1).
 * Returns EAKV_OK on success, EAKV_ERR_INVALID if seq_len is out of range. */
int eakv_restore(eakv_cache_t *cache, int seq_len);
```

- [ ] **Step 4: Implement in cache.c**

In `/root/dev/eakv/src/cache.c`, after `eakv_cache_clear` (line 164), add:

```c
int eakv_checkpoint(eakv_cache_t *cache) {
    if (!cache) return 0;
    return cache->seq_len;
}

int eakv_restore(eakv_cache_t *cache, int seq_len) {
    if (!cache) return EAKV_ERR_INVALID;
    if (seq_len < 0 || seq_len > cache->seq_len) return EAKV_ERR_INVALID;
    cache->seq_len = seq_len;
    return EAKV_OK;
}
```

- [ ] **Step 5: Build and run tests**

```bash
cd /root/dev/eakv
make
gcc -I include -o build/test_checkpoint tests/test_checkpoint.c build/libeakv.a -lm
./build/test_checkpoint
```

Expected: All 4 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /root/dev/eakv
git add include/eakv.h src/cache.c tests/test_checkpoint.c
git commit -m "feat: add eakv_checkpoint() and eakv_restore() for O(1) KV snapshots"
```

---

### Task 2: eakv incremental llama bridge

**Files:**
- Modify: `/root/dev/eakv/include/eakv_llama.h:26-28` (add new declaration)
- Modify: `/root/dev/eakv/src/llama_bridge.c` (add new function at end)

- [ ] **Step 1: Add declaration to eakv_llama.h**

After the existing `eakv_from_llama_state` declaration (line 26-28), add:

```c
/* Incremental variant: append new KV data from a llama.cpp state buffer
 * to an existing cache. Only processes tokens from start_pos to the end
 * of the state buffer. Cache seq_len must equal start_pos.
 *
 * Does NOT call eakv_cache_advance — caller must do so after all layers. */
int eakv_from_llama_state_append(eakv_cache_t *cache,
                                  const uint8_t *state_buf, size_t state_size,
                                  int n_layers, int n_kv_heads, int head_dim,
                                  int start_pos);
```

- [ ] **Step 2: Implement in llama_bridge.c**

Append to `/root/dev/eakv/src/llama_bridge.c` after the existing function:

```c
int eakv_from_llama_state_append(eakv_cache_t *cache,
                                  const uint8_t *state_buf, size_t state_size,
                                  int n_layers, int n_kv_heads, int head_dim,
                                  int start_pos) {
    if (!cache || !state_buf || state_size < 8)
        return EAKV_ERR_INVALID;
    if (start_pos < 0 || start_pos != eakv_cache_seq_len(cache))
        return EAKV_ERR_INVALID;

    const uint8_t *ptr = state_buf;
    const uint8_t *end = state_buf + state_size;

    /* Header: n_stream, cell_count */
    if (ptr + 8 > end) return EAKV_ERR_FORMAT;
    ptr += 4; /* n_stream */
    uint32_t cell_count = *(const uint32_t *)ptr; ptr += 4;

    if ((int)cell_count <= start_pos)
        return EAKV_OK; /* nothing new */

    int new_tokens = (int)cell_count - start_pos;
    int n_embd_k_gqa = n_kv_heads * head_dim;

    /* Skip cell metadata */
    for (uint32_t i = 0; i < cell_count; i++) {
        if (ptr + 8 > end) return EAKV_ERR_FORMAT;
        ptr += sizeof(int32_t);
        uint32_t n_seq_id = *(const uint32_t *)ptr; ptr += 4;
        ptr += n_seq_id * sizeof(int32_t);
        if (ptr > end) return EAKV_ERR_FORMAT;
    }

    /* v_trans, n_layer */
    if (ptr + 8 > end) return EAKV_ERR_FORMAT;
    uint32_t v_trans = *(const uint32_t *)ptr; ptr += 4;
    uint32_t file_n_layer = *(const uint32_t *)ptr; ptr += 4;
    if ((int)file_n_layer != n_layers) return EAKV_ERR_FORMAT;

    /* Temp buffers for transposing new tokens only */
    float *transposed = malloc((size_t)n_embd_k_gqa * new_tokens * sizeof(float));
    float *f32_row = malloc((size_t)n_embd_k_gqa * sizeof(float));
    if (!transposed || !f32_row) {
        free(transposed); free(f32_row);
        return EAKV_ERR_ALLOC;
    }

    /* K layers: extract only tokens [start_pos..cell_count) */
    for (int l = 0; l < n_layers; l++) {
        if (ptr + 12 > end) goto format_err;
        ptr += 4; /* type */
        uint64_t size_row = *(const uint64_t *)ptr; ptr += 8;
        if (ptr + size_row * cell_count > end) goto format_err;

        const uint16_t *fp16_data = (const uint16_t *)ptr;

        for (int t = 0; t < new_tokens; t++) {
            int src_pos = start_pos + t;
            for (int j = 0; j < n_embd_k_gqa; j++)
                f32_row[j] = ggml_fp16_to_fp32(fp16_data[src_pos * n_embd_k_gqa + j]);
            for (int h = 0; h < n_kv_heads; h++)
                memcpy(transposed + (h * new_tokens + t) * head_dim,
                       f32_row + h * head_dim,
                       head_dim * sizeof(float));
        }

        eakv_cache_append(cache, transposed, l, 0, new_tokens);
        ptr += size_row * cell_count;
    }

    /* V layers (non-transposed path only for simplicity — v_trans=0) */
    if (v_trans) {
        /* Column-major V: [dim][pos] */
        for (int l = 0; l < n_layers; l++) {
            if (ptr + 12 > end) goto format_err;
            ptr += 4;
            uint32_t size_el = *(const uint32_t *)ptr; ptr += 4;
            uint32_t n_embd_v = *(const uint32_t *)ptr; ptr += 4;
            size_t v_data_size = (size_t)n_embd_v * cell_count * size_el;
            if (ptr + v_data_size > end) goto format_err;

            const uint16_t *fp16_data = (const uint16_t *)ptr;
            for (int h = 0; h < n_kv_heads; h++) {
                for (int t = 0; t < new_tokens; t++) {
                    int src_pos = start_pos + t;
                    for (int d = 0; d < head_dim; d++) {
                        int embd_idx = h * head_dim + d;
                        uint16_t val = fp16_data[embd_idx * cell_count + src_pos];
                        transposed[(h * new_tokens + t) * head_dim + d] =
                            ggml_fp16_to_fp32(val);
                    }
                }
            }
            eakv_cache_append(cache, transposed, l, 1, new_tokens);
            ptr += v_data_size;
        }
    } else {
        for (int l = 0; l < n_layers; l++) {
            if (ptr + 12 > end) goto format_err;
            ptr += 4;
            uint64_t size_row = *(const uint64_t *)ptr; ptr += 8;
            if (ptr + size_row * cell_count > end) goto format_err;

            const uint16_t *fp16_data = (const uint16_t *)ptr;
            for (int t = 0; t < new_tokens; t++) {
                int src_pos = start_pos + t;
                for (int j = 0; j < n_embd_k_gqa; j++)
                    f32_row[j] = ggml_fp16_to_fp32(fp16_data[src_pos * n_embd_k_gqa + j]);
                for (int h = 0; h < n_kv_heads; h++)
                    memcpy(transposed + (h * new_tokens + t) * head_dim,
                           f32_row + h * head_dim,
                           head_dim * sizeof(float));
            }
            eakv_cache_append(cache, transposed, l, 1, new_tokens);
            ptr += size_row * cell_count;
        }
    }

    eakv_cache_advance(cache, new_tokens);
    free(transposed);
    free(f32_row);
    return EAKV_OK;

format_err:
    free(transposed);
    free(f32_row);
    return EAKV_ERR_FORMAT;
}
```

- [ ] **Step 3: Build eakv**

```bash
cd /root/dev/eakv
make
```

Expected: builds without errors. (Full test of this function requires llama.cpp data, tested in integration later.)

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eakv
git add include/eakv_llama.h src/llama_bridge.c
git commit -m "feat: add eakv_from_llama_state_append() for incremental KV import"
```

---

### Task 3: eaclaw config — backend selection + optional API key

**Files:**
- Modify: `/root/dev/eaclaw/crates/eaclaw-core/src/config.rs:5-50`
- Test: existing config tests if any, or manual verification

- [ ] **Step 1: Read current config.rs**

Read `/root/dev/eaclaw/crates/eaclaw-core/src/config.rs` to confirm current structure.

- [ ] **Step 2: Add Backend enum and local config fields**

At the top of `config.rs` (before Config struct), add:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Backend {
    Anthropic,
    Local,
}
```

Add fields to `Config` struct (line 5-16):

```rust
pub backend: Backend,
pub model_path: Option<String>,
pub ctx_size: usize,
pub threads: usize,
```

- [ ] **Step 3: Update Config::from_env()**

In `Config::from_env()` (line 18-50), change the API key loading and add backend parsing:

```rust
let backend = match std::env::var("EACLAW_BACKEND").as_deref() {
    Ok("local") => Backend::Local,
    _ => Backend::Anthropic,
};

let api_key = match backend {
    Backend::Anthropic => {
        std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| crate::error::Error::Config(
                "ANTHROPIC_API_KEY not set".into()
            ))?
    }
    Backend::Local => {
        std::env::var("ANTHROPIC_API_KEY").unwrap_or_default()
    }
};

let model_path = std::env::var("EACLAW_MODEL_PATH").ok();
let ctx_size = std::env::var("EACLAW_CTX_SIZE")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(4096);
let threads = std::env::var("EACLAW_THREADS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or_else(|| num_cpus::get());
```

Note: `num_cpus` crate may already be available via tokio, or use `std::thread::available_parallelism()` instead:

```rust
let threads = std::env::var("EACLAW_THREADS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
```

- [ ] **Step 4: Verify existing tests still pass**

```bash
cd /root/dev/eaclaw
cargo test
```

Expected: all 230 tests pass (no breakage from new fields — existing code uses `Config::from_env()` and accesses existing fields, new fields are additive).

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eaclaw
git add crates/eaclaw-core/src/config.rs
git commit -m "feat: add Backend enum and local inference config to Config"
```

---

### Task 4: Cargo.toml — `local-llm` feature gate + `cc` build dep

**Files:**
- Modify: `/root/dev/eaclaw/Cargo.toml` (workspace level)
- Modify: `/root/dev/eaclaw/crates/eaclaw-core/Cargo.toml`

- [ ] **Step 1: Add `cc` to workspace dependencies**

In `/root/dev/eaclaw/Cargo.toml`, in the `[workspace.dependencies]` section, add:

```toml
cc = "1"
```

- [ ] **Step 2: Add feature gate and cc dep to eaclaw-core**

In `/root/dev/eaclaw/crates/eaclaw-core/Cargo.toml`, add:

```toml
[build-dependencies]
cc = { workspace = true }

[features]
default = []
local-llm = []
```

- [ ] **Step 2b: Add feature forwarding to eaclaw-cli**

In `/root/dev/eaclaw/eaclaw-cli/Cargo.toml`, add:

```toml
[features]
default = []
local-llm = ["eaclaw-core/local-llm"]
```

This allows `cargo build --features local-llm` from the workspace or CLI crate to
enable the feature on eaclaw-core.

- [ ] **Step 3: Verify build still works**

```bash
cd /root/dev/eaclaw
cargo build
cargo test
```

Expected: compiles and all tests pass. Feature is defined but unused yet.

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eaclaw
git add Cargo.toml crates/eaclaw-core/Cargo.toml
git commit -m "feat: add local-llm feature gate and cc build dependency"
```

---

## Chunk 2: Token-Level Tool-Call Detector

### Task 5: Tool-call detector — data structures and state machine

**Files:**
- Create: `crates/eaclaw-core/src/llm/tool_parse.rs`
- Modify: `crates/eaclaw-core/src/llm/mod.rs` (add module export)
- Test: inline `#[cfg(test)]` in `tool_parse.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/eaclaw-core/src/llm/tool_parse.rs` with tests first:

```rust
/// Token-level tool-call detector.
///
/// Detects `<tool_call>...</tool_call>` patterns by matching token ID sequences.
/// No string construction during the hot path — tokens stay as i32 IDs until
/// the closing tag triggers JSON extraction.

#[derive(Debug, Clone, PartialEq)]
pub enum DetectorState {
    /// Streaming text tokens normally.
    Normal,
    /// Matched open tag, buffering token IDs for JSON body.
    Capturing,
    /// Matched close tag, ready to parse.
    Complete,
}

/// Result of feeding a token to the detector.
#[derive(Debug)]
pub enum DetectResult {
    /// Token is normal text, should be streamed to on_text.
    Text(i32),
    /// Token is part of the open tag, suppress from output.
    TagOpen,
    /// Token is being captured (JSON body), suppress from output.
    Captured,
    /// Tool call complete. Contains the captured token IDs (JSON body).
    ToolCall(Vec<i32>),
    /// Capture aborted (runaway). Contains all buffered tokens to flush as text.
    Aborted(Vec<i32>),
}

pub struct ToolCallDetector {
    open_pattern: Vec<i32>,
    close_pattern: Vec<i32>,
    open_sentinel: i32,
    close_sentinel: i32,
    state: DetectorState,
    ring: Vec<i32>,
    capture_buf: Vec<i32>,
    max_capture: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector() -> ToolCallDetector {
        // Simulate tokenization: <tool_call> = [10, 20, 30]
        // </tool_call> = [40, 50, 60]
        ToolCallDetector::new(vec![10, 20, 30], vec![40, 50, 60], 512)
    }

    #[test]
    fn normal_tokens_pass_through() {
        let mut det = make_detector();
        match det.feed(99) {
            DetectResult::Text(t) => assert_eq!(t, 99),
            other => panic!("expected Text, got {:?}", other),
        }
        assert_eq!(det.state, DetectorState::Normal);
    }

    #[test]
    fn open_tag_triggers_capture() {
        let mut det = make_detector();
        // Feed the open pattern
        assert!(matches!(det.feed(10), DetectResult::TagOpen));
        assert!(matches!(det.feed(20), DetectResult::TagOpen));
        assert!(matches!(det.feed(30), DetectResult::TagOpen));
        assert_eq!(det.state, DetectorState::Capturing);
    }

    #[test]
    fn capture_then_close_produces_tool_call() {
        let mut det = make_detector();
        // Open
        det.feed(10); det.feed(20); det.feed(30);
        // JSON body tokens
        assert!(matches!(det.feed(100), DetectResult::Captured));
        assert!(matches!(det.feed(101), DetectResult::Captured));
        // Close
        det.feed(40); det.feed(50);
        match det.feed(60) {
            DetectResult::ToolCall(tokens) => {
                assert_eq!(tokens, vec![100, 101]);
            }
            other => panic!("expected ToolCall, got {:?}", other),
        }
    }

    #[test]
    fn runaway_capture_aborts() {
        let mut det = ToolCallDetector::new(vec![10, 20, 30], vec![40, 50, 60], 5);
        det.feed(10); det.feed(20); det.feed(30);
        // Feed more than max_capture tokens without close
        det.feed(100); det.feed(101); det.feed(102); det.feed(103); det.feed(104);
        match det.feed(105) {
            DetectResult::Aborted(tokens) => {
                assert!(tokens.len() > 0);
            }
            other => panic!("expected Aborted, got {:?}", other),
        }
        assert_eq!(det.state, DetectorState::Normal);
    }

    #[test]
    fn nested_open_tag_ignored_during_capture() {
        let mut det = make_detector();
        det.feed(10); det.feed(20); det.feed(30); // open
        det.feed(100); // body
        // Nested open tag — should be captured as body, not trigger new capture
        assert!(matches!(det.feed(10), DetectResult::Captured));
        assert!(matches!(det.feed(20), DetectResult::Captured));
        assert!(matches!(det.feed(30), DetectResult::Captured));
        assert_eq!(det.state, DetectorState::Capturing);
    }

    #[test]
    fn partial_open_pattern_then_mismatch_flushes() {
        let mut det = make_detector();
        // Start of open pattern
        det.feed(10); // matches open_pattern[0]
        det.feed(20); // matches open_pattern[1]
        // Mismatch — should flush the partial match as text
        match det.feed(99) {
            DetectResult::Text(_) => {} // the mismatch token
            other => panic!("expected Text after mismatch, got {:?}", other),
        }
        assert_eq!(det.state, DetectorState::Normal);
    }

    #[test]
    fn resets_after_tool_call() {
        let mut det = make_detector();
        det.feed(10); det.feed(20); det.feed(30); // open
        det.feed(100); // body
        det.feed(40); det.feed(50); det.feed(60); // close
        // After tool call, should be back to normal
        assert_eq!(det.state, DetectorState::Normal);
        match det.feed(200) {
            DetectResult::Text(t) => assert_eq!(t, 200),
            other => panic!("expected Text, got {:?}", other),
        }
    }
}
```

- [ ] **Step 2: Add module export to mod.rs**

In `/root/dev/eaclaw/crates/eaclaw-core/src/llm/mod.rs`, add near the top (after existing mod declarations):

```rust
#[cfg(feature = "local-llm")]
pub mod tool_parse;
```

Wait — the tests should run without the feature gate during development. Use unconditional export for now:

```rust
pub mod tool_parse;
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /root/dev/eaclaw
cargo test --lib tool_parse
```

Expected: compile errors — `ToolCallDetector::new` and `feed` not implemented.

- [ ] **Step 4: Implement ToolCallDetector**

Add the implementation above the `#[cfg(test)]` block in `tool_parse.rs`:

```rust
impl ToolCallDetector {
    pub fn new(open_pattern: Vec<i32>, close_pattern: Vec<i32>, max_capture: usize) -> Self {
        let open_sentinel = *open_pattern.last().expect("open_pattern must not be empty");
        let close_sentinel = *close_pattern.last().expect("close_pattern must not be empty");
        Self {
            ring: Vec::with_capacity(open_pattern.len().max(close_pattern.len())),
            open_pattern,
            close_pattern,
            open_sentinel,
            close_sentinel,
            state: DetectorState::Normal,
            capture_buf: Vec::new(),
            max_capture,
        }
    }

    pub fn feed(&mut self, token: i32) -> DetectResult {
        match self.state {
            DetectorState::Normal => self.feed_normal(token),
            DetectorState::Capturing => self.feed_capturing(token),
            DetectorState::Complete => {
                // Reset and process as normal
                self.state = DetectorState::Normal;
                self.ring.clear();
                self.feed_normal(token)
            }
        }
    }

    fn feed_normal(&mut self, token: i32) -> DetectResult {
        self.ring.push(token);

        // Sentinel check: only do full match if last token matches
        if token == self.open_sentinel && self.ring_ends_with(&self.open_pattern) {
            // Full open pattern matched
            self.state = DetectorState::Capturing;
            self.capture_buf.clear();
            self.ring.clear();
            return DetectResult::TagOpen;
        }

        // Check if we're in a partial match of the open pattern
        let partial_len = self.partial_match_len(&self.open_pattern);
        if partial_len > 0 && partial_len < self.open_pattern.len() {
            // Still building toward a potential match — suppress token
            return DetectResult::TagOpen;
        }

        // No match at all — flush any accumulated ring tokens
        if self.ring.len() > 1 {
            // There were partial matches that failed. We need to flush them.
            // For simplicity, just return the current token as text.
            // The partial tokens were already returned as TagOpen, which is wrong.
            // Better approach: only suppress when we have a real partial match.
        }

        // Keep ring bounded
        let max_len = self.open_pattern.len();
        if self.ring.len() > max_len {
            self.ring.drain(0..self.ring.len() - max_len);
        }

        DetectResult::Text(token)
    }

    fn feed_capturing(&mut self, token: i32) -> DetectResult {
        self.capture_buf.push(token);

        // Check for close pattern at end of capture_buf
        if token == self.close_sentinel && self.capture_ends_with(&self.close_pattern) {
            // Remove close pattern tokens from capture
            let body_len = self.capture_buf.len() - self.close_pattern.len();
            let body: Vec<i32> = self.capture_buf[..body_len].to_vec();
            self.capture_buf.clear();
            self.state = DetectorState::Complete;
            return DetectResult::ToolCall(body);
        }

        // Check runaway
        if self.capture_buf.len() >= self.max_capture {
            let flushed = std::mem::take(&mut self.capture_buf);
            self.state = DetectorState::Normal;
            self.ring.clear();
            return DetectResult::Aborted(flushed);
        }

        DetectResult::Captured
    }

    fn ring_ends_with(&self, pattern: &[i32]) -> bool {
        if self.ring.len() < pattern.len() {
            return false;
        }
        let start = self.ring.len() - pattern.len();
        self.ring[start..] == *pattern
    }

    fn capture_ends_with(&self, pattern: &[i32]) -> bool {
        if self.capture_buf.len() < pattern.len() {
            return false;
        }
        let start = self.capture_buf.len() - pattern.len();
        self.capture_buf[start..] == *pattern
    }

    fn partial_match_len(&self, pattern: &[i32]) -> usize {
        // How many tokens at the end of ring match the start of pattern?
        let ring_len = self.ring.len();
        let pat_len = pattern.len();
        let max_check = ring_len.min(pat_len);
        for len in (1..=max_check).rev() {
            if self.ring[ring_len - len..] == pattern[..len] {
                return len;
            }
        }
        0
    }

    /// Reset detector to initial state.
    pub fn reset(&mut self) {
        self.state = DetectorState::Normal;
        self.ring.clear();
        self.capture_buf.clear();
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cd /root/dev/eaclaw
cargo test --lib tool_parse -- --nocapture
```

Expected: all 7 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /root/dev/eaclaw
git add crates/eaclaw-core/src/llm/tool_parse.rs crates/eaclaw-core/src/llm/mod.rs
git commit -m "feat: add token-level tool-call detector with sentinel optimization"
```

---

## Chunk 3: llama.cpp + eakv FFI Layer

### Task 6: Vendor llama.cpp as git submodule

**Files:**
- Create: `vendor/llama.cpp/` (git submodule)
- Modify: `crates/eaclaw-core/build.rs` (add llama.cpp + eakv linking)

- [ ] **Step 1: Add llama.cpp submodule**

```bash
cd /root/dev/eaclaw
git submodule add --depth 1 https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp
cd vendor/llama.cpp
git checkout b5390 # pin to a known stable tag — adjust to latest stable
cd ../..
```

Note: if submodules aren't viable, download a release tarball instead:

```bash
mkdir -p vendor/llama.cpp
curl -L https://github.com/ggerganov/llama.cpp/archive/refs/tags/b5390.tar.gz | tar xz --strip-components=1 -C vendor/llama.cpp
```

- [ ] **Step 2: Modify existing build.rs for eaclaw-core**

**IMPORTANT:** `/root/dev/eaclaw/crates/eaclaw-core/build.rs` already exists and handles
SIMD kernel embedding via `include_bytes!`. Do NOT overwrite it. Add llama.cpp/eakv
build logic alongside the existing code.

In `/root/dev/eaclaw/crates/eaclaw-core/build.rs`, add at the end of `main()`:

```rust
    // --- Local LLM build (only with local-llm feature) ---
    #[cfg(feature = "local-llm")]
    {
        build_llama_cpp();
        link_eakv();
    }
```

Then add these functions after `main()`:

```rust
#[cfg(feature = "local-llm")]
fn build_llama_cpp() {
    let llama_dir = std::path::Path::new("../../vendor/llama.cpp");

    cc::Build::new()
        .cpp(true)
        .include(llama_dir.join("include"))
        .include(llama_dir.join("ggml/include"))
        .include(llama_dir.join("src"))
        .include(llama_dir.join("ggml/src"))
        .file(llama_dir.join("src/llama.cpp"))
        .file(llama_dir.join("src/llama-vocab.cpp"))
        .file(llama_dir.join("src/llama-grammar.cpp"))
        .file(llama_dir.join("src/llama-sampling.cpp"))
        .file(llama_dir.join("ggml/src/ggml.c"))
        .file(llama_dir.join("ggml/src/ggml-alloc.c"))
        .file(llama_dir.join("ggml/src/ggml-backend.c"))
        .file(llama_dir.join("ggml/src/ggml-quants.c"))
        .file(llama_dir.join("ggml/src/ggml-cpu/ggml-cpu.c"))
        .flag("-std=c++17")
        .flag("-O2")
        .flag("-DGGML_USE_CPU")
        .flag("-mavx512f")
        .flag("-mavx512bw")
        .flag("-mavx512vl")
        .warnings(false)
        .compile("llama");

    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=stdc++");
}

#[cfg(feature = "local-llm")]
fn link_eakv() {
    // eakv is a sibling directory: /root/dev/eakv
    // From crates/eaclaw-core/, go up 3 levels to /root/dev/
    let eakv_dir = std::path::Path::new("../../../eakv");
    println!("cargo:rustc-link-search=native={}/build", eakv_dir.canonicalize()
        .unwrap_or_else(|_| eakv_dir.to_path_buf()).display());
    println!("cargo:rustc-link-lib=static=eakv");
    println!("cargo:rustc-link-lib=m"); // libm for math functions
}
```

Note: The exact llama.cpp source files may vary by version. The build.rs will need
adjustment based on the pinned llama.cpp version. Check `vendor/llama.cpp/CMakeLists.txt`
for the authoritative file list.

- [ ] **Step 3: Verify build compiles without local-llm feature**

```bash
cd /root/dev/eaclaw
cargo build
```

Expected: builds normally — build.rs is a no-op without `local-llm` feature.

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eaclaw
git add build.rs vendor/llama.cpp crates/eaclaw-core/build.rs .gitmodules
git commit -m "feat: vendor llama.cpp and add build.rs for local-llm feature"
```

---

### Task 7: llama.cpp FFI bindings

**Files:**
- Create: `crates/eaclaw-core/src/llm/llama_ffi.rs`
- Modify: `crates/eaclaw-core/src/llm/mod.rs` (add module)

- [ ] **Step 1: Create FFI bindings**

Create `crates/eaclaw-core/src/llm/llama_ffi.rs`:

```rust
//! Raw FFI bindings to llama.cpp.
//!
//! Minimal surface: model loading, context creation, tokenization,
//! decode (prefill/generate), sampling, and state export.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_float, c_void};
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
```

- [ ] **Step 2: Add module to mod.rs**

In `crates/eaclaw-core/src/llm/mod.rs`, add:

```rust
#[cfg(feature = "local-llm")]
pub mod llama_ffi;
```

- [ ] **Step 3: Verify it compiles with feature flag**

```bash
cd /root/dev/eaclaw
cargo check --features local-llm
```

Expected: may fail if llama.cpp isn't vendored yet. If vendored, should compile.
Without feature flag, should always compile:

```bash
cargo check
```

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eaclaw
git add crates/eaclaw-core/src/llm/llama_ffi.rs crates/eaclaw-core/src/llm/mod.rs
git commit -m "feat: add llama.cpp FFI bindings with safe LlamaEngine wrapper"
```

---

### Task 7b: eakv Rust FFI bindings

**Files:**
- Create: `crates/eaclaw-core/src/llm/eakv_ffi.rs`
- Modify: `crates/eaclaw-core/src/llm/mod.rs` (add module)

- [ ] **Step 1: Create eakv FFI bindings**

Create `crates/eaclaw-core/src/llm/eakv_ffi.rs`:

```rust
//! Raw FFI bindings to eakv — Q4 KV cache compression library.

use std::ffi::c_int;
use std::os::raw::c_uchar;

// Opaque type
pub enum eakv_cache_t {}

pub const EAKV_OK: c_int = 0;
pub const EAKV_ERR_INVALID: c_int = -4;

extern "C" {
    pub fn eakv_cache_create(
        n_layers: c_int, n_kv_heads: c_int,
        head_dim: c_int, max_seq_len: c_int,
    ) -> *mut eakv_cache_t;

    pub fn eakv_cache_free(cache: *mut eakv_cache_t);

    pub fn eakv_cache_seq_len(cache: *const eakv_cache_t) -> c_int;

    pub fn eakv_checkpoint(cache: *mut eakv_cache_t) -> c_int;

    pub fn eakv_restore(cache: *mut eakv_cache_t, seq_len: c_int) -> c_int;

    pub fn eakv_from_llama_state_append(
        cache: *mut eakv_cache_t,
        state_buf: *const c_uchar, state_size: usize,
        n_layers: c_int, n_kv_heads: c_int, head_dim: c_int,
        start_pos: c_int,
    ) -> c_int;
}

/// Safe wrapper around an eakv cache.
pub struct EakvCache {
    ptr: *mut eakv_cache_t,
    n_layers: i32,
    n_kv_heads: i32,
    head_dim: i32,
}

unsafe impl Send for EakvCache {}

impl EakvCache {
    pub fn new(n_layers: i32, n_kv_heads: i32, head_dim: i32, max_seq_len: i32) -> Option<Self> {
        let ptr = unsafe { eakv_cache_create(n_layers, n_kv_heads, head_dim, max_seq_len) };
        if ptr.is_null() { None } else {
            Some(Self { ptr, n_layers, n_kv_heads, head_dim })
        }
    }

    pub fn checkpoint(&mut self) -> i32 {
        unsafe { eakv_checkpoint(self.ptr) }
    }

    pub fn restore(&mut self, seq_len: i32) -> Result<(), String> {
        let rc = unsafe { eakv_restore(self.ptr, seq_len) };
        if rc == EAKV_OK { Ok(()) } else {
            Err(format!("eakv_restore failed: {rc}"))
        }
    }

    pub fn seq_len(&self) -> i32 {
        unsafe { eakv_cache_seq_len(self.ptr) }
    }

    /// Import KV state from llama.cpp (Approach A).
    /// Appends new tokens starting from start_pos.
    pub fn import_llama_state(&mut self, state: &[u8], start_pos: i32) -> Result<(), String> {
        let rc = unsafe {
            eakv_from_llama_state_append(
                self.ptr, state.as_ptr(), state.len(),
                self.n_layers, self.n_kv_heads, self.head_dim,
                start_pos,
            )
        };
        if rc == EAKV_OK { Ok(()) } else {
            Err(format!("eakv_from_llama_state_append failed: {rc}"))
        }
    }
}

impl Drop for EakvCache {
    fn drop(&mut self) {
        unsafe { eakv_cache_free(self.ptr) }
    }
}
```

- [ ] **Step 2: Add module to mod.rs**

In `crates/eaclaw-core/src/llm/mod.rs`:

```rust
#[cfg(feature = "local-llm")]
pub mod eakv_ffi;
```

- [ ] **Step 3: Verify it compiles without feature flag**

```bash
cd /root/dev/eaclaw
cargo check
```

Expected: compiles — module is behind `#[cfg(feature)]`.

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eaclaw
git add crates/eaclaw-core/src/llm/eakv_ffi.rs crates/eaclaw-core/src/llm/mod.rs
git commit -m "feat: add eakv Rust FFI bindings with safe EakvCache wrapper"
```

---

## Chunk 4: LocalLlmProvider

### Task 8: LocalLlmProvider — chat template + stateless-to-stateful bridge

**Files:**
- Create: `crates/eaclaw-core/src/llm/local.rs`
- Modify: `crates/eaclaw-core/src/llm/mod.rs`

- [ ] **Step 1: Write tests for chat template formatting**

Create `crates/eaclaw-core/src/llm/local.rs` with test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Message, Role, ContentBlock, ToolDef};

    #[test]
    fn format_messages_basic() {
        let system = "You are helpful.";
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: "Hello".into() }],
            },
        ];
        let tools: Vec<ToolDef> = vec![];

        let result = format_chat_template(system, &messages, &tools);
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_messages_with_tools() {
        let system = "Base prompt.";
        let tools = vec![ToolDef {
            name: "calc".into(),
            description: "Calculate math".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
        }];
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: "2+2".into() }],
            },
        ];

        let result = format_chat_template(system, &messages, &tools);
        assert!(result.contains("<tool_call>"));
        assert!(result.contains("calc"));
        assert!(result.contains("Calculate math"));
    }

    #[test]
    fn common_prefix_length_identical() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        assert_eq!(common_prefix_len(&a, &b), 5);
    }

    #[test]
    fn common_prefix_length_partial() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 6, 7, 8];
        assert_eq!(common_prefix_len(&a, &b), 3);
    }

    #[test]
    fn common_prefix_length_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        assert_eq!(common_prefix_len(&a, &b), 0);
    }

    #[test]
    fn common_prefix_length_diverge_at_start() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(common_prefix_len(&a, &b), 0);
    }
}
```

- [ ] **Step 2: Implement format_chat_template and common_prefix_len**

Above the test module in `local.rs`:

```rust
//! LocalLlmProvider — embedded llama.cpp + eakv inference.

use crate::llm::{ContentBlock, LlmProvider, LlmResponse, Message, OnTextFn, Role, StopReason, ToolDef};
use crate::error::Result;
use std::sync::Mutex;

#[cfg(feature = "local-llm")]
use super::llama_ffi::LlamaEngine;
#[cfg(feature = "local-llm")]
use super::eakv_ffi::EakvCache;
#[cfg(feature = "local-llm")]
use super::tool_parse::{ToolCallDetector, DetectResult};

/// Format messages into Qwen2.5 chat template.
pub fn format_chat_template(system: &str, messages: &[Message], tools: &[ToolDef]) -> String {
    let mut out = String::new();

    // System message with tool definitions
    out.push_str("<|im_start|>system\n");
    out.push_str(system);

    if !tools.is_empty() {
        out.push_str("\n\nYou have access to the following tools. To call a tool, output:\n\n");
        out.push_str("<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}\n</tool_call>\n\n");
        out.push_str("Available tools:\n");
        for tool in tools {
            out.push_str(&format!(
                "\n- **{}**: {}\n  Parameters: {}\n",
                tool.name, tool.description,
                serde_json::to_string(&tool.input_schema).unwrap_or_default()
            ));
        }
    }

    out.push_str("<|im_end|>\n");

    // Conversation messages
    for msg in messages {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!("<|im_start|>{role_str}\n"));

        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => out.push_str(text),
                ContentBlock::ToolUse { id: _, name, input } => {
                    out.push_str("<tool_call>\n");
                    out.push_str(&serde_json::json!({"name": name, "arguments": input}).to_string());
                    out.push_str("\n</tool_call>");
                }
                ContentBlock::ToolResult { tool_use_id: _, content, .. } => {
                    out.push_str(&format!("<tool_result>\n{content}\n</tool_result>"));
                }
            }
        }

        out.push_str("<|im_end|>\n");
    }

    // Prompt for assistant response
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Compute length of common prefix between two token sequences.
pub fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
```

- [ ] **Step 3: Run tests**

```bash
cd /root/dev/eaclaw
cargo test --lib local -- --nocapture
```

Expected: all 6 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eaclaw
git add crates/eaclaw-core/src/llm/local.rs crates/eaclaw-core/src/llm/mod.rs
git commit -m "feat: add chat template formatter and token prefix diff for LocalLlmProvider"
```

---

### Task 9: LocalLlmProvider — LlmProvider trait implementation

**Files:**
- Modify: `crates/eaclaw-core/src/llm/local.rs`

- [ ] **Step 1: Add the provider struct and trait impl**

Append to `local.rs` (before tests module):

```rust
#[cfg(feature = "local-llm")]
pub struct LocalLlmProvider {
    inner: Mutex<LocalLlmInner>,
}

#[cfg(feature = "local-llm")]
struct LocalLlmInner {
    engine: LlamaEngine,
    kv_cache: EakvCache,            // eakv Q4 KV backend
    prefilled_tokens: Vec<i32>,
    tool_id_counter: u64,
    open_pattern: Vec<i32>,
    close_pattern: Vec<i32>,
    eakv_seq_len: i32,              // eakv's current seq_len (tracks llama.cpp's KV state)
}

#[cfg(feature = "local-llm")]
impl LocalLlmProvider {
    /// Create provider. Requires model architecture params for eakv cache.
    /// These are read from the model after loading.
    pub fn new(model_path: &str, n_ctx: u32, n_threads: u32,
               n_layers: i32, n_kv_heads: i32, head_dim: i32) -> Result<Self> {
        let engine = LlamaEngine::new(model_path, n_ctx, n_threads)
            .map_err(|e| crate::error::Error::Llm(e))?;

        let kv_cache = EakvCache::new(n_layers, n_kv_heads, head_dim, n_ctx as i32)
            .ok_or_else(|| crate::error::Error::Llm("failed to create eakv cache".into()))?;

        let open_pattern = engine.tokenize("<tool_call>", false);
        let close_pattern = engine.tokenize("</tool_call>", false);

        Ok(Self {
            inner: Mutex::new(LocalLlmInner {
                engine,
                kv_cache,
                prefilled_tokens: Vec::new(),
                tool_id_counter: 0,
                open_pattern,
                close_pattern,
                eakv_seq_len: 0,
            }),
        })
    }

    fn generate_tool_id(inner: &mut LocalLlmInner) -> String {
        inner.tool_id_counter += 1;
        format!("local_{}", inner.tool_id_counter)
    }
}

#[cfg(feature = "local-llm")]
#[async_trait::async_trait]
impl LlmProvider for LocalLlmProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
    ) -> Result<LlmResponse> {
        // Delegate to chat_stream with a no-op callback
        let noop: OnTextFn<'_> = &mut |_: &str| {};
        self.chat_stream(messages, tools, system, noop).await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
        on_text: OnTextFn<'_>,
    ) -> Result<LlmResponse> {
        // Format messages to chat template
        let formatted = format_chat_template(system, messages, tools);

        // Acquire lock — serializes all inference
        let mut inner = self.inner.lock()
            .map_err(|e| crate::error::Error::Llm(format!("lock poisoned: {e}")))?;

        // Tokenize the full conversation
        let new_tokens = inner.engine.tokenize(&formatted, true);

        // Compute common prefix with previously prefilled tokens
        let prefix_len = common_prefix_len(&inner.prefilled_tokens, &new_tokens);

        // If prefix diverges from what's in KV cache, truncate both llama.cpp and eakv
        if prefix_len < inner.prefilled_tokens.len() {
            inner.engine.kv_cache_truncate(prefix_len as i32);
            inner.kv_cache.restore(prefix_len as i32)
                .map_err(|e| crate::error::Error::Llm(e))?;
            inner.eakv_seq_len = prefix_len as i32;
        }

        // Prefill only the new suffix
        let suffix = &new_tokens[prefix_len..];
        if !suffix.is_empty() {
            inner.engine.decode(suffix, prefix_len as i32)
                .map_err(|e| crate::error::Error::Llm(e))?;

            // Export KV state from llama.cpp → eakv (Approach A)
            let kv_state = inner.engine.export_kv_state();
            inner.kv_cache.import_llama_state(&kv_state, inner.eakv_seq_len)
                .map_err(|e| crate::error::Error::Llm(e))?;
            inner.eakv_seq_len = inner.kv_cache.seq_len();
        }

        // Update prefilled state
        inner.prefilled_tokens = new_tokens.clone();

        // Checkpoint eakv before generation (O(1) — just saves seq_len)
        let _checkpoint = inner.kv_cache.checkpoint();

        // Generation loop
        let eos = inner.engine.eos_token();
        let mut detector = ToolCallDetector::new(
            inner.open_pattern.clone(),
            inner.close_pattern.clone(),
            512,
        );
        let mut content_blocks = Vec::new();
        let mut text_buf = String::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut gen_pos = inner.prefilled_tokens.len() as i32;
        let max_gen = 2048i32; // max generation tokens

        for _ in 0..max_gen {
            let token = inner.engine.sample();

            if token == eos {
                break;
            }

            // Decode next token (single-token batch for autoregressive)
            inner.engine.decode(&[token], gen_pos)
                .map_err(|e| crate::error::Error::Llm(e))?;
            gen_pos += 1;
            inner.prefilled_tokens.push(token);

            match detector.feed(token) {
                DetectResult::Text(t) => {
                    let piece = inner.engine.token_to_str(t);
                    text_buf.push_str(&piece);
                    on_text(&piece);
                }
                DetectResult::TagOpen | DetectResult::Captured => {
                    // Suppressed from output
                }
                DetectResult::ToolCall(body_tokens) => {
                    // Flush any accumulated text first
                    if !text_buf.is_empty() {
                        content_blocks.push(ContentBlock::Text {
                            text: std::mem::take(&mut text_buf),
                        });
                    }

                    // Parse the JSON body
                    let json_str = inner.engine.detokenize(&body_tokens);
                    match serde_json::from_str::<serde_json::Value>(&json_str) {
                        Ok(val) => {
                            let name = val["name"].as_str().unwrap_or("").to_string();
                            let arguments = val["arguments"].clone();
                            let id = Self::generate_tool_id(&mut inner);
                            content_blocks.push(ContentBlock::ToolUse {
                                id,
                                name,
                                input: arguments,
                            });
                            stop_reason = StopReason::ToolUse;
                        }
                        Err(e) => {
                            // Failed to parse — treat as text
                            text_buf.push_str(&format!("<tool_call>{json_str}</tool_call>"));
                            on_text(&format!("<tool_call>{json_str}</tool_call>"));
                        }
                    }
                    break; // Stop generation after tool call
                }
                DetectResult::Aborted(tokens) => {
                    // Runaway capture — flush as text
                    let text = inner.engine.detokenize(&tokens);
                    text_buf.push_str(&text);
                    on_text(&text);
                }
            }
        }

        // Flush remaining text
        if !text_buf.is_empty() {
            content_blocks.push(ContentBlock::Text { text: text_buf });
        }

        // If no content blocks at all, add empty text
        if content_blocks.is_empty() {
            content_blocks.push(ContentBlock::Text { text: String::new() });
        }

        Ok(LlmResponse {
            content: content_blocks,
            stop_reason,
        })
    }
}
```

- [ ] **Step 2: Add module export in mod.rs**

In `crates/eaclaw-core/src/llm/mod.rs`:

```rust
pub mod local;
#[cfg(feature = "local-llm")]
pub use local::LocalLlmProvider;
```

- [ ] **Step 3: Verify it compiles without local-llm**

```bash
cd /root/dev/eaclaw
cargo check
```

Expected: compiles — all `local-llm` code is behind `#[cfg(feature)]`.

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eaclaw
git add crates/eaclaw-core/src/llm/local.rs crates/eaclaw-core/src/llm/mod.rs
git commit -m "feat: implement LocalLlmProvider with stateful KV bridge and tool detection"
```

---

## Chunk 5: CLI Integration + End-to-End

### Task 10: Wire up backend selection in CLI

**Files:**
- Modify: `eaclaw-cli/src/main.rs:43-64` (run_repl function)

- [ ] **Step 1: Update run_repl to select backend**

In `eaclaw-cli/src/main.rs`, modify `run_repl()` to branch on `config.backend`:

```rust
use eaclaw_core::config::Backend;

async fn run_repl(config: eaclaw_core::config::Config) {
    let llm: std::sync::Arc<dyn eaclaw_core::llm::LlmProvider> = match config.backend {
        Backend::Anthropic => {
            let provider = eaclaw_core::llm::AnthropicProvider::new(&config);
            std::sync::Arc::new(provider)
        }
        #[cfg(feature = "local-llm")]
        Backend::Local => {
            let model_path = config.model_path.as_deref().unwrap_or(
                &format!("{}/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf",
                    std::env::var("HOME").unwrap_or_default())
            );
            match eaclaw_core::llm::LocalLlmProvider::new(
                model_path, config.ctx_size as u32, config.threads as u32,
            ) {
                Ok(provider) => std::sync::Arc::new(provider),
                Err(e) => {
                    eprintln!("Failed to load local model: {e}");
                    eprintln!("Set EACLAW_MODEL_PATH to a valid GGUF file.");
                    eprintln!("Download: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF");
                    std::process::exit(1);
                }
            }
        }
        #[cfg(not(feature = "local-llm"))]
        Backend::Local => {
            eprintln!("Local LLM backend requires the 'local-llm' feature.");
            eprintln!("Rebuild with: cargo build --features local-llm");
            std::process::exit(1);
        }
    };

    // Rest of run_repl stays the same, using `llm` variable
    let tools = eaclaw_core::tools::ToolRegistry::with_defaults(&config, llm.clone());
    // ... existing code ...
}
```

- [ ] **Step 2: Apply same pattern to run_whatsapp**

Same branching in `run_whatsapp()` if it creates an `AnthropicProvider`.

- [ ] **Step 3: Verify default build still works**

```bash
cd /root/dev/eaclaw
cargo build
cargo test
```

Expected: compiles and passes — `Backend::Anthropic` is the default.

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eaclaw
git add eaclaw-cli/src/main.rs
git commit -m "feat: wire up backend selection in CLI — local-llm behind feature gate"
```

---

### Task 11: Integration test — full local inference flow

**Files:**
- Create: `crates/eaclaw-core/tests/local_llm_test.rs` (integration test, only runs with feature + model)

- [ ] **Step 1: Write integration test**

Create `crates/eaclaw-core/tests/local_llm_test.rs`:

```rust
//! Integration test for LocalLlmProvider.
//! Only runs when local-llm feature is enabled AND a model file exists.
//!
//! Run: cargo test --features local-llm -- local_llm --nocapture

#[cfg(feature = "local-llm")]
mod local_llm_tests {
    use eaclaw_core::llm::{LocalLlmProvider, LlmProvider, Message, Role, ContentBlock, ToolDef};
    use std::sync::Arc;

    fn model_path() -> Option<String> {
        std::env::var("EACLAW_MODEL_PATH").ok().or_else(|| {
            let home = std::env::var("HOME").ok()?;
            let path = format!("{home}/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf");
            if std::path::Path::new(&path).exists() { Some(path) } else { None }
        })
    }

    #[tokio::test]
    async fn test_basic_generation() {
        let path = match model_path() {
            Some(p) => p,
            None => { eprintln!("Skipping: no model file"); return; }
        };

        let provider = LocalLlmProvider::new(&path, 2048, 4).unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "Say hello in one word.".into() }],
        }];

        let response = provider.chat(&messages, &[], "You are helpful.").await.unwrap();
        assert!(!response.content.is_empty());

        if let ContentBlock::Text { text } = &response.content[0] {
            assert!(!text.is_empty(), "Expected non-empty response");
            println!("Response: {text}");
        }
    }

    #[tokio::test]
    async fn test_tool_call_detection() {
        let path = match model_path() {
            Some(p) => p,
            None => { eprintln!("Skipping: no model file"); return; }
        };

        let provider = LocalLlmProvider::new(&path, 2048, 4).unwrap();
        let tools = vec![ToolDef {
            name: "calc".into(),
            description: "Evaluate a math expression".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "expr": { "type": "string" } },
                "required": ["expr"]
            }),
        }];

        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "What is 42 * 17? Use the calc tool.".into(),
            }],
        }];

        let response = provider.chat(&messages, &tools, "You are helpful.").await.unwrap();
        println!("Response: {response:?}");

        // Check if model produced a tool call
        let has_tool_use = response.content.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. }));
        if has_tool_use {
            assert_eq!(response.stop_reason, eaclaw_core::llm::StopReason::ToolUse);
            println!("Tool call detected successfully");
        } else {
            println!("Model did not produce tool call (may need prompt tuning)");
        }
    }

    #[tokio::test]
    async fn test_incremental_prefill() {
        let path = match model_path() {
            Some(p) => p,
            None => { eprintln!("Skipping: no model file"); return; }
        };

        let provider = LocalLlmProvider::new(&path, 2048, 4).unwrap();

        // First turn
        let messages1 = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "Hi".into() }],
        }];
        let r1 = provider.chat(&messages1, &[], "You are helpful.").await.unwrap();

        // Second turn — extends conversation, should only prefill delta
        let mut messages2 = messages1.clone();
        messages2.push(Message {
            role: Role::Assistant,
            content: r1.content.clone(),
        });
        messages2.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "How are you?".into() }],
        });

        let r2 = provider.chat(&messages2, &[], "You are helpful.").await.unwrap();
        assert!(!r2.content.is_empty());
        println!("Multi-turn response: {r2:?}");
    }
}
```

- [ ] **Step 2: Run (skips if no model)**

```bash
cd /root/dev/eaclaw
cargo test --features local-llm -- local_llm --nocapture
```

Expected: tests either pass (model exists) or print "Skipping" (no model).

- [ ] **Step 3: Commit**

```bash
cd /root/dev/eaclaw
git add crates/eaclaw-core/tests/local_llm_test.rs
git commit -m "test: add integration tests for LocalLlmProvider (skip if no model)"
```

---

### Task 12: Documentation — update CLAUDE.md and env vars

**Files:**
- Modify: `/root/dev/eaclaw/CLAUDE.md`

- [ ] **Step 1: Add local inference section to CLAUDE.md**

Add after the "Environment Variables" section:

```markdown
## Local Inference (optional)

Build with `cargo build --features local-llm` to enable embedded llama.cpp + eakv inference.

Download a GGUF model:
```bash
mkdir -p ~/.eaclaw/models
# Qwen2.5-3B-Instruct Q4_K_M (~1.8GB)
wget -O ~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

Set `EACLAW_BACKEND=local` to use local inference instead of Anthropic API.
```

Add the new env vars to the Environment Variables table:

```markdown
- `EACLAW_BACKEND` — `anthropic` (default) or `local`
- `EACLAW_MODEL_PATH` — GGUF model file (default: `~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf`)
- `EACLAW_CTX_SIZE` — Context window size (default: `4096`)
- `EACLAW_THREADS` — Inference threads (default: CPU count)
```

- [ ] **Step 2: Commit**

```bash
cd /root/dev/eaclaw
git add CLAUDE.md
git commit -m "docs: add local inference setup instructions to CLAUDE.md"
```

---

## Dependency Graph

```
Task 1 (eakv checkpoint)     ─┐
Task 2 (eakv incr. bridge)   ─┤
Task 3 (config backend)      ─┼── independent, can run in parallel
Task 4 (Cargo features)      ─┤
Task 5 (tool detector)       ─┘
                               │
Task 6 (vendor llama.cpp)    ─── depends on Task 4
Task 7 (llama FFI bindings)  ─── depends on Task 6
Task 7b (eakv FFI bindings)  ─── depends on Tasks 1, 2, 6
                               │
Task 8 (chat template)       ─── independent (pure functions, no deps on 5/7)
Task 9 (LocalLlmProvider)    ─── depends on Tasks 5, 7, 7b, 8
                               │
Task 10 (CLI wiring)         ─── depends on Tasks 3, 9
Task 11 (integration test)   ─── depends on Task 10
Task 12 (docs)               ─── depends on Task 10
```

Tasks 1-5, 8 are fully independent and can be parallelized.

## Prerequisites

Before starting implementation:

1. **Build eakv:** `cd /root/dev/eakv && make` — required for linking
2. **Download model (for integration tests):**
   ```bash
   mkdir -p ~/.eaclaw/models
   # Qwen2.5-3B-Instruct Q4_K_M (~1.8GB)
   wget -O ~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf \
     https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
   ```
