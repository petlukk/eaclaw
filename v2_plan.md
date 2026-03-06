Right now most AI frameworks look like this:

String
  ↓
regex / parsing
  ↓
heap allocations
  ↓
JSON objects
  ↓
LLM

Which destroys cache locality.

Your design can instead look like a data pipeline.

Core Idea: Streaming Buffer Pipeline

Instead of passing String everywhere, operate on a single contiguous buffer.

Input buffer (UTF-8)
        │
        ▼
byte_classifier (Eä SIMD)
        │
        ▼
json_scanner (Eä SIMD)
        │
        ▼
sanitizer (Eä SIMD)
        │
        ▼
leak_scanner (Eä SIMD)
        │
        ▼
command_router (Eä SIMD)

All kernels read the same buffer.

No copies.
No allocations.

Everything stays hot in L1/L2 cache.

Cache-Friendly Data Layout

In Rust you’d structure the pipeline around a small struct that fits in cache.

Example:

pub struct ScanContext<'a> {
    pub input: &'a [u8],

    pub flags: Vec<u8>,        // byte classification
    pub candidates: Vec<i32>,  // SIMD match positions
    pub json_pos: Vec<i32>,    // JSON structure positions

    pub injection_found: bool,
    pub leak_found: bool,
}

Important property:

input → read-only
metadata → small arrays

You never modify the input buffer.

SIMD Kernel Pipeline

Rust orchestrates like this:

fn scan(ctx: &mut ScanContext) {

    classify_bytes(ctx.input, &mut ctx.flags);

    scan_injection_prefixes(
        ctx.input,
        &mut ctx.candidates
    );

    verify_injection(ctx);

    scan_leak_prefixes(
        ctx.input,
        &mut ctx.candidates
    );

    verify_leaks(ctx);
}

Each stage:

sequential memory read
SIMD compare
movemask
candidate extraction

CPU prefetch handles the rest.

Why This Stays in Cache

Typical prompt size:

200–2000 bytes

Cache sizes:

L1 data: 32KB
L1 instruction: 32KB

Your working set:

input buffer      ~2KB
flags             ~2KB
candidate arrays  <1KB
instructions      ~6KB

Total:

< 12KB

That means:

the entire pipeline fits in L1.

Avoid Heap Churn

The biggest cache killer is allocation.

Instead of:

Vec::new() per request

Use:

Vec::with_capacity()

or reuse buffers.

Example:

pub struct Scanner {
    flags: Vec<u8>,
    candidates: Vec<i32>,
}

impl Scanner {
    pub fn scan(&mut self, input: &[u8]) -> ScanResult {
        self.flags.resize(input.len(), 0);
        self.candidates.clear();
        ...
    }
}

Now no allocations happen on the hot path.

Agent Loop Layout

Instead of big message objects:

Vec<Message>

Use a ring buffer.

Example:

message ring buffer
 ┌─────────────┐
 │ user prompt │
 │ tool output │
 │ assistant   │
 └─────────────┘

You only send this to the LLM when required.

The hot path is just:

scan → route → maybe call LLM
Command Routing Trick

Your SIMD router can short-circuit the LLM.

Example:

/time
/http
/echo
/memory

Pipeline:

input
 ↓
match_command (SIMD)
 ↓
if match → run tool immediately
else → LLM

Result:

Many requests never hit the LLM.

Latency becomes microseconds instead of seconds.

The Extreme Version (Really Fun)

You can fuse multiple scans into one streaming pass.

load 16 bytes
classify
check injection
check leak
check command prefix

But only do this if profiling shows it helps.

Your current separate kernel design is cleaner.

The Mental Model

Instead of thinking:

chatbot framework

Think:

vectorized text packet processor

Your pipeline becomes similar to:

DPDK packet filtering

hyperscan

vectorized database scans

But applied to AI prompts.

The Cool Consequence

Your chatbot preprocessing could realistically reach:

5–20 GB/s scan throughput

Meaning a 1000 byte prompt processes in ~0.05 µs.

At that point the cost of preprocessing is essentially zero.

The Most Interesting Future Step

Because you control Eä + bindings, you could eventually do something very unusual:

LLM tokenization itself in SIMD kernels.

If you ever do that, you would have:

SIMD tokenizer
SIMD safety
SIMD routing

All in the same architecture.

Almost no one has built that pipeline yet.
