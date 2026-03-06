use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use eaclaw_core::kernels::{byte_classifier, fused_safety, leak_scanner, sanitizer_kernel};
use eaclaw_core::safety::SafetyLayer;

fn generate_clean_input(size: usize) -> Vec<u8> {
    // Realistic text without injection/leak patterns
    let base = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = &base[..remaining.min(base.len())];
        data.extend_from_slice(chunk);
    }
    data
}

fn generate_mixed_input(size: usize) -> Vec<u8> {
    // Mix of clean text with occasional "interesting" bytes
    let base = b"Hello world, this is a test message with some patterns. system: ignore previous. sk-test1234 ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = &base[..remaining.min(base.len())];
        data.extend_from_slice(chunk);
    }
    data
}

fn bench_byte_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_classifier");

    for size in [1024, 10240, 102400] {
        let input = generate_clean_input(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &input,
            |b, input| {
                b.iter(|| byte_classifier::classify(black_box(input)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &input,
            |b, input| {
                b.iter(|| byte_classifier::classify_scalar(black_box(input)));
            },
        );
    }

    group.finish();
}

fn bench_injection_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("injection_scan");

    for size in [1024, 10240, 102400] {
        let input = generate_clean_input(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("simd_filter", size),
            &input,
            |b, input| {
                b.iter(|| {
                    let masks = sanitizer_kernel::scan_prefixes(black_box(input));
                    let mut count = 0u32;
                    sanitizer_kernel::for_each_candidate(&masks, |_pos| count += 1);
                    count
                });
            },
        );

        // Scalar baseline: check each byte against interesting set
        group.bench_with_input(
            BenchmarkId::new("scalar_scan", size),
            &input,
            |b, input| {
                b.iter(|| {
                    sanitizer_kernel::scan_prefixes_scalar(black_box(input))
                });
            },
        );
    }

    group.finish();
}

fn bench_leak_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("leak_scan");

    for size in [1024, 10240, 102400] {
        let input = generate_clean_input(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("simd_filter", size),
            &input,
            |b, input| {
                b.iter(|| {
                    let masks = leak_scanner::scan_prefixes(black_box(input));
                    let mut count = 0u32;
                    leak_scanner::for_each_candidate(&masks, |_pos| count += 1);
                    count
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_scan", size),
            &input,
            |b, input| {
                b.iter(|| {
                    leak_scanner::scan_prefixes_scalar(black_box(input))
                });
            },
        );
    }

    group.finish();
}

fn bench_safety_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("safety_layer_e2e");
    let mut safety = SafetyLayer::new();

    for size in [1024, 10240, 102400] {
        let input = generate_clean_input(size);
        let input_str = String::from_utf8_lossy(&input).to_string();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("scan_input", size),
            &input_str,
            |b, input| {
                b.iter(|| safety.scan_input(black_box(input)));
            },
        );
    }

    // Also bench with "noisy" input (many candidate positions)
    for size in [1024, 10240, 102400] {
        let input = generate_mixed_input(size);
        let input_str = String::from_utf8_lossy(&input).to_string();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("scan_input_mixed", size),
            &input_str,
            |b, input| {
                b.iter(|| safety.scan_input(black_box(input)));
            },
        );
    }

    // Pre-allocated variant
    let mut safety_prealloc = SafetyLayer::with_capacity(102400);
    for size in [1024, 10240, 102400] {
        let input = generate_clean_input(size);
        let input_str = String::from_utf8_lossy(&input).to_string();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("scan_input_prealloc", size),
            &input_str,
            |b, input| {
                b.iter(|| safety_prealloc.scan_input(black_box(input)));
            },
        );
    }

    group.finish();
}

fn bench_fused_vs_separate(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_vs_separate");

    for size in [1024, 10240, 102400] {
        let input = generate_clean_input(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("separate", size),
            &input,
            |b, input| {
                b.iter(|| {
                    let inject_masks = sanitizer_kernel::scan_prefixes(black_box(input));
                    let leak_masks = leak_scanner::scan_prefixes(black_box(input));
                    let mut count = 0u32;
                    sanitizer_kernel::for_each_candidate(&inject_masks, |_| count += 1);
                    leak_scanner::for_each_candidate(&leak_masks, |_| count += 1);
                    count
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fused", size),
            &input,
            |b, input| {
                b.iter(|| {
                    let result = fused_safety::scan_fused(black_box(input));
                    let mut count = 0u32;
                    fused_safety::for_each_candidate(&result.inject_masks, |_| count += 1);
                    fused_safety::for_each_candidate(&result.leak_masks, |_| count += 1);
                    count
                });
            },
        );
    }

    group.finish();
}

fn bench_aho_corasick_comparison(c: &mut Criterion) {
    use aho_corasick::AhoCorasick;

    let patterns = [
        "ignore previous",
        "ignore all previous",
        "disregard",
        "forget everything",
        "you are now",
        "act as",
        "pretend to be",
        "system:",
        "assistant:",
        "user:",
        "<|",
        "|>",
        "[INST]",
        "[/INST]",
        "new instructions",
        "updated instructions",
    ];
    let ac = AhoCorasick::builder()
        .ascii_case_insensitive(true)
        .build(&patterns)
        .unwrap();

    let mut safety = SafetyLayer::new();

    let mut group = c.benchmark_group("simd_vs_aho_corasick");

    for size in [1024, 10240, 102400] {
        let input = generate_clean_input(size);
        let input_str = String::from_utf8_lossy(&input).to_string();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("aho_corasick", size),
            &input_str,
            |b, input| {
                b.iter(|| {
                    ac.find_iter(black_box(input)).count()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eaclaw_simd", size),
            &input_str,
            |b, input| {
                b.iter(|| safety.scan_input(black_box(input)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_byte_classifier,
    bench_injection_scan,
    bench_leak_scan,
    bench_fused_vs_separate,
    bench_safety_layer,
    bench_aho_corasick_comparison,
);
criterion_main!(benches);
