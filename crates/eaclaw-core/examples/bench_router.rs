use eaclaw_core::kernels::command_router;
use std::time::Instant;

fn main() {
    let commands: &[&[u8]] = &[
        b"/help", b"/quit", b"/tools", b"/clear", b"/model", b"/profile",
        b"/time", b"/calc 2+2", b"/shell ls", b"/recall test query",
        b"/memory list", b"/read file.txt", b"/ls /tmp",
    ];

    // Warm up
    for cmd in commands {
        let _ = command_router::match_command_verified(cmd);
    }

    let iters = 1_000_000u32;
    let start = Instant::now();
    for _ in 0..iters {
        for cmd in commands {
            let _ = command_router::match_command_verified(cmd);
        }
    }
    let total = start.elapsed();
    let per_call = total / (iters * commands.len() as u32);
    println!("Command router (verified): {:?}/call ({} commands × {} iters)", per_call, commands.len(), iters);
    println!("Total: {:?}", total);
}
