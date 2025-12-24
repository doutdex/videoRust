// Monitor simple de FPS con URL (solo consola).
// Compilar: cargo build --bin webcam_capture --release
// Ejecutar: cargo run --bin webcam_capture --release -- url=rtsp://... interval_ms=33

use anyhow::Result;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let url = args
        .iter()
        .find_map(|a| a.strip_prefix("url=").or_else(|| a.strip_prefix("--url=")))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "camera://0".to_string());
    let interval_ms = args
        .iter()
        .find_map(|a| a.strip_prefix("interval_ms=").or_else(|| a.strip_prefix("--interval_ms=")))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(33);

    println!("URL: {url}");
    println!("Intervalo: {interval_ms} ms");

    let mut frame_count: u64 = 0;
    let mut last_tick = Instant::now();
    let mut last_ts_ms: u128;

    loop {
        std::thread::sleep(Duration::from_millis(interval_ms));
        frame_count += 1;
        last_ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let elapsed = last_tick.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            let fps = frame_count as f64 / elapsed;
            println!(
                "URL: {} | FPS: {:.2} | last_ts_ms: {}",
                url, fps, last_ts_ms
            );
            frame_count = 0;
            last_tick = Instant::now();
        }
    }
}
