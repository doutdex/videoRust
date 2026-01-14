use anyhow::Result;

mod pic_analizer;
#[path = "bin/checkRtspFps.rs"]
mod check_rtsp_fps;
#[path = "bin/face_detection.rs"]
mod face_detection;

fn print_help() {
    println!("Uso:");
    println!("  vision <comando> [args...]");
    println!();
    println!("Comandos:");
    println!("  picAnalizer   Ejecuta el CLI principal de imagenes");
    println!("  checkRtspFps     Monitor FPS simple (RTSP)");
    println!("  face_detection  Demo simple sin ML");
    println!();
    println!("Ejemplo:");
    println!("  cargo run -- vision picAnalizer uishow stages=1");
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().collect();
    let _ = args.drain(0..1);
    if args.first().map(|s| s == "vision").unwrap_or(false) {
        let _ = args.drain(0..1);
    }
    if args.is_empty() {
        print_help();
        return Ok(());
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "picAnalizer" | "picanalizer" => {
            let mut forwarded = vec!["picAnalizer".to_string()];
            forwarded.extend(args);
            let code = pic_analizer::run_with_args(forwarded)?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        "checkRtspFps" | "checkrtspfps" => {
            let mut forwarded = vec!["checkRtspFps".to_string()];
            forwarded.extend(args);
            if let Err(err) = check_rtsp_fps::run_with_args(forwarded) {
                return Err(anyhow::anyhow!(err.to_string()));
            }
        }
        "face_detection" => {
            let mut forwarded = vec!["face_detection".to_string()];
            forwarded.extend(args);
            if let Err(err) = face_detection::run_with_args(forwarded) {
                return Err(anyhow::anyhow!(err.to_string()));
            }
        }
        "-h" | "--help" | "help" => {
            print_help();
        }
        _ => {
            eprintln!("Comando no reconocido: {cmd}");
            print_help();
            std::process::exit(1);
        }
    }
    Ok(())
}
