use anyhow::Result;

#[path = "../pic_analizer.rs"]
mod pic_analizer;

fn main() -> Result<()> {
    let code = pic_analizer::run_with_args(std::env::args().collect())?;
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}
