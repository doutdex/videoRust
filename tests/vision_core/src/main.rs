use anyhow::{bail, Context, Result};
use std::env;
use vision_core::{
    build_json_output, default_score_thr, DeviceMode, EmbeddingMode, FaceModel, ProcessConfig,
    Processor, ResizeMode, StageMode,
};

struct CliArgs {
    infile: String,
    models: Vec<FaceModel>,
    stage_mode: StageMode,
    embedding_mode: EmbeddingMode,
    device_mode: DeviceMode,
    resize_mode: ResizeMode,
    score_thr: Option<f32>,
    json: bool,
}

fn print_help() {
    println!("Usage:");
    println!("  vision_core_test [--model=yunet|scrfd|all] [--infile=path] [--stage=1|11|01|111]");
    println!("                  [--embed=off|qint8|raw] [--device=cpu|gpu] [--resize=fast|balanced|quality]");
    println!("                  [--score=0.10] [--json]");
}

fn find_arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .find_map(|a| a.strip_prefix(key).or_else(|| a.strip_prefix(&format!("--{key}"))))
}

fn parse_models(value: &str) -> Result<Vec<FaceModel>> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "yunet" => Ok(vec![FaceModel::YuNet]),
        "2" | "scrfd" => Ok(vec![FaceModel::Scrfd]),
        "all" => Ok(vec![FaceModel::YuNet, FaceModel::Scrfd]),
        _ => bail!("Unknown model: {value}"),
    }
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        std::process::exit(0);
    }

    let infile = find_arg(args, "infile=")
        .or_else(|| find_arg(args, "img="))
        .unwrap_or("people.jpg")
        .to_string();

    let models = if let Some(value) = find_arg(args, "model=")
        .or_else(|| find_arg(args, "facemodel="))
    {
        parse_models(value)?
    } else {
        vec![FaceModel::YuNet]
    };

    let stage_mode = find_arg(args, "stage=")
        .or_else(|| find_arg(args, "stages="))
        .map(|v| match v.trim() {
            "1" => StageMode::Detection,
            "11" => StageMode::DetectionEmbedding,
            "01" => StageMode::EmbeddingOnly,
            "111" => StageMode::DetectionEmbeddingPose,
            _ => StageMode::DetectionEmbedding,
        })
        .unwrap_or(StageMode::DetectionEmbedding);

    let embedding_mode = find_arg(args, "embed=")
        .map(|v| v.to_ascii_lowercase())
        .map(|v| match v.as_str() {
            "off" | "0" => EmbeddingMode::Off,
            "raw" | "2" => EmbeddingMode::Raw,
            _ => EmbeddingMode::QInt8,
        })
        .unwrap_or(EmbeddingMode::QInt8);

    let device_mode = find_arg(args, "device=")
        .map(|v| v.to_ascii_lowercase())
        .map(|v| if v == "gpu" { DeviceMode::Gpu } else { DeviceMode::Cpu })
        .unwrap_or(DeviceMode::Cpu);

    let resize_mode = find_arg(args, "resize=")
        .map(|v| v.to_ascii_lowercase())
        .map(|v| match v.as_str() {
            "fast" => ResizeMode::Fast,
            "quality" => ResizeMode::Quality,
            _ => ResizeMode::Balanced,
        })
        .unwrap_or(ResizeMode::Balanced);

    let score_thr = find_arg(args, "score=")
        .or_else(|| find_arg(args, "threshold="))
        .and_then(|v| v.parse::<f32>().ok());

    let json = args.iter().any(|a| a == "--json" || a == "json");

    Ok(CliArgs {
        infile,
        models,
        stage_mode,
        embedding_mode,
        device_mode,
        resize_mode,
        score_thr,
        json,
    })
}

fn format_image_type(path: &str) -> String {
    image::ImageFormat::from_path(path)
        .map(|f| format!("{:?}", f))
        .unwrap_or_else(|_| "Unknown".to_string())
}

fn stage_label(stage: StageMode) -> &'static str {
    match stage {
        StageMode::Detection => "1",
        StageMode::DetectionEmbedding => "11",
        StageMode::EmbeddingOnly => "01",
        StageMode::DetectionEmbeddingPose => "111",
    }
}

fn run_model(args: &CliArgs, model: FaceModel) -> Result<()> {
    let img = image::open(&args.infile)
        .with_context(|| format!("Failed to open image: {}", args.infile))?;
    let img_type = format_image_type(&args.infile);
    let rgb = img.to_rgb8();

    let score_thr = args.score_thr.unwrap_or(default_score_thr(model));
    let config = ProcessConfig {
        score_thr,
        stage_mode: args.stage_mode,
        embedding_mode: args.embedding_mode,
    };

    let mut processor = Processor::new(model, args.device_mode, args.resize_mode)?;
    let out = processor.process(&rgb, &config)?;

    if args.json {
        let json = build_json_output(
            &out.detection_meta,
            &args.infile,
            rgb.width(),
            rgb.height(),
            &img_type,
            &out.detect,
            &out.detect.faces,
            out.embedding_info,
            &out.embeddings,
            out.pose_info,
            &args.stage_mode,
            &args.embedding_mode,
            out.pipeline_total,
        );
        print!("{json}");
        return Ok(());
    }

    println!("MODEL: {}", out.detection_meta.model_name);
    println!("  ONNX: {}", out.detection_meta.onnx_name);
    println!(
        "  PRE: input={}x{}, color={}, normalization={}",
        out.detection_meta.input_w,
        out.detection_meta.input_h,
        out.detection_meta.color,
        out.detection_meta.normalization
    );
    println!(
        "  POST: score={}, score_thr={:.2}, iou_thr={:.2}, nms={}, sigma={:.2}",
        out.detection_meta.score_formula,
        out.detection_meta.score_thr,
        out.detection_meta.iou_thr,
        out.detection_meta.nms_type,
        out.detection_meta.soft_sigma
    );
    println!("  STAGE: {}", stage_label(args.stage_mode));
    println!(
        "  TIMING: pre={} (resize={}, norm={}), infer={}, post={}, total={}",
        out.detect.pre_ms,
        out.detect.pre_resize_ms,
        out.detect.pre_norm_ms,
        out.detect.infer_ms,
        out.detect.post_ms,
        out.detect.pre_ms + out.detect.infer_ms + out.detect.post_ms
    );
    println!("  FACES: {}", out.detect.faces.len());
    for (i, face) in out.detect.faces.iter().enumerate() {
        println!(
            "    #{:02} score={:.4} cls={:.4} obj={:.4} bbox=[{:.2},{:.2},{:.2},{:.2}]",
            i + 1,
            face.score,
            face.cls,
            face.obj,
            face.bbox[0],
            face.bbox[1],
            face.bbox[2],
            face.bbox[3]
        );
        if let Some(kps) = &face.kps {
            let mut line = String::from("       landmarks=");
            for (j, kp) in kps.iter().enumerate() {
                if j > 0 {
                    line.push_str(", ");
                }
                line.push_str(&format!("[{:.1},{:.1}]", kp[0], kp[1]));
            }
            println!("{line}");
        }
    }

    if let Some((ms, len, count)) = out.embedding_info {
        println!(
            "  EMBEDDING: mode={:?}, count={}, len={}, time_ms={}",
            args.embedding_mode, count, len, ms
        );
    }
    if let Some((det_ms, lm_ms, kp_len)) = out.pose_info {
        println!(
            "  POSE: det_ms={}, landmarks_ms={}, keypoints={}",
            det_ms, lm_ms, kp_len
        );
    }
    println!("  PIPELINE total_ms={}", out.pipeline_total);

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let cli = parse_args(&args)?;

    if cli.json && cli.models.len() > 1 {
        bail!("--json expects a single model (use --model=yunet or --model=scrfd)");
    }

    for model in &cli.models {
        run_model(&cli, *model)?;
    }

    Ok(())
}
