use anyhow::Result;
use image::RgbImage;
use minifb::{Key, Window, WindowOptions};
use std::path::{Path, PathBuf};
use vision_core::{
    annotate_image, build_json_output, cosine, default_score_thr, detection_iou_thr,
    detection_meta, EmbeddingMode, FaceModel, ProcessConfig, Processor, ResizeMode, StageMode,
};
use vision_core::DeviceMode;

struct CliArgs {
    show_ui: bool,
    show_all: bool,
    show_facial_landmarks: bool,
    face_model: FaceModel,
    image_path: String,
    score_thr: Option<f32>,
    text_mode_json: bool,
    text_mode_silent: bool,
    stage_mode: StageMode,
    out_file: Option<String>,
    embedding_mode: EmbeddingMode,
    out_img_detections: bool,
    device_mode: DeviceMode,
    resize_mode: ResizeMode,
}

struct OutBundle {
    run_id: String,
    dir: PathBuf,
}

fn print_help() {
    println!("Uso:");
    println!("  picAnalizer [uishow] [showLandmarks] [showAll] [facemodel=1|2] [infile=path] [threshold=val] [textmode=json|0] [stages=1|11|01|111] [outfile=path|0] [embed=0|1|2] [device=cpu|gpu] [resize=fast|balanced|quality]");
    println!();
    println!("Parametros:");
    println!("  uishow                 Abre ventana con detecciones");
    println!("  showLandmarks          Dibuja landmarks faciales (alias: showFacialLandmarks)");
    println!("  showAll                Muestra info de modelos y entradas/salidas");
    println!("  facemodel=1|2          Selecciona modelo de cara: 1=YuNet, 2=SCRFD");
    println!("  infile=path            Ruta de imagen (default: people.jpg)");
    println!("  threshold=val          Umbral de score (default: 0.10 en model 1, 0.50 en model 2)");
    println!("  textmode=json          Salida JSON (default: consola)");
    println!("  textmode=0             Sin output (modo silent)");
    println!("  stages=1|11|01|111      1=Detection, 11=Detection+Embedding, 01=Embedding, 111=Detection+Embedding+Pose");
    println!("  outfile=path|0         Guarda imagen anotada (0 = no genera archivo)");
    println!("  embed=0|1|2             Embedding en JSON: 0=off, 1=qint8 (default), 2=raw");
    println!("  outDirDetections=1     Genera carpeta con in/out/json y crops de embedding");
    println!("  device=cpu|gpu         Selecciona CPU o GPU (fallback a CPU si no hay EP)");
    println!("  resize=fast|balanced|quality  Controla calidad/velocidad de resize (default: balanced)");
}

fn parse_args(args: &[String]) -> CliArgs {
    let show_ui = args
        .iter()
        .any(|a| a.eq_ignore_ascii_case("uishow") || a.eq_ignore_ascii_case("--uishow"))
        || args.len() == 1;
    let show_all = args.iter().any(|a| a == "showAll" || a == "--showAll");
    let show_facial_landmarks = args.iter().any(|a| {
        a.eq_ignore_ascii_case("showFacialLandmarks")
            || a.eq_ignore_ascii_case("--showFacialLandmarks")
            || a.eq_ignore_ascii_case("showLandmarks")
            || a.eq_ignore_ascii_case("--showLandmarks")
    }) || args.len() == 1;
    let face_model = args
        .iter()
        .find_map(|a| {
            a.strip_prefix("facemodel=")
                .or_else(|| a.strip_prefix("--facemodel="))
        })
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1);
    let face_model = if face_model == 2 {
        FaceModel::Scrfd
    } else {
        FaceModel::YuNet
    };
    let image_path = args
        .iter()
        .find_map(|a| a.strip_prefix("infile=").or_else(|| a.strip_prefix("--infile=")))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "people.jpg".to_string());
    let score_thr = args
        .iter()
        .find_map(|a| a.strip_prefix("threshold=").or_else(|| a.strip_prefix("--threshold=")))
        .and_then(|v| v.parse::<f32>().ok());
    let text_mode_json = args.iter().any(|a| a == "textmode=json" || a == "--textmode=json");
    let text_mode_silent = args.iter().any(|a| a == "textmode=0" || a == "--textmode=0");
    let stage_mode = args
        .iter()
        .find_map(|a| a.strip_prefix("stages=").or_else(|| a.strip_prefix("--stages=")))
        .map(|v| v.trim().to_string())
        .map(|v| match v.as_str() {
            "1" => StageMode::Detection,
            "11" => StageMode::DetectionEmbedding,
            "01" => StageMode::EmbeddingOnly,
            "111" => StageMode::DetectionEmbeddingPose,
            _ => StageMode::DetectionEmbedding,
        })
        .unwrap_or(StageMode::DetectionEmbedding);
    let out_file = args
        .iter()
        .find_map(|a| a.strip_prefix("outfile=").or_else(|| a.strip_prefix("--outfile=")))
        .and_then(|v| if v == "0" { None } else { Some(v.to_string()) });
    let out_img_detections =
        args.iter().any(|a| a == "outDirDetections=1" || a == "--outDirDetections=1");
    let device_mode = args
        .iter()
        .find_map(|a| a.strip_prefix("device=").or_else(|| a.strip_prefix("--device=")))
        .map(|v| v.to_ascii_lowercase())
        .map(|v| if v == "gpu" { DeviceMode::Gpu } else { DeviceMode::Cpu })
        .unwrap_or(DeviceMode::Cpu);
    let resize_mode = args
        .iter()
        .find_map(|a| a.strip_prefix("resize=").or_else(|| a.strip_prefix("--resize=")))
        .map(|v| v.to_ascii_lowercase())
        .map(|v| match v.as_str() {
            "fast" => ResizeMode::Fast,
            "quality" => ResizeMode::Quality,
            _ => ResizeMode::Balanced,
        })
        .unwrap_or(ResizeMode::Balanced);
    let embedding_mode = args
        .iter()
        .find_map(|a| a.strip_prefix("embed=").or_else(|| a.strip_prefix("--embed=")))
        .and_then(|v| v.parse::<u8>().ok())
        .map(|v| match v {
            0 => EmbeddingMode::Off,
            2 => EmbeddingMode::Raw,
            _ => EmbeddingMode::QInt8,
        })
        .unwrap_or(EmbeddingMode::QInt8);

    CliArgs {
        show_ui,
        show_all,
        show_facial_landmarks,
        face_model,
        image_path,
        score_thr,
        text_mode_json,
        text_mode_silent,
        stage_mode,
        out_file,
        embedding_mode,
        out_img_detections,
        device_mode,
        resize_mode,
    }
}

fn show_image_with_detections(
    rgb: &RgbImage,
    faces: &[vision_core::FaceDet],
    keypoints: &[[f32; 3]],
    title: &str,
    show_facial_landmarks: bool,
    iou_thr: f32,
) -> Result<()> {
    let annotated = annotate_image(rgb, faces, keypoints, show_facial_landmarks, iou_thr);

    let (w, h) = annotated.dimensions();
    let max_w = 1280u32;
    let max_h = 720u32;
    let view_w = w.min(max_w);
    let view_h = h.min(max_h);

    let mut window = Window::new(title, view_w as usize, view_h as usize, WindowOptions::default())?;
    let display = if w != view_w || h != view_h {
        image::imageops::resize(&annotated, view_w, view_h, image::imageops::Triangle)
    } else {
        annotated
    };
    let mut buffer: Vec<u32> = vec![0; (view_w * view_h) as usize];
    for y in 0..view_h {
        for x in 0..view_w {
            let p = display.get_pixel(x, y);
            let r = p[0] as u32;
            let g = p[1] as u32;
            let b = p[2] as u32;
            buffer[(y * view_w + x) as usize] = (0xFF << 24) | (r << 16) | (g << 8) | b;
        }
    }

    while window.is_open() {
        if window.is_key_down(Key::Escape) {
            break;
        }
        window.update_with_buffer(&buffer, view_w as usize, view_h as usize)?;
    }

    Ok(())
}

pub fn run_with_args(args: Vec<String>) -> Result<i32> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return Ok(0);
    }
    let mut cli = parse_args(&args);
    let out_bundle = if cli.out_img_detections {
        let run_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());
        let dir = std::env::current_dir()?.join(format!("picAnalizer-{run_id}"));
        Some(OutBundle { run_id, dir })
    } else {
        None
    };

    if cli.out_img_detections {
        cli.text_mode_json = true;
        cli.text_mode_silent = true;
        cli.embedding_mode = EmbeddingMode::Raw;
        cli.show_facial_landmarks = true;
    }

    let want_json = cli.text_mode_json || cli.out_img_detections || cli.text_mode_silent;
    let result = run(cli, out_bundle.as_ref());
    if let Err(err) = result {
        let msg = err.to_string();
        eprintln!("{msg}");
        if want_json {
            let err_json = format!(
                "{{\"result\":1,\"msg\":\"{}\"}}\n",
                vision_core::json_escape(&msg)
            );
            if let Some(bundle) = out_bundle.as_ref() {
                let _ = std::fs::create_dir_all(&bundle.dir);
                let json_path = bundle.dir.join(format!("{}.json", bundle.run_id));
                let _ = std::fs::write(json_path, &err_json);
            }
            println!("{err_json}");
        }
        return Ok(1);
    }
    Ok(0)
}

fn run(cli: CliArgs, out_bundle: Option<&OutBundle>) -> Result<()> {
    let use_json = cli.text_mode_json;
    let use_silent = cli.text_mode_silent;
    if !use_json && !use_silent {
        println!("Cargando modelos...");
    }

    let mut processor = Processor::new(cli.face_model, cli.device_mode, cli.resize_mode)?;
    if !use_json && !use_silent {
        println!("Modelos cargados.");
    }

    if cli.show_all && !use_json && !use_silent {
        for model in processor.model_summaries() {
            println!("Modelo: {}", model.name);
            println!("  Ruta: {}", model.path);
            println!("  Inputs: {}", model.inputs.len());
            for input in &model.inputs {
                println!("    - {}", input);
            }
            println!("  Outputs: {}", model.outputs.len());
            for output in &model.outputs {
                println!("    - {}", output);
            }
        }
    }

    let img_path = cli.image_path.as_str();
    let img = image::open(img_path)?;
    let rgb = img.to_rgb8();
    let img_type = image::ImageFormat::from_path(img_path)
        .ok()
        .map(|f| format!("{f:?}"))
        .unwrap_or_else(|| "Unknown".to_string());
    if !use_json && !use_silent {
        println!(
            "Imagen: {} ({}x{}, tipo {})",
            img_path,
            rgb.width(),
            rgb.height(),
            img_type
        );
    }

    if let Some(bundle) = out_bundle {
        std::fs::create_dir_all(&bundle.dir)?;
        let in_path = bundle.dir.join(format!("{}-in.jpg", bundle.run_id));
        std::fs::copy(img_path, in_path)?;
    }

    let score_thr = cli
        .score_thr
        .unwrap_or(default_score_thr(cli.face_model));
    let iou_thr = detection_iou_thr(cli.face_model);

    if !use_json && !use_silent {
        let meta = detection_meta(cli.face_model, score_thr);
        println!("DETECCION / {}", meta.model_name);
        println!("  ONNX: {}", meta.onnx_name);
        println!("  PRE:");
        println!("    - Input: {}x{}, {}", meta.input_w, meta.input_h, meta.color);
        println!("    - Normalizacion: {}", meta.normalization);
        println!("  POST:");
        println!("    - Score: {}", meta.score_formula);
        println!("    - score_thr={:.2}", meta.score_thr);
        if meta.topk_enabled {
            println!("    - TopK por nivel: enabled ({})", meta.topk);
        }
        if meta.nms_type == "soft_nms" {
            println!("    - Soft-NMS: enabled");
            println!("      * IoU={:.2}", meta.iou_thr);
            println!("      * sigma={:.2}", meta.soft_sigma);
        } else {
            println!("    - NMS: IoU={:.2}", meta.iou_thr);
        }
    }

    let config = ProcessConfig {
        score_thr,
        stage_mode: cli.stage_mode,
        embedding_mode: cli.embedding_mode,
    };
    let output = processor.process(&rgb, &config)?;
    let faces = &output.detect.faces;

    if !use_json && !use_silent {
        println!("Caras detectadas: {}", faces.len());
        for (i, face) in faces.iter().enumerate() {
            println!(
                "Cara #{}: {:.2}-{:.2}={:.2} iou={:.2}",
                i + 1,
                face.cls,
                face.obj,
                face.score,
                iou_thr
            );
        }
        let detect_total = output.detect.pre_ms + output.detect.infer_ms + output.detect.post_ms;
        println!("Tiempo deteccion:");
        println!("  pre:   {} ms", output.detect.pre_ms);
        println!("         - resize: {} ms", output.detect.pre_resize_ms);
        println!("         - norm:   {} ms", output.detect.pre_norm_ms);
        println!("  infer: {} ms", output.detect.infer_ms);
        println!("  post:  {} ms", output.detect.post_ms);
        println!("  total: {} ms", detect_total);
    }

    if !use_json && !use_silent {
        if let Some((embed_ms, emb_len, count)) = output.embedding_info {
            println!("EMBEDDING / ArcFace");
            println!("  PRE:");
            println!("    - Input: 112x112");
            println!("    - Normalizacion: (x - 127.5) / 128.0");
            println!("  Tiempo embedding: {} ms", embed_ms);
            println!("  Embeddings faciales: {} (len={})", count, emb_len);
            for (i, emb) in output.embeddings.iter().enumerate() {
                println!("    - Cara #{}: {}", i + 1, emb.len());
            }
        }
    }

    if !use_json && !use_silent {
        if let Some((pose_det_ms, pose_lm_ms, kp_len)) = output.pose_info {
            println!("POSE");
            println!("  PRE:");
            println!("    - Pose det input: 224x224, norm x/255.0");
            println!("  POST:");
            println!("    - Tiempo pose_det: {} ms", pose_det_ms);
            println!("    - Pose landmarks input: 256x256, norm x/255.0");
            println!("    - Tiempo pose_landmarks: {} ms", pose_lm_ms);
            println!("    - Keypoints: {}", kp_len);
        }
    }

    if !use_json && !use_silent {
        if let Some(first) = output.embeddings.first() {
            let sim = cosine(first, first);
            println!("Similitud facial consigo mismo: {}", sim);
        }
    }

    if !use_json && !use_silent {
        println!("Tiempo pipeline total: {} ms", output.pipeline_total);
        let detect_total = output.detect.pre_ms + output.detect.infer_ms + output.detect.post_ms;
        let gap_ms = output.pipeline_total.saturating_sub(detect_total);
        println!("Tiempo gap total (fuera deteccion): {} ms", gap_ms);
    }

    let json_out = if use_json || out_bundle.is_some() {
        Some(build_json_output(
            &output.detection_meta,
            img_path,
            rgb.width(),
            rgb.height(),
            &img_type,
            &output.detect,
            faces,
            output.embedding_info,
            &output.embeddings,
            output.pose_info,
            &cli.stage_mode,
            &cli.embedding_mode,
            output.pipeline_total,
        ))
    } else {
        None
    };
    if let Some(ref json) = json_out {
        if use_json && !use_silent {
            print!("{json}");
        }
        if let Some(bundle) = out_bundle {
            let json_path = bundle.dir.join(format!("{}.json", bundle.run_id));
            std::fs::write(json_path, json)?;
        }
    }

    if let Some(out_arg) = cli.out_file.as_deref() {
        let out_path = if out_arg.trim().is_empty() {
            let path = Path::new(img_path);
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image");
            let out_name = format!("{stem}out.jpg");
            match path.parent() {
                Some(parent) => parent.join(out_name),
                None => Path::new(&out_name).to_path_buf(),
            }
        } else {
            Path::new(out_arg).to_path_buf()
        };
        let annotated = annotate_image(&rgb, faces, &output.keypoints, cli.show_facial_landmarks, iou_thr);
        annotated.save(&out_path)?;
        if !use_json && !use_silent {
            println!("Imagen guardada: {}", out_path.display());
        }
    }

    if let Some(bundle) = out_bundle {
        let out_path = bundle.dir.join(format!("{}-out.jpg", bundle.run_id));
        let annotated = annotate_image(&rgb, faces, &output.keypoints, cli.show_facial_landmarks, iou_thr);
        annotated.save(&out_path)?;
        if matches!(
            cli.stage_mode,
            StageMode::DetectionEmbedding | StageMode::DetectionEmbeddingPose | StageMode::EmbeddingOnly
        ) {
            for (i, face) in faces.iter().enumerate() {
                let crop = crop_face(&rgb, &face.bbox);
                let crop_path = bundle.dir.join(format!("{}-{}.jpg", bundle.run_id, i + 1));
                crop.save(crop_path)?;
            }
        }
    }

    if cli.show_ui {
        let title = format!(
            "{} - {}x{} - {}",
            output.detection_meta.model_name,
            rgb.width(),
            rgb.height(),
            img_type
        );
        show_image_with_detections(
            &rgb,
            faces,
            &output.keypoints,
            &title,
            cli.show_facial_landmarks,
            iou_thr,
        )?;
    }

    Ok(())
}

fn crop_face(img: &RgbImage, bbox: &[f32; 4]) -> RgbImage {
    let (x, y, w, h) = (
        bbox[0].max(0.0) as u32,
        bbox[1].max(0.0) as u32,
        bbox[2].max(1.0) as u32,
        bbox[3].max(1.0) as u32,
    );

    let x2 = (x + w).min(img.width());
    let y2 = (y + h).min(img.height());

    image::imageops::crop_imm(img, x, y, x2 - x, y2 - y).to_image()
}
