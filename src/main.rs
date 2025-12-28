use anyhow::Result;
use fast_image_resize::{FilterType as FirFilterType, ResizeAlg};
use image::RgbImage;
use minifb::{Key, Window, WindowOptions};
use ndarray::{Array2, Array4, Axis, Ix1, Ix2};
use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

mod lib_fast_resize;
use lib_fast_resize::ResizeWorkspace;
// ---------------------------------------------------------
// MODELOS
// ---------------------------------------------------------

const YUNET_ONNX: &str = "./models/yunet_n_640_640.onnx";
const SCRFD_ONNX: &str = "./models/scrfd-det_10g.onnx";
const ARCFACE_ONNX: &str = "./models/arcface.onnx";
const POSE_DET_ONNX: &str = "./models/blazepose-pose_detection.onnx";
const POSE_LM_ONNX: &str = "./models/blazepose-pose_landmarks_detector_full.onnx";

const YUNET_INPUT_W: u32 = 640;
const YUNET_INPUT_H: u32 = 640;
const YUNET_IOU_THR: f32 = 0.45;
const YUNET_TOPK_ENABLED: bool = true;
const YUNET_TOPK: usize = 200;

const SCRFD_INPUT_W: u32 = 640;
const SCRFD_INPUT_H: u32 = 640;
const SCRFD_IOU_THR: f32 = 0.45;
const SCRFD_SOFT_NMS: bool = true;
const SCRFD_SOFT_SIGMA: f32 = 0.5;

enum FaceModel {
    YuNet,
    Scrfd,
}

struct Models {
    face_model: FaceModel,
    yunet: Option<Session>,
    scrfd: Option<Session>,
    arcface: Session,
    pose_det: Session,
    pose_landmarks: Session,
}

impl Models {
    fn load(face_model: FaceModel, device_mode: DeviceMode) -> Result<Self> {
        let _ = ort::init().with_name("vision-core").commit()?;

        let (yunet, scrfd) = match face_model {
            FaceModel::YuNet => {
                let s = build_session(YUNET_ONNX, &device_mode)?;
                (Some(s), None)
            }
            FaceModel::Scrfd => {
                let s = build_session(SCRFD_ONNX, &device_mode)?;

                (None, Some(s))
            }
        };

        let arcface = build_session(ARCFACE_ONNX, &device_mode)?;

        let pose_det = build_session(POSE_DET_ONNX, &device_mode)?;

        let pose_landmarks = build_session(POSE_LM_ONNX, &device_mode)?;

        Ok(Self {
            face_model,
            yunet,
            scrfd,
            arcface,
            pose_det,
            pose_landmarks,
        })
    }
}

fn build_session(path: &str, device_mode: &DeviceMode) -> Result<Session> {
    match device_mode {
        DeviceMode::Gpu => {
            let ep = DirectMLExecutionProvider::default().build();
            let builder = Session::builder()?
                .with_execution_providers([ep])?;
            match builder.commit_from_file(path) {
                Ok(session) => Ok(session),
                Err(_) => {
                    let cpu = CPUExecutionProvider::default().build();
                    Ok(Session::builder()?
                        .with_execution_providers([cpu])?
                        .commit_from_file(path)?)
                }
            }
        }
        DeviceMode::Cpu => {
            let cpu = CPUExecutionProvider::default().build();
            Ok(Session::builder()?
                .with_execution_providers([cpu])?
                .commit_from_file(path)?)
        }
    }
}

// ---------------------------------------------------------
// UTILS
// ---------------------------------------------------------

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-6)
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

fn extract_2d(value: &ort::value::Value) -> Result<Array2<f32>> {
    let arr = value.try_extract_array::<f32>()?;
    match arr.ndim() {
        2 => Ok(arr.into_dimensionality::<Ix2>()?.to_owned()),
        3 if arr.shape()[0] == 1 => Ok(arr.index_axis(Axis(0), 0).into_dimensionality::<Ix2>()?.to_owned()),
        _ => Err(anyhow::anyhow!(
            "ShapeError/IncompatibleShape: esperado 2D, got {:?}",
            arr.shape()
        )),
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter = inter_w * inter_h;

    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;

    if union <= 0.0 { 0.0 } else { inter / union }
}

fn nms(mut dets: Vec<FaceDet>, iou_thr: f32) -> Vec<FaceDet> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<FaceDet> = Vec::new();

    while let Some(det) = dets.pop() {
        let mut should_keep = true;
        for k in &keep {
            if iou(&det.bbox, &k.bbox) > iou_thr {
                should_keep = false;
                break;
            }
        }
        if should_keep {
            keep.push(det);
        }
    }

    keep
}

fn soft_nms(mut dets: Vec<FaceDet>, iou_thr: f32, sigma: f32, score_thr: f32) -> Vec<FaceDet> {
    let mut result = Vec::new();
    while !dets.is_empty() {
        dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let best = dets.remove(0);
        let mut remaining = Vec::new();
        for mut det in dets.into_iter() {
            let overlap = iou(&best.bbox, &det.bbox);
            if overlap > iou_thr {
                let weight = (-overlap * overlap / sigma).exp();
                det.score *= weight;
            }
            if det.score >= score_thr {
                remaining.push(det);
            }
        }
        result.push(best);
        dets = remaining;
    }
    result
}

fn draw_rect(img: &mut RgbImage, bbox: [f32; 4], color: [u8; 3]) {
    let x1 = bbox[0].max(0.0).floor() as i32;
    let y1 = bbox[1].max(0.0).floor() as i32;
    let x2 = (bbox[0] + bbox[2]).min(img.width() as f32 - 1.0).ceil() as i32;
    let y2 = (bbox[1] + bbox[3]).min(img.height() as f32 - 1.0).ceil() as i32;

    for x in x1..=x2 {
        if x >= 0 && x < img.width() as i32 {
            if y1 >= 0 && y1 < img.height() as i32 {
                img.put_pixel(x as u32, y1 as u32, image::Rgb(color));
            }
            if y2 >= 0 && y2 < img.height() as i32 {
                img.put_pixel(x as u32, y2 as u32, image::Rgb(color));
            }
        }
    }
    for y in y1..=y2 {
        if y >= 0 && y < img.height() as i32 {
            if x1 >= 0 && x1 < img.width() as i32 {
                img.put_pixel(x1 as u32, y as u32, image::Rgb(color));
            }
            if x2 >= 0 && x2 < img.width() as i32 {
                img.put_pixel(x2 as u32, y as u32, image::Rgb(color));
            }
        }
    }
}

fn draw_point(img: &mut RgbImage, x: f32, y: f32, color: [u8; 3]) {
    let xi = x.round() as i32;
    let yi = y.round() as i32;
    for dy in -3..=3 {
        for dx in -3..=3 {
            let px = xi + dx;
            let py = yi + dy;
            if px >= 0 && py >= 0 && px < img.width() as i32 && py < img.height() as i32 {
                img.put_pixel(px as u32, py as u32, image::Rgb(color));
            }
        }
    }
}

fn draw_char(img: &mut RgbImage, x: i32, y: i32, ch: char, scale: i32, color: [u8; 3]) {
    let bitmap = match ch {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        _ => [0b000, 0b000, 0b000, 0b000, 0b000],
    };

    for (row, bits) in bitmap.iter().enumerate() {
        for col in 0..3 {
            if (bits >> (2 - col)) & 1 == 1 {
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = x + col as i32 * scale + dx;
                        let py = y + row as i32 * scale + dy;
                        if px >= 0
                            && py >= 0
                            && px < img.width() as i32
                            && py < img.height() as i32
                        {
                            img.put_pixel(px as u32, py as u32, image::Rgb(color));
                        }
                    }
                }
            }
        }
    }
}

fn draw_text(img: &mut RgbImage, x: f32, y: f32, text: &str, color: [u8; 3]) {
    let scale = 2;
    let mut cursor_x = x.round() as i32;
    let cursor_y = y.round() as i32;
    for ch in text.chars() {
        draw_char(img, cursor_x, cursor_y, ch, scale, color);
        cursor_x += 4 * scale;
    }
}

fn annotate_image(
    rgb: &RgbImage,
    faces: &[FaceDet],
    keypoints: &[[f32; 3]],
    show_facial_landmarks: bool,
    iou_thr: f32,
) -> RgbImage {
    let mut annotated = rgb.clone();
    for (idx, face) in faces.iter().enumerate() {
        draw_rect(&mut annotated, face.bbox, [255, 0, 0]);
        let face_id = idx + 1;
        let label = format!(
            "#{}: {:.2}-{:.2}={:.2} iou={:.2}",
            face_id,
            face.cls,
            face.obj,
            face.score,
            iou_thr
        );
        let text_x = face.bbox[0];
        let text_y = face.bbox[1] + face.bbox[3] + 4.0;
        draw_text(&mut annotated, text_x, text_y, &label, [255, 255, 0]);
        if show_facial_landmarks {
            if let Some(kps) = &face.kps {
                for kp in kps {
                    draw_point(&mut annotated, kp[0], kp[1], [255, 255, 0]);
                }
            }
        }
    }
    for kp in keypoints {
        draw_point(&mut annotated, kp[0], kp[1], [0, 255, 0]);
    }
    annotated
}

fn show_image_with_detections(
    rgb: &RgbImage,
    faces: &[FaceDet],
    keypoints: &[[f32; 3]],
    roi: Option<[f32; 4]>,
    title: &str,
    show_facial_landmarks: bool,
    iou_thr: f32,
) -> Result<()> {
    let annotated = annotate_image(rgb, faces, keypoints, show_facial_landmarks, iou_thr);
    let _ = roi;

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

fn print_model_info(name: &str, path: &str, session: &Session) {
    println!("Modelo: {name}");
    println!("  Ruta: {path}");
    println!("  Inputs: {}", session.inputs.len());
    for input in &session.inputs {
        println!("    - {}: {:?}", input.name, input.input_type);
    }
    println!("  Outputs: {}", session.outputs.len());
    for output in &session.outputs {
        println!("    - {}: {:?}", output.name, output.output_type);
    }
}

fn print_face_model_details(model: &FaceModel, score_thr: f32, iou_thr: f32) {
    match model {
        FaceModel::YuNet => {
            println!("DETECCION / YuNet");
            println!("  ONNX: {}", YUNET_ONNX);
            println!("  PRE:");
            println!("    - Input: {}x{}, BGR", YUNET_INPUT_W, YUNET_INPUT_H);
            println!("    - Normalizacion: ninguna");
            println!("  POST:");
            println!("    - Score: cls * obj");
            println!("    - score_thr={:.2}", score_thr);
            println!("    - TopK por nivel: enabled ({})", YUNET_TOPK);
            println!("    - NMS: IoU={:.2}", iou_thr);
        }
        FaceModel::Scrfd => {
            println!("DETECCION / SCRFD");
            println!("  ONNX: {}", SCRFD_ONNX);
            println!("  PRE:");
            println!("    - Input: resize+pad a {}x{}", SCRFD_INPUT_W, SCRFD_INPUT_H);
            println!("    - Normalizacion: (x - 127.5) / 128.0");
            println!("  POST:");
            println!("    - Score: salida directa del modelo");
            println!("    - score_thr={:.2}", score_thr);
            println!("    - Soft-NMS: {}", if SCRFD_SOFT_NMS { "enabled" } else { "disabled" });
            println!("      * IoU={:.2}", iou_thr);
            println!("      * sigma={:.2}", SCRFD_SOFT_SIGMA);
        }
    }
}

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

enum EmbeddingMode {
    Off,
    Raw,
    QInt8,
}

enum DeviceMode {
    Cpu,
    Gpu,
}

enum ResizeMode {
    Fast,
    Balanced,
    Quality,
}

enum StageMode {
    Detection,
    DetectionEmbedding,
    EmbeddingOnly,
    DetectionEmbeddingPose,
}

struct OutBundle {
    run_id: String,
    dir: std::path::PathBuf,
}

struct PreprocessBuffers {
    yunet: Array4<f32>,
    scrfd: Array4<f32>,
    arcface: Array4<f32>,
    pose_det: Array4<f32>,
    pose_landmarks: Array4<f32>,
    resize: ResizeWorkspace,
    buf_640: Vec<u8>,
    buf_640_tmp: Vec<u8>,
    buf_112: Vec<u8>,
    buf_224: Vec<u8>,
    buf_256: Vec<u8>,
}

impl PreprocessBuffers {
    fn new(resize_alg: ResizeAlg) -> Self {
        Self {
            yunet: Array4::<f32>::zeros((1, 3, YUNET_INPUT_H as usize, YUNET_INPUT_W as usize)),
            scrfd: Array4::<f32>::zeros((1, 3, SCRFD_INPUT_H as usize, SCRFD_INPUT_W as usize)),
            arcface: Array4::<f32>::zeros((1, 112, 112, 3)),
            pose_det: Array4::<f32>::zeros((1, 224, 224, 3)),
            pose_landmarks: Array4::<f32>::zeros((1, 256, 256, 3)),
            resize: ResizeWorkspace::new(resize_alg),
            buf_640: vec![0; (YUNET_INPUT_W * YUNET_INPUT_H * 3) as usize],
            buf_640_tmp: vec![0; (YUNET_INPUT_W * YUNET_INPUT_H * 3) as usize],
            buf_112: vec![0; (112 * 112 * 3) as usize],
            buf_224: vec![0; (224 * 224 * 3) as usize],
            buf_256: vec![0; (256 * 256 * 3) as usize],
        }
    }
}
fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let help = args.iter().any(|a| a == "-h" || a == "--help");
    if help {
        println!("Uso:");
        println!("  picAnalizer [uiShow] [showLandmarks] [showAll] [facemodel=1|2] [infile=path] [threshold=val] [textmode=json|0] [stages=1|11|01|111] [outfile=path|0] [embed=0|1|2] [device=cpu|gpu]");
        println!();
        println!("Parametros:");
        println!("  uiShow                 Abre ventana con detecciones");
        println!("  showLandmarks          Dibuja landmarks faciales (alias: showFacialLandmarks)");
        println!("  showAll                Muestra info de modelos y entradas/salidas");
        println!("  facemodel=1|2          Selecciona modelo de cara: 1=YuNet, 2=SCRFD");
        println!("  infile=path            Ruta de imagen (default: people.jpg)");
        println!("  threshold=val          Umbral de score (default: 0.10 en model 1, 0.50 en model 2)");
        println!("  textmode=json          Salida JSON (default: consola)");
        println!("  textmode=0             Sin output (modo silent)");
        println!("  stages=1|11|01|111       1=Detection, 11=Detection+Embedding, 01=Embedding, 111=Detection+Embedding+Pose");
        println!("  outfile=path|0         Guarda imagen anotada (0 = no genera archivo)");
        println!("  embed=0|1|2             Embedding en JSON: 0=off, 1=qint8 (default), 2=raw");
        println!("  outDirDetections=1     Genera carpeta con in/out/json y crops de embedding");
        println!("  device=cpu|gpu         Selecciona CPU o GPU (fallback a CPU si no hay EP)");
        println!("  resize=fast|balanced|quality  Controla calidad/velocidad de resize (default: balanced)");
        std::process::exit(0);
    }

    let show_ui = args
        .iter()
        .any(|a| a.eq_ignore_ascii_case("uiShow") || a.eq_ignore_ascii_case("--uiShow"))
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
    let face_model = if face_model == 2 { FaceModel::Scrfd } else { FaceModel::YuNet };
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
        .and_then(|v| {
            if v == "0" {
                None
            } else {
                Some(v.to_string())
            }
        });
    let out_img_detections = args.iter().any(|a| a == "outDirDetections=1" || a == "--outDirDetections=1");
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

// ---------------------------------------------------------
// YUNET — DETECCIÓN DE CARAS
// ---------------------------------------------------------


#[derive(Clone, Copy, Debug)]
struct FaceDet {
    bbox: [f32; 4],
    score: f32,
    cls: f32,
    obj: f32,
    kps: Option<[[f32; 2]; 5]>,
}

struct DetectionMeta {
    model_name: &'static str,
    onnx_name: &'static str,
    input_w: u32,
    input_h: u32,
    color: &'static str,
    normalization: &'static str,
    score_formula: &'static str,
    score_thr: f32,
    iou_thr: f32,
    topk_enabled: bool,
    topk: usize,
    nms_type: &'static str,
    soft_sigma: f32,
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn build_json_output(
    meta: &DetectionMeta,
    img_path: &str,
    img_w: u32,
    img_h: u32,
    img_type: &str,
    detect: &DetectResult,
    faces: &[FaceDet],
    embedding: Option<(u128, usize, usize)>,
    embeddings: &[Vec<f32>],
    pose: Option<(u128, u128, usize)>,
    stage_mode: &StageMode,
    embedding_mode: &EmbeddingMode,
    pipeline_total: u128,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"image\": {\n");
    out.push_str(&format!("    \"path\": \"{}\",\n", json_escape(img_path)));
    out.push_str(&format!("    \"width\": {},\n", img_w));
    out.push_str(&format!("    \"height\": {},\n", img_h));
    out.push_str(&format!("    \"type\": \"{}\"\n", img_type));
    out.push_str("  },\n");
    out.push_str("  \"detection\": {\n");
    out.push_str("    \"model\": {\n");
    out.push_str(&format!("      \"name\": \"{}\",\n", meta.model_name));
    out.push_str(&format!("      \"onnx\": \"{}\",\n", meta.onnx_name));
    out.push_str(&format!("      \"score_thr\": {:.2},\n", meta.score_thr));
    out.push_str(&format!("      \"iou_thr\": {:.2}\n", meta.iou_thr));
    out.push_str("    },\n");
    out.push_str("    \"pre\": {\n");
    out.push_str(&format!("      \"input\": \"{}x{}\",\n", meta.input_w, meta.input_h));
    out.push_str(&format!("      \"color\": \"{}\",\n", meta.color));
    out.push_str(&format!("      \"normalization\": \"{}\"\n", meta.normalization));
    out.push_str("    },\n");
    out.push_str("    \"post\": {\n");
    out.push_str(&format!("      \"score\": \"{}\",\n", meta.score_formula));
    out.push_str(&format!("      \"topk\": {{\"enabled\":{},\"value\":{}}},\n", meta.topk_enabled, meta.topk));
    out.push_str(&format!(
        "      \"nms\": {{\"type\":\"{}\",\"iou\":{:.2},\"sigma\":{:.2}}}\n",
        meta.nms_type, meta.iou_thr, meta.soft_sigma
    ));
    out.push_str("    },\n");
    out.push_str("    \"timing_ms\": {\n");
    out.push_str(&format!("      \"pre\": {},\n", detect.pre_ms));
    out.push_str(&format!("      \"pre_resize\": {},\n", detect.pre_resize_ms));
    out.push_str(&format!("      \"pre_norm\": {},\n", detect.pre_norm_ms));
    out.push_str(&format!("      \"infer\": {},\n", detect.infer_ms));
    out.push_str(&format!("      \"post\": {},\n", detect.post_ms));
    out.push_str(&format!("      \"total\": {}\n", detect.pre_ms + detect.infer_ms + detect.post_ms));
    out.push_str("    },\n");
    out.push_str("    \"faces\": [\n");
    for (i, face) in faces.iter().enumerate() {
        let sep = if i + 1 == faces.len() { "" } else { "," };
        let mut line = String::new();
        line.push_str("      {\"id\":");
        line.push_str(&(i + 1).to_string());
        line.push_str(&format!(
            ",\"score\":{:.4},\"cls\":{:.4},\"obj\":{:.4},\"bbox\":[{:.2},{:.2},{:.2},{:.2}]",
            face.score,
            face.cls,
            face.obj,
            face.bbox[0],
            face.bbox[1],
            face.bbox[2],
            face.bbox[3]
        ));
        if let Some(kps) = &face.kps {
            line.push_str(",\"landmarks\":[");
            for (j, kp) in kps.iter().enumerate() {
                if j > 0 {
                    line.push(',');
                }
                line.push_str(&format!("[{:.2},{:.2}]", kp[0], kp[1]));
            }
            line.push(']');
        }
        line.push('}');
        line.push_str(sep);
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str("    ]\n");
    out.push_str("  },\n");
    out.push_str("  \"embedding\": {\n");
    if let Some((ms, len, count)) = embedding {
        out.push_str("    \"enabled\": true,\n");
        out.push_str("    \"input\": \"112x112\",\n");
        out.push_str("    \"normalization\": \"(x - 127.5) / 128.0\",\n");
        let encoding = match embedding_mode {
            EmbeddingMode::Off => "off",
            EmbeddingMode::Raw => "f32",
            EmbeddingMode::QInt8 => "int8-hex",
        };
        out.push_str(&format!("    \"encoding\": \"{}\",\n", encoding));
        out.push_str(&format!("    \"time_ms\": {},\n", ms));
        out.push_str(&format!("    \"count\": {},\n", count));
        out.push_str(&format!("    \"len\": {},\n", len));
        if matches!(embedding_mode, EmbeddingMode::QInt8) {
            out.push_str("    \"quant\": {\"zero_point\": 0, \"scale\": \"per-vector\"},\n");
        }
        out.push_str("    \"vectors\": [\n");
        for (i, emb) in embeddings.iter().enumerate() {
            let sep = if i + 1 == embeddings.len() { "" } else { "," };
            let mut line = String::new();
            line.push_str("      {\"id\": ");
            line.push_str(&(i + 1).to_string());
            if matches!(embedding_mode, EmbeddingMode::Raw) {
                line.push_str(", \"values\": [");
                for (j, v) in emb.iter().enumerate() {
                    if j > 0 {
                        line.push_str(", ");
                    }
                    line.push_str(&format!("{:.6}", v));
                }
                line.push_str("]}");
            } else {
                let mut max_abs = 0.0f32;
                for v in emb {
                    let a = v.abs();
                    if a > max_abs {
                        max_abs = a;
                    }
                }
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                line.push_str(", \"scale\": ");
                line.push_str(&format!("{:.8}", scale));
                line.push_str(", \"hex\": \"");
                for v in emb {
                    let mut q = (*v / scale).round();
                    if q > 127.0 {
                        q = 127.0;
                    } else if q < -128.0 {
                        q = -128.0;
                    }
                    let hex = (q as i8 as u8) as u8;
                    line.push_str(&format!("{:02X}", hex));
                }
                line.push_str("\"}");
            }
            line.push_str(sep);
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str("    ]\n");
    } else {
        out.push_str("    \"enabled\": false\n");
    }
    out.push_str("  },\n");
    out.push_str("  \"pose\": {\n");
    if let Some((det_ms, lm_ms, kp_len)) = pose {
        out.push_str("    \"enabled\": true,\n");
        out.push_str(&format!("    \"det\": {{\"input\":\"224x224\",\"normalization\":\"x/255.0\",\"time_ms\":{}}},\n", det_ms));
        out.push_str(&format!("    \"landmarks\": {{\"input\":\"256x256\",\"normalization\":\"x/255.0\",\"time_ms\":{}}},\n", lm_ms));
        out.push_str(&format!("    \"keypoints_len\": {}\n", kp_len));
    } else {
        out.push_str("    \"enabled\": false\n");
    }
    out.push_str("  },\n");
    let stage_label = match stage_mode {
        StageMode::Detection => "1",
        StageMode::DetectionEmbedding => "11",
        StageMode::EmbeddingOnly => "01",
        StageMode::DetectionEmbeddingPose => "111",
    };
    let detect_total = detect.pre_ms + detect.infer_ms + detect.post_ms;
    let gap_ms = pipeline_total.saturating_sub(detect_total);
    out.push_str(&format!(
        "  \"pipeline\": {{\"stages\": \"{}\", \"total_ms\": {}, \"gap_ms\": {}}}\n",
        stage_label, pipeline_total, gap_ms
    ));
    out.push_str("}\n");
    out
}

struct DetectResult {
    faces: Vec<FaceDet>,
    pre_ms: u128,
    pre_resize_ms: u128,
    pre_norm_ms: u128,
    infer_ms: u128,
    post_ms: u128,
}
fn detect_faces_yunet(
    session: &mut Session,
    rgb: &RgbImage,
    score_thr: f32,
    buffers: &mut PreprocessBuffers,
) -> Result<DetectResult> {
    let t_pre = std::time::Instant::now();
    let input_w = YUNET_INPUT_W;
    let input_h = YUNET_INPUT_H;
    let t_resize = std::time::Instant::now();
    let needs_resize = rgb.width() != input_w || rgb.height() != input_h;
    let (resize, buf_640, input_buf) = (&mut buffers.resize, &mut buffers.buf_640, &mut buffers.yunet);
    let raw = if needs_resize {
        resize.resize_rgb_into(rgb, input_w, input_h, buf_640)?;
        buf_640.as_slice()
    } else {
        rgb.as_raw().as_slice()
    };
    let pre_resize_ms = if needs_resize { t_resize.elapsed().as_millis() } else { 0 };

    let t_norm = std::time::Instant::now();
    let w = input_w as usize;
    for y in 0..input_h as usize {
        let row = y * w * 3;
        for x in 0..w {
            let idx = row + x * 3;
            let r = raw[idx] as f32;
            let g = raw[idx + 1] as f32;
            let b = raw[idx + 2] as f32;
            // YuNet se entreno con BGR, no RGB.
            input_buf[[0, 0, y, x]] = b;
            input_buf[[0, 1, y, x]] = g;
            input_buf[[0, 2, y, x]] = r;
        }
    }

    let input = Tensor::from_array(input_buf.clone())?;
    let pre_norm_ms = t_norm.elapsed().as_millis();
    let pre_ms = t_pre.elapsed().as_millis();

    let t_infer = std::time::Instant::now();
    let outputs = session.run(ort::inputs![input])?;
    let infer_ms = t_infer.elapsed().as_millis();

    let score_indices = [0usize, 1, 2];
    let obj_indices = [3usize, 4, 5];
    let bbox_indices = [6usize, 7, 8];
    let kps_indices = [9usize, 10, 11];
    let strides = [8u32, 16u32, 32u32];

    let mut dets = Vec::new();
    let iou_thr = YUNET_IOU_THR;
    let use_topk = YUNET_TOPK_ENABLED;
    let topk_per_level = YUNET_TOPK;

    for level in 0..strides.len() {
        let stride = strides[level];
        let scores = extract_2d(&outputs[score_indices[level]])?;
        let obj = extract_2d(&outputs[obj_indices[level]])?;
        let boxes = extract_2d(&outputs[bbox_indices[level]])?;
        let kps = extract_2d(&outputs[kps_indices[level]])?;

        if scores.ncols() < 1 || obj.ncols() < 1 || boxes.ncols() < 4 || kps.ncols() < 10 {
            return Err(anyhow::anyhow!(
                "YuNet output invalido: scores cols={}, obj cols={}, boxes cols={}, kps cols={}",
                scores.ncols(),
                obj.ncols(),
                boxes.ncols(),
                kps.ncols()
            ));
        }

        let feat_w = (input_w / stride) as usize;
        let feat_h = (input_h / stride) as usize;
        let expected = feat_w * feat_h;
        let count = scores.nrows().min(obj.nrows()).min(boxes.nrows()).min(kps.nrows());
        if count != expected {
            return Err(anyhow::anyhow!(
                "YuNet output invalido: esperado {} priors, got {}",
                expected,
                count
            ));
        }

        let mut level_dets: Vec<FaceDet> = Vec::new();
        for i in 0..count {
            let cls_score = scores[[i, 0]];
            let obj_score = obj[[i, 0]];
            let score = cls_score * obj_score;
            if score < score_thr {
                continue;
            }

            let x = (i % feat_w) as f32 * stride as f32;
            let y = (i / feat_w) as f32 * stride as f32;
            let cx = x;
            let cy = y;
            let sx = stride as f32;
            let sy = stride as f32;

            let dx = boxes[[i, 0]];
            let dy = boxes[[i, 1]];
            let dw = boxes[[i, 2]].exp();
            let dh = boxes[[i, 3]].exp();

            let bx = (dx * sx) + cx;
            let by = (dy * sy) + cy;
            let bw = dw * sx;
            let bh = dh * sy;

            let x1 = bx - bw * 0.5;
            let y1 = by - bh * 0.5;
            let x2 = bx + bw * 0.5;
            let y2 = by + bh * 0.5;

            let mut points = [[0.0f32; 2]; 5];
            for p in 0..5 {
                let kx = kps[[i, p * 2]];
                let ky = kps[[i, p * 2 + 1]];
                let px = (kx * sx) + cx;
                let py = (ky * sy) + cy;
                points[p] = [px, py];
            }

            level_dets.push(FaceDet {
                bbox: [x1, y1, x2, y2],
                score,
                cls: cls_score,
                obj: obj_score,
                kps: Some(points),
            });
        }
        if use_topk && level_dets.len() > topk_per_level {
            level_dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            level_dets.truncate(topk_per_level);
        }
        dets.extend(level_dets);
    }

    let t_post = std::time::Instant::now();
    let dets = nms(dets, iou_thr);
    let scale_x = rgb.width() as f32 / input_w as f32;
    let scale_y = rgb.height() as f32 / input_h as f32;

    let mut faces = Vec::new();
    for det in dets {
        let x1 = (det.bbox[0] * scale_x).max(0.0);
        let y1 = (det.bbox[1] * scale_y).max(0.0);
        let x2 = (det.bbox[2] * scale_x).max(0.0);
        let y2 = (det.bbox[3] * scale_y).max(0.0);
        let kps = det.kps.map(|mut pts| {
            for p in &mut pts {
                p[0] *= scale_x;
                p[1] *= scale_y;
            }
            pts
        });
        faces.push(FaceDet {
            bbox: [x1, y1, (x2 - x1).max(1.0), (y2 - y1).max(1.0)],
            score: det.score,
            cls: det.cls,
            obj: det.obj,
            kps,
        });
    }

    let post_ms = t_post.elapsed().as_millis();

    Ok(DetectResult {
        faces,
        pre_ms,
        pre_resize_ms,
        pre_norm_ms,
        infer_ms,
        post_ms,
    })
}

fn detect_faces_scrfd(
    session: &mut Session,
    rgb: &RgbImage,
    score_thr: f32,
    buffers: &mut PreprocessBuffers,
) -> Result<DetectResult> {
    let t_pre = std::time::Instant::now();
    let input_w = SCRFD_INPUT_W;
    let input_h = SCRFD_INPUT_H;
    let t_resize = std::time::Instant::now();
    let needs_resize = rgb.width() != input_w || rgb.height() != input_h;
    let (resize, buf_640, buf_640_tmp, input_buf) = (
        &mut buffers.resize,
        &mut buffers.buf_640,
        &mut buffers.buf_640_tmp,
        &mut buffers.scrfd,
    );
    let (raw, det_scale) = if needs_resize {
        let scale = resize.resize_with_pad_into(rgb, input_w, input_h, buf_640_tmp, buf_640)?;
        (buf_640.as_slice(), scale)
    } else {
        (rgb.as_raw().as_slice(), 1.0f32)
    };
    let pre_resize_ms = if needs_resize { t_resize.elapsed().as_millis() } else { 0 };

    let t_norm = std::time::Instant::now();
    let w = input_w as usize;
    for y in 0..input_h as usize {
        let row = y * w * 3;
        for x in 0..w {
            let idx = row + x * 3;
            let r = raw[idx] as f32;
            let g = raw[idx + 1] as f32;
            let b = raw[idx + 2] as f32;
            input_buf[[0, 0, y, x]] = (r - 127.5) / 128.0;
            input_buf[[0, 1, y, x]] = (g - 127.5) / 128.0;
            input_buf[[0, 2, y, x]] = (b - 127.5) / 128.0;
        }
    }

    let input = Tensor::from_array(input_buf.clone())?;
    let pre_norm_ms = t_norm.elapsed().as_millis();
    let pre_ms = t_pre.elapsed().as_millis();

    let t_infer = std::time::Instant::now();
    let outputs = session.run(ort::inputs![input])?;
    let infer_ms = t_infer.elapsed().as_millis();

    let outputs_len = outputs.len();
    let (fmc, strides, num_anchors, use_kps) = match outputs_len {
        6 => (3usize, vec![8u32, 16, 32], 2usize, false),
        9 => (3usize, vec![8u32, 16, 32], 2usize, true),
        10 => (5usize, vec![8u32, 16, 32, 64, 128], 1usize, false),
        15 => (5usize, vec![8u32, 16, 32, 64, 128], 1usize, true),
        _ => {
            return Err(anyhow::anyhow!(
                "SCRFD outputs inesperados: {}",
                outputs_len
            ))
        }
    };

    let mut dets = Vec::new();
    let iou_thr = SCRFD_IOU_THR;
    let soft_sigma = SCRFD_SOFT_SIGMA;
    let use_soft_nms = SCRFD_SOFT_NMS;

    for (level, stride) in strides.iter().enumerate() {
        let scores = extract_2d(&outputs[level])?;
        let mut bbox = extract_2d(&outputs[level + fmc])?;
        if bbox.ncols() < 4 || scores.ncols() < 1 {
            return Err(anyhow::anyhow!(
                "SCRFD output invalido: scores cols={}, bbox cols={}",
                scores.ncols(),
                bbox.ncols()
            ));
        }

        // scale distances by stride
        for v in bbox.iter_mut() {
            *v *= *stride as f32;
        }

        let kps = if use_kps {
            let mut kps = extract_2d(&outputs[level + fmc * 2])?;
            if kps.ncols() < 10 {
                return Err(anyhow::anyhow!(
                    "SCRFD kps cols invalidas: {}",
                    kps.ncols()
                ));
            }
            for v in kps.iter_mut() {
                *v *= *stride as f32;
            }
            Some(kps)
        } else {
            None
        };

        let feat_w = (input_w / *stride) as usize;
        let feat_h = (input_h / *stride) as usize;
        let expected = feat_w * feat_h * num_anchors;
        let count = scores.nrows().min(bbox.nrows());
        if count != expected {
            return Err(anyhow::anyhow!(
                "SCRFD output invalido: esperado {} priors, got {}",
                expected,
                count
            ));
        }

        for i in 0..count {
            let score = scores[[i, 0]];
            if score < score_thr {
                continue;
            }

            let grid_idx = i / num_anchors;
            let ax = (grid_idx % feat_w) as f32 * *stride as f32;
            let ay = (grid_idx / feat_w) as f32 * *stride as f32;

            let l = bbox[[i, 0]];
            let t = bbox[[i, 1]];
            let r = bbox[[i, 2]];
            let b = bbox[[i, 3]];
            let x1 = ax - l;
            let y1 = ay - t;
            let x2 = ax + r;
            let y2 = ay + b;

            let mut points = None;
            if let Some(kps_val) = &kps {
                let mut pts = [[0.0f32; 2]; 5];
                for p in 0..5 {
                    let kx = kps_val[[i, p * 2]];
                    let ky = kps_val[[i, p * 2 + 1]];
                    pts[p] = [ax + kx, ay + ky];
                }
                points = Some(pts);
            }

            dets.push(FaceDet {
                bbox: [x1, y1, x2, y2],
                score,
                cls: score,
                obj: 1.0,
                kps: points,
            });
        }
    }

    let t_post = std::time::Instant::now();
    let dets = if use_soft_nms {
        soft_nms(dets, iou_thr, soft_sigma, score_thr)
    } else {
        nms(dets, iou_thr)
    };
    let scale_x = 1.0f32 / det_scale;
    let scale_y = 1.0f32 / det_scale;

    let mut faces = Vec::new();
    for det in dets {
        let x1 = (det.bbox[0] * scale_x).max(0.0);
        let y1 = (det.bbox[1] * scale_y).max(0.0);
        let x2 = (det.bbox[2] * scale_x).max(0.0);
        let y2 = (det.bbox[3] * scale_y).max(0.0);
        let kps = det.kps.map(|mut pts| {
            for p in &mut pts {
                p[0] *= scale_x;
                p[1] *= scale_y;
            }
            pts
        });
        faces.push(FaceDet {
            bbox: [x1, y1, (x2 - x1).max(1.0), (y2 - y1).max(1.0)],
            score: det.score,
            cls: det.cls,
            obj: det.obj,
            kps,
        });
    }

    let post_ms = t_post.elapsed().as_millis();

    Ok(DetectResult {
        faces,
        pre_ms,
        pre_resize_ms,
        pre_norm_ms,
        infer_ms,
        post_ms,
    })
}

// ---------------------------------------------------------
// ARCFACE — EMBEDDING FACIAL
// ---------------------------------------------------------

fn face_embedding(
    session: &mut Session,
    face: &RgbImage,
    buffers: &mut PreprocessBuffers,
) -> Result<Vec<f32>> {
    let needs_resize = face.width() != 112 || face.height() != 112;
    let (resize, buf_112, input_buf) = (&mut buffers.resize, &mut buffers.buf_112, &mut buffers.arcface);
    let raw = if needs_resize {
        resize.resize_rgb_into(face, 112, 112, buf_112)?;
        buf_112.as_slice()
    } else {
        face.as_raw().as_slice()
    };
    for y in 0..112usize {
        let row = y * 112 * 3;
        for x in 0..112usize {
            let idx = row + x * 3;
            let r = raw[idx] as f32;
            let g = raw[idx + 1] as f32;
            let b = raw[idx + 2] as f32;
            input_buf[[0, y, x, 0]] = (r - 127.5) / 128.0;
            input_buf[[0, y, x, 1]] = (g - 127.5) / 128.0;
            input_buf[[0, y, x, 2]] = (b - 127.5) / 128.0;
        }
    }

    let input = Tensor::from_array(input_buf.clone())?;
    let outputs = session.run(ort::inputs![input])?;

    let emb = outputs[0].try_extract_array::<f32>()?;
    let emb = match emb.ndim() {
        1 => emb.into_dimensionality::<Ix1>()?,
        2 if emb.shape()[0] == 1 => emb.index_axis(Axis(0), 0).into_dimensionality::<Ix1>()?,
        _ => {
            return Err(anyhow::anyhow!(
                "ShapeError/IncompatibleShape: esperado 1D, got {:?}",
                emb.shape()
            ))
        }
    };
    Ok(emb.to_vec())
}

// ---------------------------------------------------------
// BLAZEPOSE — DETECCIÓN + LANDMARKS
// ---------------------------------------------------------

fn pose_detection(
    session: &mut Session,
    rgb: &RgbImage,
    buffers: &mut PreprocessBuffers,
) -> Result<[f32; 4]> {
    let needs_resize = rgb.width() != 224 || rgb.height() != 224;
    let (resize, buf_224, input_buf) = (&mut buffers.resize, &mut buffers.buf_224, &mut buffers.pose_det);
    let raw = if needs_resize {
        resize.resize_rgb_into(rgb, 224, 224, buf_224)?;
        buf_224.as_slice()
    } else {
        rgb.as_raw().as_slice()
    };
    for y in 0..224usize {
        let row = y * 224 * 3;
        for x in 0..224usize {
            let idx = row + x * 3;
            input_buf[[0, y, x, 0]] = raw[idx] as f32 / 255.0;
            input_buf[[0, y, x, 1]] = raw[idx + 1] as f32 / 255.0;
            input_buf[[0, y, x, 2]] = raw[idx + 2] as f32 / 255.0;
        }
    }

    let input = Tensor::from_array(input_buf.clone())?;
    let outputs = session.run(ort::inputs![input])?;

    let det = outputs[0].try_extract_array::<f32>()?;
    let det = match det.ndim() {
        1 => det.into_dimensionality::<Ix1>()?.insert_axis(Axis(0)),
        2 => det.into_dimensionality::<Ix2>()?,
        3 if det.shape()[0] == 1 => det.index_axis(Axis(0), 0).into_dimensionality::<Ix2>()?,
        _ => {
            return Err(anyhow::anyhow!(
                "ShapeError/IncompatibleShape: esperado 2D, got {:?}",
                det.shape()
            ))
        }
    };

    if det.shape()[1] < 4 {
        return Err(anyhow::anyhow!(
            "Pose detection output invalido: cols={} (se esperan >=4)",
            det.shape()[1]
        ));
    }

    let score_col = det.shape()[1] - 1;
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, row) in det.outer_iter().enumerate() {
        let score = row[score_col];
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    let row = det.row(best_idx);
    Ok([row[0], row[1], row[2], row[3]])
}

fn pose_landmarks(
    session: &mut Session,
    rgb: &RgbImage,
    _roi: [f32; 4],
    buffers: &mut PreprocessBuffers,
) -> Result<Vec<[f32; 3]>> {
    let needs_resize = rgb.width() != 256 || rgb.height() != 256;
    let (resize, buf_256, input_buf) =
        (&mut buffers.resize, &mut buffers.buf_256, &mut buffers.pose_landmarks);
    let raw = if needs_resize {
        resize.resize_rgb_into(rgb, 256, 256, buf_256)?;
        buf_256.as_slice()
    } else {
        rgb.as_raw().as_slice()
    };
    for y in 0..256usize {
        let row = y * 256 * 3;
        for x in 0..256usize {
            let idx = row + x * 3;
            input_buf[[0, y, x, 0]] = raw[idx] as f32 / 255.0;
            input_buf[[0, y, x, 1]] = raw[idx + 1] as f32 / 255.0;
            input_buf[[0, y, x, 2]] = raw[idx + 2] as f32 / 255.0;
        }
    }

    let input = Tensor::from_array(input_buf.clone())?;
    let outputs = session.run(ort::inputs![input])?;

    let kp = outputs[0].try_extract_array::<f32>()?;
    let kp = match kp.ndim() {
        2 => kp.into_dimensionality::<Ix2>()?,
        3 if kp.shape()[0] == 1 => kp.index_axis(Axis(0), 0).into_dimensionality::<Ix2>()?,
        _ => {
            return Err(anyhow::anyhow!(
                "ShapeError/IncompatibleShape: esperado 2D, got {:?}",
                kp.shape()
            ))
        }
    };

    let mut points = Vec::new();
    for row in kp.outer_iter() {
        if row.len() < 3 {
            return Err(anyhow::anyhow!(
                "Pose landmarks output invalido: columnas={} (se esperan >=3)",
                row.len()
            ));
        }
        points.push([row[0], row[1], row[2]]);
    }
    Ok(points)
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------

fn main() -> Result<()> {
    let mut cli = parse_args();
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
                json_escape(&msg)
            );
            if let Some(bundle) = out_bundle.as_ref() {
                let _ = std::fs::create_dir_all(&bundle.dir);
                let json_path = bundle.dir.join(format!("{}.json", bundle.run_id));
                let _ = std::fs::write(json_path, &err_json);
            }
            println!("{err_json}");
        }
        std::process::exit(1);
    }
    Ok(())
}

fn run(cli: CliArgs, out_bundle: Option<&OutBundle>) -> Result<()> {
    let use_json = cli.text_mode_json;
    let use_silent = cli.text_mode_silent;
    let resize_alg = match cli.resize_mode {
        ResizeMode::Fast => ResizeAlg::Nearest,
        ResizeMode::Quality => ResizeAlg::Convolution(FirFilterType::CatmullRom),
        ResizeMode::Balanced => ResizeAlg::Convolution(FirFilterType::Bilinear),
    };
    if !use_json && !use_silent {
        println!("Cargando modelos...");
    }
    let mut models = Models::load(cli.face_model, cli.device_mode)?;
    if !use_json && !use_silent {
        println!("Modelos cargados.");
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

    let default_thr = match models.face_model {
        FaceModel::YuNet => 0.10f32,
        FaceModel::Scrfd => 0.50f32,
    };
    let score_thr = cli.score_thr.unwrap_or(default_thr);
    let iou_thr = match models.face_model {
        FaceModel::YuNet => YUNET_IOU_THR,
        FaceModel::Scrfd => SCRFD_IOU_THR,
    };
    if !use_json && !use_silent {
        print_face_model_details(&models.face_model, score_thr, iou_thr);
    }
    let detection_meta = match models.face_model {
        FaceModel::YuNet => DetectionMeta {
            model_name: "YuNet",
            onnx_name: YUNET_ONNX,
            input_w: YUNET_INPUT_W,
            input_h: YUNET_INPUT_H,
            color: "BGR",
            normalization: "none",
            score_formula: "cls*obj",
            score_thr,
            iou_thr,
            topk_enabled: YUNET_TOPK_ENABLED,
            topk: YUNET_TOPK,
            nms_type: "nms",
            soft_sigma: 0.0,
        },
        FaceModel::Scrfd => DetectionMeta {
            model_name: "SCRFD",
            onnx_name: SCRFD_ONNX,
            input_w: SCRFD_INPUT_W,
            input_h: SCRFD_INPUT_H,
            color: "RGB",
            normalization: "(x - 127.5) / 128.0",
            score_formula: "score",
            score_thr,
            iou_thr,
            topk_enabled: false,
            topk: 0,
            nms_type: if SCRFD_SOFT_NMS { "soft_nms" } else { "nms" },
            soft_sigma: SCRFD_SOFT_SIGMA,
        },
    };
    let mut buffers = PreprocessBuffers::new(resize_alg);
    let pipeline_start = std::time::Instant::now();

    if cli.show_all && !use_json && !use_silent {
        match models.face_model {
            FaceModel::YuNet => {
                if let Some(ref m) = models.yunet {
                    print_model_info("YuNet", YUNET_ONNX, m);
                }
            }
            FaceModel::Scrfd => {
                if let Some(ref m) = models.scrfd {
                    print_model_info("SCRFD", SCRFD_ONNX, m);
                }
            }
        }
        print_model_info("ArcFace", ARCFACE_ONNX, &models.arcface);
        print_model_info("Pose detection", POSE_DET_ONNX, &models.pose_det);
        print_model_info(
            "Pose landmarks",
            POSE_LM_ONNX,
            &models.pose_landmarks,
        );
    }

    // 1) CARAS
    let detect = match cli.stage_mode {
        StageMode::EmbeddingOnly => {
            if rgb.width() != 112 || rgb.height() != 112 {
                return Err(anyhow::anyhow!(
                    "Embedding-only requiere imagen 112x112, got {}x{}",
                    rgb.width(),
                    rgb.height()
                ));
            }
            DetectResult {
                faces: vec![FaceDet {
                    bbox: [0.0, 0.0, rgb.width() as f32, rgb.height() as f32],
                    score: 1.0,
                    cls: 1.0,
                    obj: 1.0,
                    kps: None,
                }],
                pre_ms: 0,
                pre_resize_ms: 0,
                pre_norm_ms: 0,
                infer_ms: 0,
                post_ms: 0,
            }
        }
        _ => match models.face_model {
            FaceModel::YuNet => {
                let s = models.yunet.as_mut().ok_or_else(|| anyhow::anyhow!("YuNet no cargado"))?;
                detect_faces_yunet(s, &rgb, score_thr, &mut buffers)?
            }
            FaceModel::Scrfd => {
                let s = models.scrfd.as_mut().ok_or_else(|| anyhow::anyhow!("SCRFD no cargado"))?;
                detect_faces_scrfd(s, &rgb, score_thr, &mut buffers)?
            }
        },
    };
    let faces = &detect.faces;
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
        let detect_total = detect.pre_ms + detect.infer_ms + detect.post_ms;
        println!("Tiempo deteccion:");
        println!("  pre:   {} ms", detect.pre_ms);
        println!("         - resize: {} ms", detect.pre_resize_ms);
        println!("         - norm:   {} ms", detect.pre_norm_ms);
        println!("  infer: {} ms", detect.infer_ms);
        println!("  post:  {} ms", detect.post_ms);
        println!("  total: {} ms", detect_total);
    }

    let mut keypoints: Vec<[f32; 3]> = Vec::new();
    let mut roi_opt: Option<[f32; 4]> = None;
    let mut embedding_info: Option<(u128, usize, usize)> = None;
    let mut pose_info: Option<(u128, u128, usize)> = None;

    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    if matches!(cli.stage_mode, StageMode::DetectionEmbedding | StageMode::DetectionEmbeddingPose | StageMode::EmbeddingOnly)
        && !matches!(cli.embedding_mode, EmbeddingMode::Off)
    {
        if !faces.is_empty() {
            if !use_json && !use_silent {
                println!("EMBEDDING / ArcFace");
                println!("  PRE:");
                println!("    - Input: 112x112");
                println!("    - Normalizacion: (x - 127.5) / 128.0");
            }
            let t_embed = std::time::Instant::now();
            for face in faces {
                let crop = crop_face(&rgb, &face.bbox);
                let emb = face_embedding(&mut models.arcface, &crop, &mut buffers)?;
                embeddings.push(emb);
            }
            let embed_ms = t_embed.elapsed().as_millis();
            let emb_len = embeddings.first().map(|e| e.len()).unwrap_or(0);
            embedding_info = Some((embed_ms, emb_len, embeddings.len()));
            if !use_json && !use_silent {
                println!("  Tiempo embedding: {} ms", embed_ms);
                println!("  Embeddings faciales: {} (len={})", embeddings.len(), emb_len);
                for (i, emb) in embeddings.iter().enumerate() {
                    println!("    - Cara #{}: {}", i + 1, emb.len());
                }
            }
        }
    }

    if matches!(cli.stage_mode, StageMode::DetectionEmbeddingPose) {
        if !use_json && !use_silent {
            println!("POSE");
            println!("  PRE:");
            println!("    - Pose det input: 224x224, norm x/255.0");
        }
        let t_pose_det = std::time::Instant::now();
        let roi = pose_detection(&mut models.pose_det, &rgb, &mut buffers)?;
        let roi_scale_x = rgb.width() as f32 / 224.0;
        let roi_scale_y = rgb.height() as f32 / 224.0;
        roi_opt = Some([
            roi[0] * roi_scale_x,
            roi[1] * roi_scale_y,
            roi[2] * roi_scale_x,
            roi[3] * roi_scale_y,
        ]);
        let pose_det_ms = t_pose_det.elapsed().as_millis();
        if !use_json && !use_silent {
            println!(
                "    - Rescaling ROI: scale_x={:.4}, scale_y={:.4}",
                roi_scale_x, roi_scale_y
            );
            println!("  POST:");
            println!("    - ROI pose: {:?}", roi);
            println!("    - Tiempo pose_det: {} ms", pose_det_ms);
        }

        if !use_json && !use_silent {
            println!("    - Pose landmarks input: 256x256, norm x/255.0");
        }
        let t_pose_lm = std::time::Instant::now();
        keypoints = pose_landmarks(&mut models.pose_landmarks, &rgb, roi, &mut buffers)?;
        let kp_scale_x = rgb.width() as f32 / 256.0;
        let kp_scale_y = rgb.height() as f32 / 256.0;
        for kp in &mut keypoints {
            kp[0] *= kp_scale_x;
            kp[1] *= kp_scale_y;
        }
        let pose_lm_ms = t_pose_lm.elapsed().as_millis();
        pose_info = Some((pose_det_ms, pose_lm_ms, keypoints.len()));
        if !use_json && !use_silent {
            println!(
                "    - Rescaling keypoints: scale_x={:.4}, scale_y={:.4}",
                kp_scale_x, kp_scale_y
            );
            println!("    - Tiempo pose_landmarks: {} ms", pose_lm_ms);
            println!("    - Keypoints: {}", keypoints.len());
        }
    }

    if !use_json && !use_silent {
        if let Some(first) = embeddings.first() {
            let sim = cosine(first, first);
            println!("Similitud facial consigo mismo: {}", sim);
        }
    }

    let pipeline_total = pipeline_start.elapsed().as_millis();
    if !use_json && !use_silent {
        println!("Tiempo pipeline total: {} ms", pipeline_total);
        let detect_total = detect.pre_ms + detect.infer_ms + detect.post_ms;
        let gap_ms = pipeline_total.saturating_sub(detect_total);
        println!("Tiempo gap total (fuera deteccion): {} ms", gap_ms);
    }

    let json_out = if use_json || out_bundle.is_some() {
        Some(build_json_output(
            &detection_meta,
            img_path,
            rgb.width(),
            rgb.height(),
            &img_type,
            &detect,
            faces,
            embedding_info,
            &embeddings,
            pose_info,
            &cli.stage_mode,
            &cli.embedding_mode,
            pipeline_total,
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
        let annotated = annotate_image(&rgb, faces, &keypoints, cli.show_facial_landmarks, iou_thr);
        annotated.save(&out_path)?;
        if !use_json && !use_silent {
            println!("Imagen guardada: {}", out_path.display());
        }
    }

    if let Some(bundle) = out_bundle {
        let out_path = bundle.dir.join(format!("{}-out.jpg", bundle.run_id));
        let annotated = annotate_image(&rgb, faces, &keypoints, cli.show_facial_landmarks, iou_thr);
        annotated.save(&out_path)?;
        if matches!(cli.stage_mode, StageMode::DetectionEmbedding | StageMode::DetectionEmbeddingPose | StageMode::EmbeddingOnly) {
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
            detection_meta.model_name,
            rgb.width(),
            rgb.height(),
            img_type
        );
        show_image_with_detections(
            &rgb,
            &faces,
            &keypoints,
            roi_opt,
            &title,
            cli.show_facial_landmarks,
            iou_thr,
        )?;
    }

    Ok(())
}
