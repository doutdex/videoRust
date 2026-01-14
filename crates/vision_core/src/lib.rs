use anyhow::Result;
use fast_image_resize::{FilterType as FirFilterType, ResizeAlg};
use image::RgbImage;
use ndarray::{Array2, Array4, Axis, Ix1, Ix2};
use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
use ort::session::Session;
use ort::value::Tensor;

mod lib_fast_resize;
use lib_fast_resize::ResizeWorkspace;
// ---------------------------------------------------------
// MODELOS
// ---------------------------------------------------------

pub const YUNET_ONNX: &str = "./models/yunet_n_640_640.onnx";
pub const SCRFD_ONNX: &str = "./models/scrfd-det_10g.onnx";
pub const ARCFACE_ONNX: &str = "./models/arcface.onnx";
pub const POSE_DET_ONNX: &str = "./models/blazepose-pose_detection.onnx";
pub const POSE_LM_ONNX: &str = "./models/blazepose-pose_landmarks_detector_full.onnx";

pub const YUNET_INPUT_W: u32 = 640;
pub const YUNET_INPUT_H: u32 = 640;
pub const YUNET_IOU_THR: f32 = 0.45;
pub const YUNET_TOPK_ENABLED: bool = true;
pub const YUNET_TOPK: usize = 200;

pub const SCRFD_INPUT_W: u32 = 640;
pub const SCRFD_INPUT_H: u32 = 640;
pub const SCRFD_IOU_THR: f32 = 0.45;
pub const SCRFD_SOFT_NMS: bool = true;
pub const SCRFD_SOFT_SIGMA: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub enum FaceModel {
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

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
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

pub fn annotate_image(
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

#[derive(Clone, Copy, Debug)]
pub enum EmbeddingMode {
    Off,
    Raw,
    QInt8,
}

#[derive(Clone, Copy, Debug)]
pub enum DeviceMode {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug)]
pub enum ResizeMode {
    Fast,
    Balanced,
    Quality,
}

#[derive(Clone, Copy, Debug)]
pub enum StageMode {
    Detection,
    DetectionEmbedding,
    EmbeddingOnly,
    DetectionEmbeddingPose,
}

#[derive(Clone, Copy, Debug)]
pub struct ProcessConfig {
    pub score_thr: f32,
    pub stage_mode: StageMode,
    pub embedding_mode: EmbeddingMode,
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

pub struct Processor {
    models: Models,
    buffers: PreprocessBuffers,
}

#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub name: &'static str,
    pub path: &'static str,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub detection_meta: DetectionMeta,
    pub detect: DetectResult,
    pub embeddings: Vec<Vec<f32>>,
    pub embedding_info: Option<(u128, usize, usize)>,
    pub keypoints: Vec<[f32; 3]>,
    pub roi: Option<[f32; 4]>,
    pub pose_info: Option<(u128, u128, usize)>,
    pub pipeline_total: u128,
}

pub fn default_score_thr(face_model: FaceModel) -> f32 {
    match face_model {
        FaceModel::YuNet => 0.10,
        FaceModel::Scrfd => 0.50,
    }
}

pub fn detection_iou_thr(face_model: FaceModel) -> f32 {
    match face_model {
        FaceModel::YuNet => YUNET_IOU_THR,
        FaceModel::Scrfd => SCRFD_IOU_THR,
    }
}

pub fn detection_meta(face_model: FaceModel, score_thr: f32) -> DetectionMeta {
    match face_model {
        FaceModel::YuNet => DetectionMeta {
            model_name: "YuNet",
            onnx_name: YUNET_ONNX,
            input_w: YUNET_INPUT_W,
            input_h: YUNET_INPUT_H,
            color: "BGR",
            normalization: "none",
            score_formula: "cls*obj",
            score_thr,
            iou_thr: YUNET_IOU_THR,
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
            iou_thr: SCRFD_IOU_THR,
            topk_enabled: false,
            topk: 0,
            nms_type: if SCRFD_SOFT_NMS { "soft_nms" } else { "nms" },
            soft_sigma: SCRFD_SOFT_SIGMA,
        },
    }
}

impl Processor {
    pub fn new(face_model: FaceModel, device_mode: DeviceMode, resize_mode: ResizeMode) -> Result<Self> {
        let models = Models::load(face_model, device_mode)?;
        let resize_alg = match resize_mode {
            ResizeMode::Fast => ResizeAlg::Nearest,
            ResizeMode::Quality => ResizeAlg::Convolution(FirFilterType::CatmullRom),
            ResizeMode::Balanced => ResizeAlg::Convolution(FirFilterType::Bilinear),
        };
        let buffers = PreprocessBuffers::new(resize_alg);
        Ok(Self { models, buffers })
    }

    pub fn face_model(&self) -> FaceModel {
        self.models.face_model
    }

    pub fn model_summaries(&self) -> Vec<ModelSummary> {
        let mut summaries = Vec::new();
        match self.models.face_model {
            FaceModel::YuNet => {
                if let Some(ref m) = self.models.yunet {
                    summaries.push(build_summary("YuNet", YUNET_ONNX, m));
                }
            }
            FaceModel::Scrfd => {
                if let Some(ref m) = self.models.scrfd {
                    summaries.push(build_summary("SCRFD", SCRFD_ONNX, m));
                }
            }
        }
        summaries.push(build_summary("ArcFace", ARCFACE_ONNX, &self.models.arcface));
        summaries.push(build_summary("Pose detection", POSE_DET_ONNX, &self.models.pose_det));
        summaries.push(build_summary("Pose landmarks", POSE_LM_ONNX, &self.models.pose_landmarks));
        summaries
    }

    pub fn process(&mut self, rgb: &RgbImage, config: &ProcessConfig) -> Result<PipelineOutput> {
        let pipeline_start = std::time::Instant::now();
        let detection_meta = detection_meta(self.models.face_model, config.score_thr);

        let detect = match config.stage_mode {
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
            _ => match self.models.face_model {
                FaceModel::YuNet => {
                    let s = self
                        .models
                        .yunet
                        .as_mut()
                        .ok_or_else(|| anyhow::anyhow!("YuNet no cargado"))?;
                    detect_faces_yunet(s, rgb, config.score_thr, &mut self.buffers)?
                }
                FaceModel::Scrfd => {
                    let s = self
                        .models
                        .scrfd
                        .as_mut()
                        .ok_or_else(|| anyhow::anyhow!("SCRFD no cargado"))?;
                    detect_faces_scrfd(s, rgb, config.score_thr, &mut self.buffers)?
                }
            },
        };

        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        let mut embedding_info: Option<(u128, usize, usize)> = None;
        if matches!(
            config.stage_mode,
            StageMode::DetectionEmbedding | StageMode::DetectionEmbeddingPose | StageMode::EmbeddingOnly
        ) && !matches!(config.embedding_mode, EmbeddingMode::Off)
        {
            if !detect.faces.is_empty() {
                let t_embed = std::time::Instant::now();
                for face in &detect.faces {
                    let crop = crop_face(rgb, &face.bbox);
                    let emb = face_embedding(&mut self.models.arcface, &crop, &mut self.buffers)?;
                    embeddings.push(emb);
                }
                let embed_ms = t_embed.elapsed().as_millis();
                let emb_len = embeddings.first().map(|e| e.len()).unwrap_or(0);
                embedding_info = Some((embed_ms, emb_len, embeddings.len()));
            }
        }

        let mut keypoints: Vec<[f32; 3]> = Vec::new();
        let mut roi_opt: Option<[f32; 4]> = None;
        let mut pose_info: Option<(u128, u128, usize)> = None;

        if matches!(config.stage_mode, StageMode::DetectionEmbeddingPose) {
            let t_pose_det = std::time::Instant::now();
            let roi = pose_detection(&mut self.models.pose_det, rgb, &mut self.buffers)?;
            let roi_scale_x = rgb.width() as f32 / 224.0;
            let roi_scale_y = rgb.height() as f32 / 224.0;
            roi_opt = Some([
                roi[0] * roi_scale_x,
                roi[1] * roi_scale_y,
                roi[2] * roi_scale_x,
                roi[3] * roi_scale_y,
            ]);
            let pose_det_ms = t_pose_det.elapsed().as_millis();

            let t_pose_lm = std::time::Instant::now();
            keypoints = pose_landmarks(&mut self.models.pose_landmarks, rgb, roi, &mut self.buffers)?;
            let kp_scale_x = rgb.width() as f32 / 256.0;
            let kp_scale_y = rgb.height() as f32 / 256.0;
            for kp in &mut keypoints {
                kp[0] *= kp_scale_x;
                kp[1] *= kp_scale_y;
            }
            let pose_lm_ms = t_pose_lm.elapsed().as_millis();
            pose_info = Some((pose_det_ms, pose_lm_ms, keypoints.len()));
        }

        let pipeline_total = pipeline_start.elapsed().as_millis();
        Ok(PipelineOutput {
            detection_meta,
            detect,
            embeddings,
            embedding_info,
            keypoints,
            roi: roi_opt,
            pose_info,
            pipeline_total,
        })
    }
}

fn build_summary(name: &'static str, path: &'static str, session: &Session) -> ModelSummary {
    let inputs = session
        .inputs
        .iter()
        .map(|input| format!("{}: {:?}", input.name, input.input_type))
        .collect();
    let outputs = session
        .outputs
        .iter()
        .map(|output| format!("{}: {:?}", output.name, output.output_type))
        .collect();
    ModelSummary {
        name,
        path,
        inputs,
        outputs,
    }
}

// ---------------------------------------------------------
// YUNET — DETECCIÓN DE CARAS
// ---------------------------------------------------------


#[derive(Clone, Copy, Debug)]
pub struct FaceDet {
    pub bbox: [f32; 4],
    pub score: f32,
    pub cls: f32,
    pub obj: f32,
    pub kps: Option<[[f32; 2]; 5]>,
}

#[derive(Clone, Copy, Debug)]
pub struct DetectionMeta {
    pub model_name: &'static str,
    pub onnx_name: &'static str,
    pub input_w: u32,
    pub input_h: u32,
    pub color: &'static str,
    pub normalization: &'static str,
    pub score_formula: &'static str,
    pub score_thr: f32,
    pub iou_thr: f32,
    pub topk_enabled: bool,
    pub topk: usize,
    pub nms_type: &'static str,
    pub soft_sigma: f32,
}

pub fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn build_json_output(
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

#[derive(Clone, Debug)]
pub struct DetectResult {
    pub faces: Vec<FaceDet>,
    pub pre_ms: u128,
    pub pre_resize_ms: u128,
    pub pre_norm_ms: u128,
    pub infer_ms: u128,
    pub post_ms: u128,
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


