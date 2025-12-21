use anyhow::Result;
use image::RgbImage;
use minifb::{Key, Window, WindowOptions};
use ndarray::{Array2, Array4, Axis, Ix1, Ix2};
use ort::session::Session;
use ort::value::Tensor;

// ---------------------------------------------------------
// MODELOS
// ---------------------------------------------------------

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
    fn load(face_model: FaceModel) -> Result<Self> {
        let _ = ort::init().with_name("vision-core").commit()?;

        let (yunet, scrfd) = match face_model {
            FaceModel::YuNet => {
                let s = Session::builder()?
                    .commit_from_file("./models/yunet_n_640_640.onnx")?;
                (Some(s), None)
            }
            FaceModel::Scrfd => {
                let s = Session::builder()?
                   // .commit_from_file("./models/scrfd-det_2.5g.onnx")?;
                    .commit_from_file("./models/scrfd-det_10g.onnx")?;

                (None, Some(s))
            }
        };

        let arcface = Session::builder()?
            .commit_from_file("./models/arcface.onnx")?;

        let pose_det = Session::builder()?
            .commit_from_file("./models/blazepose-pose_detection.onnx")?;

        let pose_landmarks = Session::builder()?
            .commit_from_file("./models/blazepose-pose_landmarks_detector_full.onnx")?;

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

fn resize_with_pad(img: &RgbImage, target_w: u32, target_h: u32) -> (RgbImage, f32) {
    let img_w = img.width();
    let img_h = img.height();
    if img_w == target_w && img_h == target_h {
        return (img.clone(), 1.0);
    }
    let im_ratio = img_h as f32 / img_w as f32;
    let model_ratio = target_h as f32 / target_w as f32;

    let (new_w, new_h) = if im_ratio > model_ratio {
        let new_h = target_h;
        let new_w = (new_h as f32 / im_ratio).round().max(1.0) as u32;
        (new_w, new_h)
    } else {
        let new_w = target_w;
        let new_h = (new_w as f32 * im_ratio).round().max(1.0) as u32;
        (new_w, new_h)
    };

    let det_scale = new_h as f32 / img_h as f32;
    let resized = image::imageops::resize(img, new_w, new_h, image::imageops::Triangle);
    let mut det_img = RgbImage::new(target_w, target_h);
    image::imageops::replace(&mut det_img, &resized, 0, 0);

    (det_img, det_scale)
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

fn show_image_with_detections(
    rgb: &RgbImage,
    faces: &[FaceDet],
    keypoints: &[[f32; 3]],
    roi: Option<[f32; 4]>,
    title: &str,
    show_facial_landmarks: bool,
    iou_thr: f32,
) -> Result<()> {
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
    let _ = roi;
    for kp in keypoints {
        draw_point(&mut annotated, kp[0], kp[1], [0, 255, 0]);
    }

    let (w, h) = annotated.dimensions();
    let max_w = 1920u32;
    let max_h = 1080u32;
    let view_w = max_w;
    let view_h = max_h;

    let mut window = Window::new(title, view_w as usize, view_h as usize, WindowOptions::default())?;
    let mut buffer: Vec<u32> = vec![0; (view_w * view_h) as usize];
    let mut offset_x: i32 = 0;
    let mut offset_y: i32 = 0;

    while window.is_open() {
        let step = if window.is_key_down(Key::LeftShift) { 50 } else { 10 };
        let max_offset_x = (w as i32 - view_w as i32).max(0);
        let max_offset_y = (h as i32 - view_h as i32).max(0);
        if window.is_key_down(Key::Left) || window.is_key_down(Key::A) {
            offset_x = (offset_x - step).max(0);
        }
        if window.is_key_down(Key::Right) || window.is_key_down(Key::D) {
            offset_x = (offset_x + step).min(max_offset_x);
        }
        if window.is_key_down(Key::Up) || window.is_key_down(Key::W) {
            offset_y = (offset_y - step).max(0);
        }
        if window.is_key_down(Key::Down) || window.is_key_down(Key::S) {
            offset_y = (offset_y + step).min(max_offset_y);
        }
        if window.is_key_down(Key::Escape) {
            break;
        }

        for y in 0..view_h {
            for x in 0..view_w {
                let sx = (x as i32 + offset_x) as u32;
                let sy = (y as i32 + offset_y) as u32;
                let pixel = if sx < w && sy < h {
                    let p = annotated.get_pixel(sx, sy);
                    let r = p[0] as u32;
                    let g = p[1] as u32;
                    let b = p[2] as u32;
                    (0xFF << 24) | (r << 16) | (g << 8) | b
                } else {
                    0xFF000000
                };
                buffer[(y * view_w + x) as usize] = pixel;
            }
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

fn print_face_model_details(model: &FaceModel) {
    match model {
        FaceModel::YuNet => {
    println!("DETECCION / YuNet");
    println!("  ONNX: yunet_n_640_640.onnx");
    println!("  PRE:");
    println!("    - Input: 640x640, BGR");
    println!("    - Normalizacion: ninguna");
    println!("  POST:");
    println!("    - Score: cls * obj");
    println!("    - score_thr=0.02");
    println!("    - TopK por nivel: enabled (200)");
    println!("    - NMS: IoU=0.45");
        }
        FaceModel::Scrfd => {
    println!("DETECCION / SCRFD");
    println!("  ONNX: scrfd-det_2.5g.onnx");
    println!("  PRE:");
    println!("    - Input: resize+pad a 640x640");
    println!("    - Normalizacion: (x - 127.5) / 128.0");
    println!("  POST:");
    println!("    - Score: salida directa del modelo");
    println!("    - score_thr=0.5");
    println!("    - Soft-NMS: enabled");
    println!("      * IoU=0.45");
    println!("      * sigma=0.5");
        }
    }
}

struct CliArgs {
    show_ui: bool,
    show_all: bool,
    show_facial_landmarks: bool,
    face_model: FaceModel,
    image_path: String,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let help = args.iter().any(|a| a == "-h" || a == "--help");
    if help {
        println!("Uso:");
        println!("  picAnalizer [showUI] [showLandmarks] [showAll] [facemodel=1|2] [img=path]");
        println!();
        println!("Parametros:");
        println!("  showUI                 Abre ventana con detecciones");
        println!("  showLandmarks          Dibuja landmarks faciales (alias: showFacialLandmarks)");
        println!("  showAll                Muestra info de modelos y entradas/salidas");
        println!("  facemodel=1|2          1=YuNet, 2=SCRFD");
        println!("  img=path               Ruta de imagen (default: people1.jpg)");
        std::process::exit(0);
    }

    let show_ui = args.iter().any(|a| a == "showUI" || a == "--showUI");
    let show_all = args.iter().any(|a| a == "showAll" || a == "--showAll");
    let show_facial_landmarks = args.iter().any(|a| {
        a == "showFacialLandmarks"
            || a == "--showFacialLandmarks"
            || a == "showLandmarks"
            || a == "--showLandmarks"
    });
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
        .find_map(|a| a.strip_prefix("img=").or_else(|| a.strip_prefix("--img=")))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "people1.jpg".to_string());

    CliArgs {
        show_ui,
        show_all,
        show_facial_landmarks,
        face_model,
        image_path,
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

struct DetectResult {
    faces: Vec<FaceDet>,
    pre_ms: u128,
    pre_resize_ms: u128,
    pre_norm_ms: u128,
    infer_ms: u128,
    post_ms: u128,
}
fn detect_faces_yunet(session: &mut Session, rgb: &RgbImage) -> Result<DetectResult> {
    let t_pre = std::time::Instant::now();
    let input_w = 640u32;
    let input_h = 640u32;
    let t_resize = std::time::Instant::now();
    let resized = if rgb.width() == input_w && rgb.height() == input_h {
        rgb.clone()
    } else {
        image::imageops::resize(rgb, input_w, input_h, image::imageops::Triangle)
    };
    let pre_resize_ms = t_resize.elapsed().as_millis();

    let t_norm = std::time::Instant::now();
    let mut input = Array4::<f32>::zeros((1, 3, input_h as usize, input_w as usize));
    for y in 0..input_h {
        for x in 0..input_w {
            let p = resized.get_pixel(x as u32, y as u32);
            // YuNet se entreno con BGR, no RGB.
            input[[0, 0, y as usize, x as usize]] = p[2] as f32;
            input[[0, 1, y as usize, x as usize]] = p[1] as f32;
            input[[0, 2, y as usize, x as usize]] = p[0] as f32;
        }
    }

    let input = Tensor::from_array(input)?;
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
    let score_thr = 0.02f32;
    let iou_thr = 0.45f32;
    let use_topk = true;
    let topk_per_level = 200usize;

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

fn detect_faces_scrfd(session: &mut Session, rgb: &RgbImage) -> Result<DetectResult> {
    let t_pre = std::time::Instant::now();
    let input_w = 640u32;
    let input_h = 640u32;
    let t_resize = std::time::Instant::now();
    let (resized, det_scale) = resize_with_pad(rgb, input_w, input_h);
    let pre_resize_ms = t_resize.elapsed().as_millis();

    let t_norm = std::time::Instant::now();
    let mut input = Array4::<f32>::zeros((1, 3, input_h as usize, input_w as usize));
    for y in 0..input_h {
        for x in 0..input_w {
            let p = resized.get_pixel(x as u32, y as u32);
            input[[0, 0, y as usize, x as usize]] = (p[0] as f32 - 127.5) / 128.0;
            input[[0, 1, y as usize, x as usize]] = (p[1] as f32 - 127.5) / 128.0;
            input[[0, 2, y as usize, x as usize]] = (p[2] as f32 - 127.5) / 128.0;
        }
    }

    let input = Tensor::from_array(input)?;
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
    let score_thr = 0.5f32;
    let iou_thr = 0.45f32;
    let soft_sigma = 0.5f32;
    let use_soft_nms = true;

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
    let scale_x = 1.0 / det_scale;
    let scale_y = 1.0 / det_scale;

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

fn face_embedding(session: &mut Session, face: &RgbImage) -> Result<Vec<f32>> {
    let resized = image::imageops::resize(face, 112, 112, image::imageops::Lanczos3);

    let mut input = Array4::<f32>::zeros((1, 112, 112, 3));
    for y in 0..112 {
        for x in 0..112 {
            let p = resized.get_pixel(x as u32, y as u32);
            input[[0, y, x, 0]] = (p[0] as f32 - 127.5) / 128.0;
            input[[0, y, x, 1]] = (p[1] as f32 - 127.5) / 128.0;
            input[[0, y, x, 2]] = (p[2] as f32 - 127.5) / 128.0;
        }
    }

    let input = Tensor::from_array(input)?;
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

fn pose_detection(session: &mut Session, rgb: &RgbImage) -> Result<[f32; 4]> {
    let resized = image::imageops::resize(rgb, 224, 224, image::imageops::Triangle);

    let mut input = Array4::<f32>::zeros((1, 224, 224, 3));
    for y in 0..224 {
        for x in 0..224 {
            let p = resized.get_pixel(x as u32, y as u32);
            input[[0, y, x, 0]] = p[0] as f32 / 255.0;
            input[[0, y, x, 1]] = p[1] as f32 / 255.0;
            input[[0, y, x, 2]] = p[2] as f32 / 255.0;
        }
    }

    let input = Tensor::from_array(input)?;
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

fn pose_landmarks(session: &mut Session, rgb: &RgbImage, _roi: [f32; 4]) -> Result<Vec<[f32; 3]>> {
    let resized = image::imageops::resize(rgb, 256, 256, image::imageops::Triangle);

    let mut input = Array4::<f32>::zeros((1, 256, 256, 3));
    for y in 0..256 {
        for x in 0..256 {
            let p = resized.get_pixel(x as u32, y as u32);
            input[[0, y, x, 0]] = p[0] as f32 / 255.0;
            input[[0, y, x, 1]] = p[1] as f32 / 255.0;
            input[[0, y, x, 2]] = p[2] as f32 / 255.0;
        }
    }

    let input = Tensor::from_array(input)?;
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
    let cli = parse_args();

    println!("Cargando modelos...");
    let mut models = Models::load(cli.face_model)?;
    println!("Modelos cargados.");

    let img_path = cli.image_path.as_str();
    let img = image::open(img_path)?;
    let rgb = img.to_rgb8();
    let img_type = image::ImageFormat::from_path(img_path)
        .ok()
        .map(|f| format!("{f:?}"))
        .unwrap_or_else(|| "Unknown".to_string());
    println!(
        "Imagen: {} ({}x{}, tipo {})",
        img_path,
        rgb.width(),
        rgb.height(),
        img_type
    );
    print_face_model_details(&models.face_model);
    let pipeline_start = std::time::Instant::now();

    if cli.show_all {
        match models.face_model {
            FaceModel::YuNet => {
                if let Some(ref m) = models.yunet {
                    print_model_info("YuNet", "./models/yunet_n_640_640.onnx", m);
                }
            }
            FaceModel::Scrfd => {
                if let Some(ref m) = models.scrfd {
                    print_model_info("SCRFD", "./models/scrfd-det_2.5g.onnx", m);
                }
            }
        }
        print_model_info("ArcFace", "./models/arcface.onnx", &models.arcface);
        print_model_info("Pose detection", "./models/blazepose-pose_detection.onnx", &models.pose_det);
        print_model_info(
            "Pose landmarks",
            "./models/blazepose-pose_landmarks_detector_full.onnx",
            &models.pose_landmarks,
        );
    }

    // 1) CARAS
    let detect = match models.face_model {
        FaceModel::YuNet => {
            let s = models.yunet.as_mut().ok_or_else(|| anyhow::anyhow!("YuNet no cargado"))?;
            detect_faces_yunet(s, &rgb)?
        }
        FaceModel::Scrfd => {
            let s = models.scrfd.as_mut().ok_or_else(|| anyhow::anyhow!("SCRFD no cargado"))?;
            detect_faces_scrfd(s, &rgb)?
        }
    };
    let faces = &detect.faces;
    println!("Caras detectadas: {}", faces.len());
    let iou_info = match models.face_model {
        FaceModel::YuNet => 0.45f32,
        FaceModel::Scrfd => 0.45f32,
    };
    for (i, face) in faces.iter().enumerate() {
        println!(
            "Cara #{}: {:.2}-{:.2}={:.2} iou={:.2}",
            i + 1,
            face.cls,
            face.obj,
            face.score,
            iou_info
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

    let mut keypoints: Vec<[f32; 3]> = Vec::new();
    let mut roi_opt: Option<[f32; 4]> = None;

    if let Some(face) = faces.first() {
        println!("EMBEDDING / ArcFace");
        println!("  PRE:");
        println!("    - Input: 112x112");
        println!("    - Normalizacion: (x - 127.5) / 128.0");
        let t_embed = std::time::Instant::now();
        let crop = crop_face(&rgb, &face.bbox);
        let emb = face_embedding(&mut models.arcface, &crop)?;
        println!("  Tiempo embedding: {} ms", t_embed.elapsed().as_millis());
        println!("  Embedding facial: {}", emb.len());

        // 2) POSE
        println!("POSE");
        println!("  PRE:");
        println!("    - Pose det input: 224x224, norm x/255.0");
        let t_pose_det = std::time::Instant::now();
        let roi = pose_detection(&mut models.pose_det, &rgb)?;
        let roi_scale_x = rgb.width() as f32 / 224.0;
        let roi_scale_y = rgb.height() as f32 / 224.0;
        roi_opt = Some([
            roi[0] * roi_scale_x,
            roi[1] * roi_scale_y,
            roi[2] * roi_scale_x,
            roi[3] * roi_scale_y,
        ]);
        println!(
            "    - Rescaling ROI: scale_x={:.4}, scale_y={:.4}",
            roi_scale_x, roi_scale_y
        );
        println!("  POST:");
        println!("    - ROI pose: {:?}", roi);
        println!("    - Tiempo pose_det: {} ms", t_pose_det.elapsed().as_millis());

        println!("    - Pose landmarks input: 256x256, norm x/255.0");
        let t_pose_lm = std::time::Instant::now();
        keypoints = pose_landmarks(&mut models.pose_landmarks, &rgb, roi)?;
        let kp_scale_x = rgb.width() as f32 / 256.0;
        let kp_scale_y = rgb.height() as f32 / 256.0;
        for kp in &mut keypoints {
            kp[0] *= kp_scale_x;
            kp[1] *= kp_scale_y;
        }
        println!(
            "    - Rescaling keypoints: scale_x={:.4}, scale_y={:.4}",
            kp_scale_x, kp_scale_y
        );
        println!("    - Tiempo pose_landmarks: {} ms", t_pose_lm.elapsed().as_millis());
        println!("    - Keypoints: {}", keypoints.len());

        let sim = cosine(&emb, &emb);
        println!("Similitud facial consigo mismo: {}", sim);
    }

    let pipeline_total = pipeline_start.elapsed().as_millis();
    println!("Tiempo pipeline total: {} ms", pipeline_total);

    if cli.show_ui {
        let title = format!(
            "YuNet - {}x{} - {}",
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
            iou_info,
        )?;
    }

    Ok(())
}
