use anyhow::Result;
use image::RgbImage;
use ndarray::{Array4, Ix1, Ix2};
use ort::session::Session;
use ort::value::Tensor;

// ---------------------------------------------------------
// MODELOS
// ---------------------------------------------------------

struct Models {
    yunet: Session,
    arcface: Session,
    pose_det: Session,
    pose_landmarks: Session,
}

impl Models {
    fn load() -> Result<Self> {
        let _ = ort::init().with_name("vision-core").commit()?;

        let yunet = Session::builder()?
            .commit_from_file("./models/yunet_n_640_640.onnx")?;

        let arcface = Session::builder()?
            .commit_from_file("./models/arcface.onnx")?;

        let pose_det = Session::builder()?
            .commit_from_file("./models/pose_detection.onnx")?;

        let pose_landmarks = Session::builder()?
            .commit_from_file("./models/pose_landmarks_detector_full.onnx")?;

        Ok(Self {
            yunet,
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

// ---------------------------------------------------------
// YUNET — DETECCIÓN DE CARAS
// ---------------------------------------------------------

fn detect_faces(session: &mut Session, rgb: &RgbImage) -> Result<Vec<[f32; 4]>> {
    let resized = image::imageops::resize(rgb, 640, 640, image::imageops::Triangle);

    let mut input = Array4::<f32>::zeros((1, 3, 640, 640));
    for y in 0..640 {
        for x in 0..640 {
            let p = resized.get_pixel(x as u32, y as u32);
            input[[0, 0, y, x]] = p[0] as f32;
            input[[0, 1, y, x]] = p[1] as f32;
            input[[0, 2, y, x]] = p[2] as f32;
        }
    }

    let input = Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![input])?;

    let dets = outputs[0].try_extract_array::<f32>()?;
    let dets = dets.into_dimensionality::<Ix2>()?;

    let mut faces = Vec::new();
    for row in dets.outer_iter() {
        if row[4] > 0.7 {
            faces.push([row[0], row[1], row[2], row[3]]);
        }
    }

    Ok(faces)
}

// ---------------------------------------------------------
// ARCFACE — EMBEDDING FACIAL
// ---------------------------------------------------------

fn face_embedding(session: &mut Session, face: &RgbImage) -> Result<Vec<f32>> {
    let resized = image::imageops::resize(face, 112, 112, image::imageops::Lanczos3);

    let mut input = Array4::<f32>::zeros((1, 3, 112, 112));
    for y in 0..112 {
        for x in 0..112 {
            let p = resized.get_pixel(x as u32, y as u32);
            input[[0, 0, y, x]] = (p[0] as f32 - 127.5) / 128.0;
            input[[0, 1, y, x]] = (p[1] as f32 - 127.5) / 128.0;
            input[[0, 2, y, x]] = (p[2] as f32 - 127.5) / 128.0;
        }
    }

    let input = Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![input])?;

    let emb = outputs[0].try_extract_array::<f32>()?;
    let emb = emb.into_dimensionality::<Ix1>()?;
    Ok(emb.to_vec())
}

// ---------------------------------------------------------
// BLAZEPOSE — DETECCIÓN + LANDMARKS
// ---------------------------------------------------------

fn pose_detection(session: &mut Session, rgb: &RgbImage) -> Result<[f32; 4]> {
    let resized = image::imageops::resize(rgb, 256, 256, image::imageops::Triangle);

    let mut input = Array4::<f32>::zeros((1, 3, 256, 256));
    for y in 0..256 {
        for x in 0..256 {
            let p = resized.get_pixel(x as u32, y as u32);
            input[[0, 0, y, x]] = p[0] as f32 / 255.0;
            input[[0, 1, y, x]] = p[1] as f32 / 255.0;
            input[[0, 2, y, x]] = p[2] as f32 / 255.0;
        }
    }

    let input = Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![input])?;

    let det = outputs[0].try_extract_array::<f32>()?;
    let det = det.into_dimensionality::<Ix1>()?;
    Ok([det[0], det[1], det[2], det[3]])
}

fn pose_landmarks(session: &mut Session, rgb: &RgbImage, _roi: [f32; 4]) -> Result<Vec<[f32; 3]>> {
    let resized = image::imageops::resize(rgb, 256, 256, image::imageops::Triangle);

    let mut input = Array4::<f32>::zeros((1, 3, 256, 256));
    for y in 0..256 {
        for x in 0..256 {
            let p = resized.get_pixel(x as u32, y as u32);
            input[[0, 0, y, x]] = p[0] as f32 / 255.0;
            input[[0, 1, y, x]] = p[1] as f32 / 255.0;
            input[[0, 2, y, x]] = p[2] as f32 / 255.0;
        }
    }

    let input = Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![input])?;

    let kp = outputs[0].try_extract_array::<f32>()?;
    let kp = kp.into_dimensionality::<Ix2>()?;

    Ok(kp.outer_iter().map(|row| [row[0], row[1], row[2]]).collect())
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------

fn main() -> Result<()> {
    println!("Cargando modelos...");
    let mut models = Models::load()?;
    println!("Modelos cargados.");

    let img = image::open("people1.jpg")?;
    let rgb = img.to_rgb8();

    // 1) CARAS
    let faces = detect_faces(&mut models.yunet, &rgb)?;
    println!("Caras detectadas: {}", faces.len());

    if let Some(face) = faces.first() {
        let crop = crop_face(&rgb, face);
        let emb = face_embedding(&mut models.arcface, &crop)?;
        println!("Embedding facial: {}", emb.len());

        // 2) POSE
        let roi = pose_detection(&mut models.pose_det, &rgb)?;
        println!("ROI pose: {:?}", roi);

        let keypoints = pose_landmarks(&mut models.pose_landmarks, &rgb, roi)?;
        println!("Keypoints: {}", keypoints.len());

        let sim = cosine(&emb, &emb);
        println!("Similitud facial consigo mismo: {}", sim);
    }

    Ok(())
}
