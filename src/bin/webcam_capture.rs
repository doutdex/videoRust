// Demo ONNX: deteccion facial, embedding facial y pose corporal.
// Compilar: cargo build --bin webcam_capture --release
// Ejecutar: cargo run --bin webcam_capture --release

use anyhow::Result;
use image::RgbImage;
use ndarray::{Array4, Ix1, Ix2};
use ort::session::Session;
use ort::value::Tensor;

// ----------------------
// Modelos ONNX
// ----------------------

struct Models {
    yunet: Session,
    arcface: Session,
    blazepose: Session,
}

impl Models {
    fn load() -> Result<Self> {
        let _ = ort::init().with_name("vision-core").commit()?;

        let yunet = Session::builder()?
            .commit_from_file("yunet.onnx")?;

        let arcface = Session::builder()?
            .commit_from_file("arcface.onnx")?;

        let blazepose = Session::builder()?
            .commit_from_file("blazepose_full.onnx")?;

        Ok(Self { yunet, arcface, blazepose })
    }
}

// ----------------------
// Utilidades generales
// ----------------------

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-6)
}

// crop de cara a partir de bounding box [x, y, w, h]
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

// ----------------------
// Deteccion de cara (YuNet)
// ----------------------

fn detect_faces(session: &mut Session, rgb: &RgbImage) -> Result<Vec<[f32; 4]>> {
    let (w, h) = rgb.dimensions();

    // entrada NHWC
    let input = Array4::from_shape_fn((1, h as usize, w as usize, 3), |(_, y, x, c)| {
        rgb.get_pixel(x as u32, y as u32)[c] as f32
    });

    let input = Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![input])?;

    let dets = outputs[0].try_extract_array::<f32>()?;
    let dets = dets.into_dimensionality::<Ix2>()?;

    let mut faces = Vec::new();
    for row in dets.outer_iter() {
        let score = row[4];
        if score > 0.7 {
            faces.push([row[0], row[1], row[2], row[3]]);
        }
    }

    Ok(faces)
}

// ----------------------
// Embedding facial (ArcFace)
// ----------------------

fn face_embedding(session: &mut Session, face: &RgbImage) -> Result<Vec<f32>> {
    let resized = image::imageops::resize(face, 112, 112, image::imageops::Lanczos3);

    let input = Array4::from_shape_fn((1, 112, 112, 3), |(_, y, x, c)| {
        resized.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    });

    let input = Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![input])?;

    let emb = outputs[0].try_extract_array::<f32>()?;
    let emb = emb.into_dimensionality::<Ix1>()?;
    Ok(emb.to_vec())
}

// ----------------------
// BlazePose: keypoints
// ----------------------

fn blazepose(session: &mut Session, rgb: &RgbImage) -> Result<Vec<[f32; 3]>> {
    let (w, h) = rgb.dimensions();

    let input = Array4::from_shape_fn((1, h as usize, w as usize, 3), |(_, y, x, c)| {
        rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    });

    let input = Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![input])?;

    let kp = outputs[0].try_extract_array::<f32>()?;
    let kp = kp.into_dimensionality::<Ix2>()?;

    let mut keypoints = Vec::new();
    for row in kp.outer_iter() {
        keypoints.push([row[0], row[1], row[2]]);
    }

    Ok(keypoints)
}

// ----------------------
// ROI tracking BlazePose
// ----------------------

#[derive(Clone, Debug)]
struct BlazePoseROI {
    cx: f32,
    cy: f32,
    scale: f32,
}

fn update_roi(keypoints: &[[f32; 3]]) -> BlazePoseROI {
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for kp in keypoints {
        xs.push(kp[0]);
        ys.push(kp[1]);
    }

    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;
    let scale = ((max_x - min_x).max(max_y - min_y)) * 1.5;

    BlazePoseROI { cx, cy, scale }
}

// ----------------------
// Embedding corporal simple
// ----------------------

fn pose_embedding(keypoints: &[[f32; 3]]) -> Vec<f32> {
    let mut emb = Vec::new();

    let pairs = [
        (11, 12), // hombros
        (23, 24), // caderas
        (11, 23), // torso izq
        (12, 24), // torso der
        (13, 15), // brazo izq
        (14, 16), // brazo der
        (25, 27), // pierna izq
        (26, 28), // pierna der
    ];

    for (a, b) in pairs {
        if a < keypoints.len() && b < keypoints.len() {
            let dx = keypoints[a][0] - keypoints[b][0];
            let dy = keypoints[a][1] - keypoints[b][1];
            let dist = (dx * dx + dy * dy).sqrt();
            emb.push(dist);
        } else {
            emb.push(0.0);
        }
    }

    emb
}

// ----------------------
// MAIN: demo con una imagen
// ----------------------

fn main() -> Result<()> {
    println!("Cargando modelos...");
    let mut models = Models::load()?;
    println!("Modelos cargados.");

    // TODO: por ahora se usa una imagen estatica.
    // Luego se reemplaza por frames de Media Foundation.
    let img_path = "test.jpg";
    let img = image::open(img_path)?;
    let rgb = img.to_rgb8();

    // 1) Deteccion de caras
    let faces = detect_faces(&mut models.yunet, &rgb)?;
    println!("Caras detectadas: {}", faces.len());

    if let Some(first_face) = faces.first() {
        let face_crop = crop_face(&rgb, first_face);
        let face_emb = face_embedding(&mut models.arcface, &face_crop)?;
        println!("Embedding facial len = {}", face_emb.len());

        // 2) Pose
        let keypoints = blazepose(&mut models.blazepose, &rgb)?;
        println!("Keypoints pose = {}", keypoints.len());

        let roi = update_roi(&keypoints);
        println!("ROI BlazePose: {:?}", roi);

        let body_emb = pose_embedding(&keypoints);
        println!("Embedding corporal len = {}", body_emb.len());

        // 3) Ejemplo de similitud consigo mismo
        let sim_face = cosine(&face_emb, &face_emb);
        let sim_body = cosine(&body_emb, &body_emb);
        println!("Similitud cara = {sim_face}, cuerpo = {sim_body}");
    } else {
        println!("No se detecto ninguna cara en la imagen.");
    }

    Ok(())
}
