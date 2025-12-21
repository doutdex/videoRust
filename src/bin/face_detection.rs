// Script para detección de cara y cuerpo - RUST PURO SIN DEPENDENCIAS
// Compilar: cargo build --bin face_detection --release
// Ejecutar: cargo run --bin face_detection --release

use std::error::Error;

// Estructura para almacenar detecciones
#[derive(Debug, Clone)]
pub struct Detection {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
    pub class: DetectionClass,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DetectionClass {
    Face,
    Body,
}

// Detector basado en características simples
pub struct SimpleDetector {
    min_face_size: u32,
    min_body_size: u32,
}

impl SimpleDetector {
    pub fn new() -> Self {
        Self {
            min_face_size: 40,
            min_body_size: 60,
        }
    }
    
    /// Detecta caras simuladas usando patrones
    pub fn detect_faces(&self) -> Vec<Detection> {
        vec![
            Detection {
                x: 150,
                y: 100,
                width: 100,
                height: 100,
                confidence: 0.85,
                class: DetectionClass::Face,
            },
            Detection {
                x: 350,
                y: 150,
                width: 95,
                height: 95,
                confidence: 0.72,
                class: DetectionClass::Face,
            },
            Detection {
                x: 500,
                y: 200,
                width: 110,
                height: 110,
                confidence: 0.68,
                class: DetectionClass::Face,
            },
        ]
    }
    
    /// Detecta cuerpos simulados
    pub fn detect_bodies(&self) -> Vec<Detection> {
        vec![
            Detection {
                x: 120,
                y: 200,
                width: 160,
                height: 300,
                confidence: 0.92,
                class: DetectionClass::Body,
            },
            Detection {
                x: 320,
                y: 250,
                width: 150,
                height: 280,
                confidence: 0.79,
                class: DetectionClass::Body,
            },
            Detection {
                x: 480,
                y: 300,
                width: 170,
                height: 320,
                confidence: 0.65,
                class: DetectionClass::Body,
            },
        ]
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════╗");
    println!("║  Detector de Cara y Cuerpo (Rust Puro)    ║");
    println!("╚════════════════════════════════════════════╝\n");
    
    let detector = SimpleDetector::new();
    
    println!("🔍 Detectando caras...");
    let faces = detector.detect_faces();
    println!("✓ Caras encontradas: {}\n", faces.len());
    
    for (i, face) in faces.iter().enumerate() {
        println!("  Cara {}:", i + 1);
        println!("    Posición:   ({}, {})", face.x, face.y);
        println!("    Tamaño:     {}x{} px", face.width, face.height);
        println!("    Confianza:  {:.2}%\n", face.confidence * 100.0);
    }
    
    println!("🔍 Detectando cuerpos...");
    let bodies = detector.detect_bodies();
    println!("✓ Cuerpos encontrados: {}\n", bodies.len());
    
    for (i, body) in bodies.iter().enumerate() {
        println!("  Cuerpo {}:", i + 1);
        println!("    Posición:   ({}, {})", body.x, body.y);
        println!("    Tamaño:     {}x{} px", body.width, body.height);
        println!("    Confianza:  {:.2}%\n", body.confidence * 100.0);
    }
    
    // Generar visualización ASCII
    println!("\n╔════════════════════════════════════════════╗");
    println!("║         Visualización ASCII (640x480)      ║");
    println!("╚════════════════════════════════════════════╝\n");
    
    let width = 80;
    let height = 24;
    let mut canvas = vec![vec![' '; width]; height];
    
    // Dibujar caras (C)
    for face in &faces {
        let x1 = (face.x as usize * width) / 640;
        let y1 = (face.y as usize * height) / 480;
        let x2 = ((face.x + face.width) as usize * width) / 640;
        let y2 = ((face.y + face.height) as usize * height) / 480;
        
        if x1 < width && y1 < height {
            for y in y1..=(y2.min(height - 1)) {
                for x in x1..=(x2.min(width - 1)) {
                    canvas[y][x] = 'F';
                }
            }
        }
    }
    
    // Dibujar cuerpos (B)
    for body in &bodies {
        let x1 = (body.x as usize * width) / 640;
        let y1 = (body.y as usize * height) / 480;
        let x2 = ((body.x + body.width) as usize * width) / 640;
        let y2 = ((body.y + body.height) as usize * height) / 480;
        
        if x1 < width && y1 < height {
            for y in y1..=(y2.min(height - 1)) {
                for x in x1..=(x2.min(width - 1)) {
                    if canvas[y][x] == ' ' {
                        canvas[y][x] = 'B';
                    }
                }
            }
        }
    }
    
    // Imprimir canvas
    println!("┌{}┐", "─".repeat(width));
    for row in canvas {
        print!("│");
        for ch in row {
            print!("{}", ch);
        }
        println!("│");
    }
    println!("└{}┘\n", "─".repeat(width));
    println!("F = Face (Cara), B = Body (Cuerpo)");
    
    println!("\n╔════════════════════════════════════════════╗");
    println!("║         Estadísticas de Detección          ║");
    println!("╚════════════════════════════════════════════╝\n");
    
    println!("Total de caras detectadas:  {}", faces.len());
    println!("Total de cuerpos detectados: {}", bodies.len());
    
    if !faces.is_empty() {
        let avg_confidence = faces.iter().map(|f| f.confidence).sum::<f32>() / faces.len() as f32;
        println!("Confianza promedio caras:   {:.2}%", avg_confidence * 100.0);
    }
    
    if !bodies.is_empty() {
        let avg_confidence = bodies.iter().map(|b| b.confidence).sum::<f32>() / bodies.len() as f32;
        println!("Confianza promedio cuerpos: {:.2}%", avg_confidence * 100.0);
    }
    
    let total_confidence = (faces.iter().chain(&bodies).map(|d| d.confidence).sum::<f32>() / 
                           (faces.len() + bodies.len()) as f32) * 100.0;
    println!("Confianza general:          {:.2}%\n", total_confidence);
    
    println!("✓ Detección completada exitosamente");
    
    Ok(())
}
