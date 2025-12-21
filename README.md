# Vision Core (Rust) - Face and Pose Pipeline

Objetivo
- Probar deteccion facial con modelos ONNX (YuNet o SCRFD) y generar embeddings faciales (ArcFace).
- Mostrar resultados en consola y en una ventana (UI) con cajas y landmarks.
- Base para integrar webcam y tracking en tiempo real.

Estructura
- `src/main.rs` -> `picAnalizer` (pipeline ONNX principal).
- `src/bin/webcam_capture.rs` -> demo ONNX basico con imagen estatica.
- `src/bin/face_detection.rs` -> demo simple sin ML (rust puro).

Requisitos
- Windows 10/11.
- Rust estable.
- MSVC Build Tools (linker de Windows).

Modelos ONNX esperados (carpeta `./models`)
- `yunet_n_640_640.onnx` (facemodel=1).
- `scrfd-det_2.5g.onnx` (facemodel=2).
- `arcface.onnx` (embedding 112x112).
- `pose_detection.onnx`.
- `pose_landmarks_detector_full.onnx`.

Imagen de prueba
- Por defecto usa `people1.jpg` en la raiz del proyecto.

Uso rapido
```powershell
cargo run --bin picAnalizer -- showUI showLandmarks facemodel=1
cargo run --bin picAnalizer -- showUI showLandmarks facemodel=2
```

Parametros CLI
- `facemodel=1` -> YuNet (score = cls * obj).
- `facemodel=2` -> SCRFD (score = cls).
- `showUI` -> abre ventana con detecciones.
- `showLandmarks` o `showFacialLandmarks` -> dibuja landmarks faciales.
- `showAll` -> imprime info de inputs/outputs de modelos.

Postprocesos por modelo
YuNet (facemodel=1)
- Input: 640x640, BGR sin normalizacion adicional.
- Score real: `cls * obj`.
- NMS: IoU 0.45.
- Landmarks: 5 puntos por cara (decodificados desde salida ONNX).

SCRFD (facemodel=2)
- Input: resize + pad a 640x640, con normalizacion (x - 127.5) / 128.0.
- Score real: salida directa del modelo.
- Soft-NMS (gauss): IoU 0.45, sigma 0.5.
- Landmarks: solo si el modelo los expone.

UI
- Ventana fija 1920x1080 (1080p).
- Pan con flechas o WASD, Shift acelera.
- Esc cierra la ventana.

Notas
- Si `cargo build` falla con "access denied", cierra la ventana del binario.
- Los keypoints de pose son corporales, no de cara. Los landmarks faciales vienen de YuNet/SCRFD.

Troubleshooting
- Error `msvcrt.lib`: instalar workload "Desktop development with C++" en Visual Studio Installer.
- Error "file not found": verifica que los ONNX y `people1.jpg` esten en la ruta correcta.
