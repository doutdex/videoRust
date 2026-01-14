# vision_core

Core inference pipeline shared by the CLI and any future API/DLL wrapper.
This crate owns the model loading, pre-process, inference, and post-process
logic for:

- Face detection: YuNet or SCRFD
- Face embedding: ArcFace
- Pose: BlazePose

## Key API

- `Processor::new(face_model, device_mode, resize_mode)`
- `Processor::process(&RgbImage, &ProcessConfig)`
- `build_json_output(...)` for deterministic JSON output

Main config types:

- `FaceModel`: `YuNet` | `Scrfd`
- `StageMode`: `Detection` | `DetectionEmbedding` | `EmbeddingOnly` | `DetectionEmbeddingPose`
- `EmbeddingMode`: `Off` | `QInt8` | `Raw`
- `DeviceMode`: `Cpu` | `Gpu` (GPU uses DirectML when available)
- `ResizeMode`: `Fast` | `Balanced` | `Quality`

## Minimal usage

```rust
use anyhow::Result;
use vision_core::{DeviceMode, EmbeddingMode, FaceModel, ProcessConfig, Processor, ResizeMode, StageMode};

fn main() -> Result<()> {
    let img = image::open("people.jpg")?.to_rgb8();
    let mut processor = Processor::new(FaceModel::YuNet, DeviceMode::Cpu, ResizeMode::Balanced)?;
    let config = ProcessConfig {
        score_thr: 0.10,
        stage_mode: StageMode::DetectionEmbedding,
        embedding_mode: EmbeddingMode::QInt8,
    };
    let out = processor.process(&img, &config)?;
    println!("faces: {}", out.detect.faces.len());
    Ok(())
}
```

## Model files

Paths are defined in `crates/vision_core/src/lib.rs`:

- `YUNET_ONNX`, `SCRFD_ONNX`, `ARCFACE_ONNX`
- `POSE_DET_ONNX`, `POSE_LM_ONNX`

## Notes

- Preprocess buffers are reused to reduce allocations.
- GPU mode falls back to CPU if DirectML is not available.
