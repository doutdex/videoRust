# Vision Core (Rust) - Face and Pose Pipeline

Objetivo
- Probar deteccion facial con modelos ONNX (YuNet o SCRFD) y generar embeddings faciales (ArcFace).
- Mostrar resultados en consola o JSON, y en una ventana (UI) con cajas y landmarks.
- Guardar resultados en disco cuando se requiera (outDirDetections).
- Monitor simple de FPS para RTSP en consola.

Estructura
- `src/main.rs` -> `picAnalizer` (pipeline ONNX principal).
- `src/bin/webcam_capture.rs` -> monitor simple de FPS en consola (URL configurable).
- `src/bin/face_detection.rs` -> demo simple sin ML (rust puro).

Requisitos
- Windows 10/11.
- Rust estable.
- MSVC Build Tools (linker de Windows).

Modelos ONNX esperados (carpeta `./models`)
- `yunet_n_640_640.onnx` (facemodel=1).
- `scrfd-det_10g.onnx` (facemodel=2).
- `arcface.onnx` (embedding 112x112).
- `blazepose-pose_detection.onnx`.
- `blazepose-pose_landmarks_detector_full.onnx`.

Imagen de prueba
- Por defecto usa `people.jpg` en la raiz del proyecto.

Uso rapido
```powershell
cargo run --bin picAnalizer -- uishow showLandmarks facemodel=1
cargo run --bin picAnalizer -- uishow showLandmarks facemodel=2
cargo run --bin picAnalizer -- textmode=json facemodel=1 infile=people.jpg
cargo run --bin picAnalizer -- stages=1 outfile=peopleout.jpg infile=people.jpg
cargo run --bin webcam_capture -- url=rtsp://mi-camara interval_ms=33
```

Ejemplo de imagen (entrada/salida)

Entrada:
![Entrada](readme-img1.jpg)

Salida:
![Salida](readme-img1 out.jpg)

Parametros CLI
- `facemodel=1` -> YuNet (score = cls * obj).
- `facemodel=2` -> SCRFD (score = cls).  Robusto
- `uishow` -> abre ventana con detecciones (case-insensitive, `uiShow` tambien funciona).
- `showLandmarks` o `showFacialLandmarks` -> dibuja landmarks faciales.
- `showAll` -> imprime info de inputs/outputs de modelos.
- `infile=path` -> ruta de imagen (default: people.jpg).
- `threshold=val` -> umbral de score (default: 0.10 en model 1, 0.50 en model 2  ).
- `textmode=0|json` -> sin output (modo silent) | salida JSON por etapas.
- `stages=1|11|01|111` -> 1=Detection, 11=Detection+Embedding, 01=Embedding (sin Detection), 111=Detection+Embedding+Pose.
- `outfile=path|0` -> guarda imagen anotada (0 = no genera archivo).
- `embed=0|1|2` -> Activa embeddings en JSON: 0=off, 1=qint8 quantization en int 8 hex concatenado (default), 2=raw (f32) vector 512.
- `outDirDetections=1` -> genera carpeta `picAnalizer-unixtime` con:
  - `unixtime-in.jpg` (copia input)
  - `unixtime-out.jpg` (render)
  - `unixtime.json` (analisis)
  - `unixtime-1.jpg`, `unixtime-2.jpg`, ... (crops para embedding)
  - fuerza: `textmode=json`, `textmode=0`, `embed=2`, `showLandmarks`
- `device=cpu|gpu` -> selecciona CPU o GPU (fallback a CPU si no hay EP).
- `resize=fast|balanced|quality` -> controla calidad/velocidad del resize (default: balanced).
  - fast = Nearest, balanced = Bilinear, quality = CatmullRom (fast_image_resize).

Embedding (JSON)
- En `textmode=json`, los embeddings salen  quantizado como `qint8` en hex concatenado (default) o `raw` con `embed=2`.
- Con `embed=0`, no se incluyen embeddings en el JSON.
- Cada vector incluye `scale` por cara y `hex` (string con bytes int8 en hex concatenado, two's complement).
- Reconstruccion aproximada: `float = int8 * scale` (zero_point=0).

Ejemplos de dequantizacion
Python (qint8 hex concatenado)
```python
import numpy as np

scale = vec["scale"]
hex_str = vec["hex"]
vals_u8 = np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
values = vals_u8.view(np.int8)
emb_f32 = values.astype(np.float32) * scale
```

Rust (qint8 hex concatenado)
```rust
let scale: f32 = vec.scale;
let hex_str = &vec.hex;
let values_u8: Vec<u8> = (0..hex_str.len())
    .step_by(2)
    .map(|i| u8::from_str_radix(&hex_str[i..i + 2], 16).unwrap_or(0))
    .collect();
let values: Vec<i8> = values_u8.into_iter().map(|v| v as i8).collect();
let emb_f32: Vec<f32> = values.into_iter().map(|v| v as f32 * scale).collect();
```

Raw (f32)
- Con `embed=2`, cada vector trae `values` como floats directamente.

Tiempos por etapa
- Stage 1: solo Detection (pre, infer, post y total).
- Stage 11: Detection + Embedding (tiempo embedding adicional).
- Stage 111: Detection + Embedding + Pose (tiempo pose_det y pose_landmarks).
- Stage 01: solo Embedding (requiere imagen 112x112; no usa Detection).
- `Tiempo pipeline total` incluye el tiempo de las etapas activas + overhead (I/O y render).

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
- Ventana con tamano de imagen y tope 1280x720.
- La imagen se escala al tamano de la ventana (sin pan).
- Tecla "Esc" cierra la ventana.

Webcam / RTSP (monitor FPS)
- El bin `webcam_capture` solo imprime la URL, FPS y timestamp.
- Cambia la URL con `url=...` en la linea de comandos.
- Cambia el intervalo con `interval_ms=...` (default 33 ms).
- En Windows, `Ctrl+C` termina el proceso (exit code 0xc000013a es normal).

Ayuda
```powershell
cargo run --bin picAnalizer -- --help
```

Notas
- Si `cargo build` falla con "access denied", cierra la ventana del binario.
- Los keypoints de pose son corporales, no de cara. Los landmarks faciales vienen de YuNet/SCRFD.

Troubleshooting
- Error `msvcrt.lib`: instalar workload "Desktop development with C++" en Visual Studio Installer.
- Error "file not found": verifica que los ONNX y `people.jpg` esten en la ruta correcta.
Defaults (sin parametros)
- Obligatorios: ninguno.
- Procesa `people.jpg` con `facemodel=1` y `stages=11`.
- Salida consola (texto), con UI y landmarks faciales activos.
- No genera archivo de salida.

   
     CLI Ejemplo Outout TXT 
   
Cargando modelos...
Modelos cargados.
Imagen: people640.jpg (640x640, tipo Jpeg)
DETECCION / YuNet
  ONNX: ./models/yunet_n_640_640.onnx
  PRE:
    - Input: 640x640, BGR
    - Normalizacion: ninguna
  POST:
    - Score: cls * obj
    - score_thr=0.10
    - TopK por nivel: enabled (200)
    - NMS: IoU=0.45
Caras detectadas: 4
Cara #1: 0.87-0.12=0.11 iou=0.45
Cara #2: 0.82-0.23=0.19 iou=0.45
Cara #3: 0.88-0.30=0.27 iou=0.45
Cara #4: 0.88-0.93=0.82 iou=0.45
Tiempo deteccion:
  pre:   3 ms
         - resize: 0 ms
         - norm:   2 ms
  infer: 21 ms
  post:  0 ms
  total: 24 ms
EMBEDDING / ArcFace
  PRE:
    - Input: 112x112
    - Normalizacion: (x - 127.5) / 128.0
  Tiempo embedding: 247 ms
  Embeddings faciales: 4 (len=512)
    - Cara #1: 512
    - Cara #2: 512
    - Cara #3: 512
    - Cara #4: 512
Similitud facial consigo mismo: 0.9999999
Tiempo pipeline total: 274 ms      
   

CLI Ejemplo Outout JSON   (cargo run --bin picAnalizer -- textmode=json stages=111 infile=people640.jpg facemodel=2)

 {
  "image": {
    "path": "people640.jpg",
    "width": 640,
    "height": 640,
    "type": "Jpeg"
  },
  "detection": {
    "model": {
      "name": "SCRFD",
      "onnx": "./models/scrfd-det_10g.onnx",
      "score_thr": 0.50,
      "iou_thr": 0.45
    },
    "pre": {
      "input": "640x640",
      "color": "RGB",
      "normalization": "(x - 127.5) / 128.0"
    },
    "post": {
      "score": "score",
      "topk": {"enabled":false,"value":0},
      "nms": {"type":"soft_nms","iou":0.45,"sigma":0.50}
    },
    "timing_ms": {
      "pre": 1,
      "pre_resize": 0,
      "pre_norm": 1,
      "infer": 157,
      "post": 0,
      "total": 158
    },
    "faces": [
      {"id":1,"score":0.9033,"cls":0.9033,"obj":1.0000,"bbox":[381.90,145.52,50.24,75.23],"landmarks":[[394.51,180.61],[417.51,175.17],[408.61,192.63],[397.58,200.53],[421.76,195.91]]},
      {"id":2,"score":0.8960,"cls":0.8960,"obj":1.0000,"bbox":[217.85,126.42,54.23,70.56],"landmarks":[[230.79,155.20],[257.14,152.08],[245.50,165.64],[232.93,177.49],[259.66,174.73]]},
      {"id":3,"score":0.8629,"cls":0.8629,"obj":1.0000,"bbox":[28.00,98.55,58.07,71.70],"landmarks":[[57.55,123.07],[80.62,127.54],[75.08,136.59],[53.08,146.29],[76.98,150.11]]},
      {"id":4,"score":0.8623,"cls":0.8623,"obj":1.0000,"bbox":[545.77,102.84,54.26,81.68],"landmarks":[[555.67,133.35],[566.65,136.80],[547.32,148.65],[556.56,161.05],[565.94,163.44]]}
    ]
  },
  "embedding": {
    "enabled": true,
    "input": "112x112",
    "normalization": "(x - 127.5) / 128.0",
    "encoding": "int8-hex",
    "time_ms": 251,
    "count": 4,
    "len": 512,
    "quant": {"zero_point": 0, "scale": "per-vector"},
    "vectors": [
      {"id": 1, "scale": 0.00410302, "hex": "FACAD78EF4382615C63824F1F822E728B4090DC0F1F6F904EEFF3EE3C2DAE4D9D42C07D0182004FED1C2DAF4EDEB1D4119062EE7030FF3F660E8160C4CE0E80818FBEAE6FA01D6F91921DDE817A53A04CDE7DF4A4002B30EE03D1F0950F71ED0130208DDEEE23D10FA2D160522F61448DB11EC0AC9F421D410E82935500186F732AC36E9FF0612D7FB1D21141A23EA42B8072C19C47858D3F0F4131FCC4DF339BF41FF0AF91C39E2E920E4DDE027EE3001EB052FDEE0B111E438EBFBF0F4680D04D4F40DD4F5D8C009C92DF72EBBDBC0C2EF1EE8E0CDF80CF3111DD1F6EE1AFEF3FF05F5F8219E02FED705D21E16FCEE08C1F94ACF02E6F723181CDA13F14D511C1032A90C3406C6EB4F02DBF223EE010FEC1807E0061CE5D2E91201F1DC3403EFBDD5D55F2EC60E11060BDFE40923E0EB443FF2ECB5F287C2D2D413EC0907FCF41B0020052043DE22FDF00B00231DFF19ABCFFF12CB08C5CBE3FAC2FA04EEE9B4B7DF25CE0239D5C8FC1FF4D82FF1F9E104FF0928F32830D006CD36142807230DE60002F8E4C9046001D502E8FFFDEFD0DFFDF60E1F290AEC21D7D3F8CE2BC6C425FBF0371D5928E3EE36072117CAFF07BB2EE9FE0704EEFBE6C62FEDE825F045FE32BCE9EED318BEC906DFD3ECE4E5D92DE9C6D41FFB7FF8D8F20B0D29100BF72D29EE060433E13FFED4FD23E1310DD910C4F754D39D1D47F632240D20D1F8"},
      {"id": 2, "scale": 0.00592792, "hex": "13EE1718FAE7FBEB2AF9EEEC17DB4D3B0726E9029130C3F10ED50A16E332F9F404170FA0EE2CF329FE1DE82FEEE0030CB4EB102D3B2DE6191FEEFDE527D4DBB21BC8F638F9EFE3E20EED28FF31EF30E5E904F32D072BA2D30FFBD8A45932C7B802210DC00DFF47F710FDE83215170F24C60EE70F0408D6DD18CA68261D1FAD0B53D7AD08E4EAEF06F3D626FD4342FB0AC6FC1E2BF4363A2E3C111DF1EA27EA00E328B2360DDD4433FBF614DE1B001800EDE39D3113BCAC2DF1CFE70BD8244112172EFD2FF540F5CAE5E2ABF008E3B2C80C05220B1DEAFE010D16D2D3FDD606F71E1643F8D91DF80FE3FF5CE702F21918F80FCB38F02CFEDAE94535CC1CF010EB0D0E1B08EB18D6C33048D20F0D164629F201C90516E7DDEEC07C21FEEAF57FF8C1EF14D7251F2C10EB3BBAE4FB225A2806003823F3AEFEB5C806BFB604F2F71058FC150F3800FC2C03B2F20ED0E802E812D1DA302B1E2409D9EB03D1032CC91A13120B18DB4603CB021B4FF2D3010F1BF0E6E70A33FD342291D8083C34F61733EDF6DA0DD8CF1617F33BE52406B80419D6BD22F8053923E914D1E1FDCE1217DECBF1D70FEF03524836FC22DD1AD2FB12F9F2B4140804F00C9DCAF842D3DE18FEFB0160BDDE03D44CF9E30E18B6F42816E726ECD6DAD9F46CE3F0140E09473F6E4E40FD29F9F2D0E8FE0D4BE5F82E032047EDE4F4DA4B2129BBE5F1F0CC4AC69F"},
      {"id": 3, "scale": 0.00407367, "hex": "EDD9CDBB1E19E630D6F6FDF50E0A073BFBB9F9D21D3ADAD1E7D50A34FDCF2712F4E219F6E2E42B00FF0A071ECD0DEC2140C9C3F039DA050B63111BCFF028EAC009E3E9C119F9B047F1F2961D0FDBF2E31106F8F91400E6F23E02191013D6F3333D061D170303DD53FACCFF5E0615E3E9E5DF180E13290FD3EE0235F32BF502D322F9CBEB59031AFBBA123227254E090EDE2F3017D32C32D72D00DE0B8100C909D4ECEF1E3E1330FFE6040FE704FDF5A8001BF61503E2FFFC081CF9F603F63E1C064C122322EEE6E0141F17FF1CD223CFD95F21FEDBEB2009DD0604F702E2D023FAE02B13061AE907F7251A362932D5FDDFDB0A06FBE9FC011DD908D412E60A41E734FB00D407ECEB02CD30FCEEF0FA01F1E9F81A0F230CE3C8E2E912FFE92C04EED2090F201907FA18CFFDB6290F2B02183B3904EDFD0CB6F0EAE3E90327D40D1B18EEEBDFEF130601BEEEF72522D6D0C11D81161C02FBC7EC40FAE7080A070F00D920F804EBE7BDDE23030314F915DC0CCE18EF12EAFC1FBFEAF5162F4F122026A5F2F0161BBF2D7BDE18EF2CABDEE1921BED0EF11806D30AED2B21F5160F0C014DF9ECFEF2F6FCFBCC072ADB10EF01E1F5CFD4F20619E0D9100AE113B7011333230321EA03030ADA1C24F100FB29D8BC140DD300E40A2F140A0817CB23E9D84225EAE309FFBE2813FE04CDE6002116CC04FCDA32F449FE18E8CD2514F1FBCE"},
      {"id": 4, "scale": 0.00459593, "hex": "EA652EB7F032FF19CDF3F74CECDBFD1F402EE4D8260F1A11C972ED18EFEDE569C0300E0F200439084330042400511AD1C5DA0F0B05DE37233D431514FA09FE0307BC12E4C704ECD25ADDBC16EB470A19DDDFEE0A01073BF8F9151A0DD527F7F8141152EBD208E70310F0E84FD1FA31E907BE0224E10430D409FA0108F0F7140010AD4202AD24EA081BF0000A426DEBF2CAEE127E105C210BDE13D31881602A53C2D62C1C33300332FE12D14DBF1BEC1B3CDE22AB19D3E607D7FBF656D0EB5D07190ED9EF12B913D60E1219E609C3FF06E04EE5F8E50416DC44E6E41937BB515BD3F0FDEC0701A0E2EE31CCD70046C80CDBDD06ECCF2CECE2381623CC20E819D70F230C292103F8150AEEF60029E5449A3C0C0636D23D0DEA02CDD3FFF6E508FEF6F4E016F218E8FD313B3815FA136721294717BC539EC7932011DAE0F5FB2FDBEAA2511B0E3823C44A22DD240B07CCEEEEF1C8D1EA2021CF0737E8BEFD3413FA06F3190CDBEF01E6EA3329EBFA1215D7160457FB3C0DFEF9D0C8FF2B1300FCDFFECCFC15A612DC0F3E4B2505CCE8EDF7E3F72205C4D6E8DEF2F6221E03D4DF372100C416F9EDB5FF0CFE2CCF0A4132E6ED13C02EF62E49BA140DEA1B1201183E3329E51609F50A0D2325D235D3E049F60AE2E5CB2E2613E90CC2C6BB1B1D14122FF3DFC82911BC0115F8060B0B095AEDDDEAE5FE0FAE2026DA051D4A15F51AE4"}
    ]
  },
  "pose": {
    "enabled": true,
    "det": {"input":"224x224","normalization":"x/255.0","time_ms": 70},
    "landmarks": {"input":"256x256","normalization":"x/255.0","time_ms": 90},
    "keypoints_len": 33
  },
  "pipeline": {"stages": "111", "total_ms": 411, "gap_ms": 0}
