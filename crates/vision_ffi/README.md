# vision_ffi

Minimal FFI wrapper for `vision_core` (cdylib). This is the starting point
if you want to call the Rust pipeline from C/C# (P/Invoke).

## Build

```bash
cargo build -p vision_ffi --release
```

The compiled library is placed in:

- `target/release/vision_ffi.dll` (Windows)
- `target/release/libvision_ffi.so` (Linux)
- `target/release/libvision_ffi.dylib` (macOS)

## Current API

Only a version function is exported right now:

- `vision_core_version() -> *const c_char`

## Next steps (when ready)

- Add `init`, `process`, and `free` APIs.
- Define C structs for inputs/outputs.
- Expose a JSON output helper for easy integration.
