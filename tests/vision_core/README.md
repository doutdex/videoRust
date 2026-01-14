# vision_core_test

Small QA harness that exercises the same pre/infer/post pipeline used by
`vision_core::Processor`. This is meant to be a readable reference when
deciding between:

- Wrapping `vision_core` as a DLL (see `crates/vision_ffi`).
- Reimplementing the pipeline in C#.

The JSON output is built by `vision_core::build_json_output`, which writes
keys in a fixed order, so the output is deterministic and consistent across
models.

## Run

From repo root:

```bash
cargo run --manifest-path tests/vision_core/Cargo.toml -- --model=yunet --infile=people1.jpg
```

Optional flags:

- `--model=yunet|scrfd|all`
- `--stage=1|11|01|111`
- `--embed=off|qint8|raw`
- `--device=cpu|gpu`
- `--resize=fast|balanced|quality`
- `--score=0.10`
- `--json`
