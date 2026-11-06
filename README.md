# AVTR-1 Offline Avatar Video (prebuilt engines)

Generate a talking-avatar video from an audio file and a face image using the
[avaturn-live/avtr-1](https://github.com/avaturn-live/avtr-1) engine, reusing
**prebuilt TensorRT engines** so you skip the slow `build-*-engines` step.

## What this notebook does
1. Clones the repo and installs the runtime environment (pixi).
2. Logs into HuggingFace and downloads all non-engine artifacts
   (reference frames, backgrounds, onnx fallbacks, ditto face models, warp plugin).
3. Copies your prebuilt `.engine` files from a Kaggle dataset into place.
4. Regenerates `avtr1_normalizer.safetensors` from the scripted checkpoint.
5. Runs `generate_offline` to produce the MP4.

## Requirements
- **Same Kaggle GPU accelerator** you used when the engines were built. TRT engines
  only deserialize on the matching GPU architecture + TensorRT version. If you built
  on T4, run on T4; mismatched hardware fails with a deserialization error.
- A HuggingFace token with access to the `avaturn-live/avtr-1` repo.
- A Kaggle dataset containing your 7 prebuilt engines.

## Engine dataset layout
Upload the engines (the `avtr1_production_engines.zip` from the build notebook) as a
Kaggle dataset. Kaggle auto-extracts it; any folder layout works because Cell 5
searches recursively. The 7 files it looks for:

Tip: also drop `avtr1_normalizer.safetensors` into the dataset so you can skip the
normalizer step on future runs (it is tiny and GPU-independent).

## Cell order
| Cell | Purpose |
|------|---------|
| 1 | Clone repo + `pixi install` |
| 2 | HuggingFace login |
| 3 | Install `onnxruntime-gpu` |
| 4 | `download_artifacts.py` (base artifacts, NOT engines) |
| 4b/4c | Build `avtr1_normalizer.safetensors` from the scripted checkpoint |
| 5 | Copy prebuilt engines from the dataset into the `*_cc` folders |
| 6 | Verify engines + warp plugin are present |
| 7 | Set your audio + add your avatar face as a reference frame |
| 8 | Run `generate_offline` |
| 9 | Preview + download the MP4 |

## Passing your own inputs
- **Audio -> `--speech`.** Point it at any file on the machine (wav / mp3 / ogg):
  `--speech /kaggle/input/<your-dataset>/your_audio.wav`
- **Image -> reference frame + `--avatar`.** You do not pass the image path on the
  command line. Convert it to a 1920x1080 RGBA PNG, save it as
  `artifacts/main/avatars_artifacts/reference_frames/<NAME>.png`, then call
  `--avatar <NAME>` (filename without `.png`). The PIL block in Cell 7 does this.
- **Background -> `--bg`.** `plain_white` uses the bundled `plain_white.png`.

Example:
pixi run generate_offline
--speech /kaggle/working/speech.mp3
--avatar myavatar
--bg plain_white
--out /kaggle/working/avatar_out.mp4


## Why the normalizer step exists
`avtr1_normalizer.safetensors` is **built locally and never uploaded to HuggingFace**,
so `download_artifacts.py` cannot fetch it and an engines-only dataset omits it. Cells
4b/4c rebuild it from `build_artifacts/avtr1.scripted.pt` (which Cell 4 downloads),
loading the scripted checkpoint and saving 9 buffers. No TRT rebuild needed.

## Troubleshooting
- **`FileNotFoundError: .../avtr1_normalizer.safetensors`** — run Cells 4b/4c, then
  re-run generate. If 4c says the scripted checkpoint is missing, re-run Cell 4 first.
- **Engine deserialization / version error on generate** — the engines were built on a
  different GPU or TensorRT version. Switch the Kaggle accelerator to match, or rebuild.
- **Cell 5 reports MISSING** — run `!find /kaggle/input -name "*.engine"` and set
  `ENGINES_DATASET` to the folder shown.
- **Missing `libgrid_sample_3d_plugin.so`** — Cell 6 materializes it via the artifact
  manager; if it is 0 bytes, re-run Cell 4.
