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
