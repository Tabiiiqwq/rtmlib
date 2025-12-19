# RTMW-X Toolkit

Utilities and assets that simplify experimenting with RTMW-X pose models live in this folder. Key use cases include running large batches of videos through multiple RTMW-X checkpoints and hosting quick visual comparisons.

## Folder Layout

- `batch_wholebody_videos_to_npy.py` &mdash; multiprocess pipeline that runs `rtmlib.Wholebody` with both `rtmw-x_256*192` and `rtmw-x_384*288`, then saves per-frame keypoints to `.npy` files.
- `convert_videos_opencv.py` / `convert_videos_to_h264.py` &mdash; helper scripts for re-encoding raw clips before pose extraction.
- `comparison/` &mdash; static web demo (`start_server.py`, `video_comparison.html`) for side-by-side inspection of model outputs.
- `mmdeploy_output/rtmw-x_*` &mdash; exported ONNX bundles (`end2end.onnx`, `deploy.json`, etc.) consumed by `rtmlib.Wholebody` when the custom modes are selected.
- `start_server.py`, `test_one_video.py`, `video_comparison.html` &mdash; small utilities for ad-hoc debugging and visualization.

## Requirements

- Python 3.8+
- Dependencies listed in the root `rtmlib/requirements.txt` (install with `pip install -r requirements.txt`).
- Valid RTMW-X ONNX models placed under `mmdeploy_output/rtmw-x_256x192` and `mmdeploy_output/rtmw-x_384x288`.
- GPU acceleration is optional but recommended (`onnxruntime-gpu` or compatible backend).

## Batch Processing

Run the multi-model extractor from this directory (paths below are examples; override with CLI arguments as needed):

```bash
cd rtmw-x
python batch_wholebody_videos_to_npy.py \
  --video-dir Z:\DDDataLang\raw_data\How2Sign\test_rgb_front_clips \
  --output-dir Z:\DDDataLang\raw_data\How2Sign\test_rgb_front_clips\output_train \
  --device cuda \
  --backend onnxruntime \
  --workers 4
```

The script caches RTMW-X models per worker process, selects the highest-confidence person per frame, and emits arrays shaped `(num_frames, 133, 3)` (`x, y, score`). When network shares are unavailable, provide alternative `--video-dir` / `--output-dir` pointing to local disks.

## Tips

- To inspect output fidelity quickly, copy selected `.npy` files into `comparison/` and launch the local server there.
- When testing new ONNX exports, drop them into the corresponding `mmdeploy_output/rtmw-x_*` subfolder; `Wholebody.MODE` resolves the paths automatically, so no code changes are required.
- If you need custom thresholds or extra post-processing, extend `batch_wholebody_videos_to_npy.py` and keep the modifications in this folder to avoid impacting the packaged `rtmlib` library.

