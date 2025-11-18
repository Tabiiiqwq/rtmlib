import os
import cv2
import json
import numpy as np
import fire
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict

from rtmlib import Wholebody, draw_skeleton


def process_single_frame(frame, wholebody_model):
    """处理单帧图像，返回关键点和边界框"""
    keypoints, scores_raw, bbox, bbox_scores_raw = wholebody_model(frame)

    if bbox is None or len(bbox) == 0:
        return None, None, None, None

    bbox = np.asarray(bbox)
    if len(bbox) > 1:
        widths = np.maximum(0, bbox[:, 2] - bbox[:, 0])
        heights = np.maximum(0, bbox[:, 3] - bbox[:, 1])
        areas = widths * heights
        max_idx = int(np.argmax(areas))
        keypoints, scores_raw, bbox, bbox_scores_raw = keypoints[[max_idx]], scores_raw[[max_idx]], bbox[[max_idx]], bbox_scores_raw[[max_idx]]
    
    return keypoints, scores_raw, bbox, bbox_scores_raw


def get_bbox_from_keypoints(keypoints: np.ndarray, score_thr: float = 0.5) -> List[int]:
    assert keypoints.shape == (133, 3)
    valid_kpts = keypoints[keypoints[:, 2] > score_thr]
    if valid_kpts.shape[0] > 0:
        x_min, y_min = int(np.min(valid_kpts[:, 0])), int(np.min(valid_kpts[:, 1]))
        x_max, y_max = int(np.max(valid_kpts[:, 0])), int(np.max(valid_kpts[:, 1]))
        return [x_min, y_min, x_max, y_max]
    return [0, 0, 0, 0]


def run(
    video_path: str = "./test_video.mp4",
    output_dir: str = "./output/RTMW",
    extract_mode: str = "balanced",
    save_mode: str = "json",
    device: str = "cpu",
    vis: bool = True,
    wholebody_model=None,
):
    if wholebody_model is None:
        print(f"Initializing model with device: {device}...")
        wholebody_model = Wholebody(
            to_openpose=False, mode=extract_mode, backend="onnxruntime", device=device
        )
    
    vis_folder = os.path.join(output_dir, "visualized_frames")
    os.makedirs(vis_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    all_frames_data = []

    # 使用tqdm创建进度条
    with tqdm(total=total_frames, desc="Processing video frame-by-frame") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores_raw, bbox, bbox_scores_raw = process_single_frame(frame, wholebody_model)

            frame_results = []
            if keypoints is not None:
                scores = scores_raw[:, :, np.newaxis]
                out_data = np.concatenate([keypoints, scores], axis=-1)[0]
                
                bbox_out = np.concatenate([bbox, bbox_scores_raw[:, np.newaxis]], axis=-1)
                bbox_conf = bbox_out[0, 4] if bbox_out.shape[0] > 0 else 0.0

                person_dict = {
                    "personID": 0,
                    "video_resolution": [W, H],
                    "bbox": get_bbox_from_keypoints(out_data, score_thr=0.5),
                    "bbox_confidence": bbox_conf,
                    "keypoints": out_data.tolist(),
                    "isKeyFrame": False,
                }
                frame_results.append(person_dict)
                
                if vis:
                    vis_img = draw_skeleton(frame, keypoints, scores_raw, kpt_thr=0.5, radius=2, line_width=2)
                    vis_out_path = os.path.join(vis_folder, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(vis_out_path, vis_img)
            elif vis: # 如果没有检测到目标，也保存原图
                vis_out_path = os.path.join(vis_folder, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(vis_out_path, frame)

            all_frames_data.append(frame_results)
            pbar.update(1)
            frame_idx += 1
            
    cap.release()

    if save_mode == "json":
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_out_path = os.path.join(output_dir, f"{video_name}.json")
        with open(json_out_path, "w", encoding='utf-8') as f:
            json.dump(all_frames_data, f, indent=4)
        print(f"JSON results saved to: {json_out_path}")


if __name__ == "__main__":
    fire.Fire(run)