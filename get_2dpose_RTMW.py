import os
import glob
import cv2
import json
import numpy as np
import fire

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict

from rtmlib import Wholebody, draw_skeleton


def process_images(
    images: List[np.ndarray], wholebody_model, output_dir: str, vis: bool = True
) -> np.ndarray:
    """
    Process a list of images and return raw model outputs.

    Args:
        images: List of images as numpy arrays
        wholebody_model: Initialized Wholebody model
        output_dir: Directory to save results
        vis: Whether to save visualization results

    Returns:
        List of dictionaries containing keypoints data for each image
    """
    results = []

    for frame_idx, image in enumerate(tqdm(images, desc="Processing images")):
        keypoints, scores_raw = wholebody_model(image)

        scores = scores_raw[:, :, np.newaxis]  # (num_person, 133, 1)
        out_data = np.concatenate([keypoints, scores], axis=-1)  # (num_person, 133, 3)

        if vis:
            vis_out = draw_skeleton(
                image, keypoints, scores_raw, kpt_thr=0.5, radius=1, line_width=1
            )
            vis_out_path = os.path.join(
                output_dir, "vis_RTMW", f"pose_{frame_idx:05d}.jpg"
            )
            Path(os.path.join(output_dir, "vis_RTMW")).mkdir(
                parents=True, exist_ok=True
            )
            cv2.imwrite(vis_out_path, vis_out)

        results.append(out_data)

    results = np.array(results)  # Convert list to numpy array, (frame, person, 133, 3)
    return results


def get_bbox_from_keypoints(
    keypoints: np.ndarray, score_thr: float = 0.5
) -> List[int]:  # xyxy
    assert keypoints.shape == (133, 3)
    valid_kpts = keypoints[keypoints[:, 2] > score_thr]
    if valid_kpts.shape[0] > 0:
        x_min = int(np.min(valid_kpts[:, 0]))
        y_min = int(np.min(valid_kpts[:, 1]))
        x_max = int(np.max(valid_kpts[:, 0]))
        y_max = int(np.max(valid_kpts[:, 1]))
        bbox = [x_min, y_min, x_max, y_max]
    else:
        bbox = [0, 0, 0, 0]  # dummy bbox
    return bbox


def get_npy_results(results: np.ndarray, images: List[np.ndarray]) -> np.array:
    # get video resolution
    H, W = images[0].shape[:2]
    video_res_info = np.array([W, H, 1], dtype=np.int32)  # (3,)
    video_res_info = np.tile(video_res_info, (results.shape[0], 1, 1))  # (frame, 1, 3)

    npy_results = results[:, 0, :, :]  # (frame, 133, 3)
    npy_results = np.concatenate(
        [npy_results, video_res_info], axis=1
    )  # (frame, 134, 3)
    return npy_results


def get_json_results(results: np.ndarray) -> List:  # return (frame, person, dict)
    json_results = []
    for frame in results:
        frame_results = []
        for i, person in enumerate(frame):
            bbox = get_bbox_from_keypoints(person, score_thr=0.5)  # xyxy
            person_dict = {
                "personID": i,
                "bbox": bbox,  # xyxy
                "keypoints": person.tolist(),  # (133, 3)
                "isKeyFrame": False,
            }
            frame_results.append(person_dict)
        json_results.append(frame_results)
    return json_results


def main(
    video_path: str = "./test_video.mp4",  # video file path
    output_dir: str = "./output/RTMW",
    extract_mode: str = "balanced",  # 'performance', 'lightweight', 'balanced'
    save_mode: str = "json",  # 'json', 'npy': json follow red_output.json, npy follow trainning data format
    device: str = "cuda",  # cpu, cuda, mps
    vis: bool = False,
):
    backend = "onnxruntime"  # opencv, onnxruntime, openvino
    openpose_skeleton = False  # True for openpose-style, False for mmpose-style
    wholebody = Wholebody(
        to_openpose=openpose_skeleton,
        mode=extract_mode,  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
        backend=backend,
        device=device,
    )

    video_extension = os.path.splitext(video_path)[1]  # get file extension

    if os.path.isfile(video_path):
        # Process video file
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read all frames into a list
        images = []
        print(f"Reading {total_frames} frames from video...")
        with tqdm(total=total_frames, desc="Reading video frames") as pbar:
            while True:
                ret, image = cap.read()
                if not ret:
                    break
                images.append(image)
                pbar.update(1)

        cap.release()

        # Process all images using the extracted function
        results = process_images(images, wholebody, output_dir, vis)
        if save_mode == "npy":
            npy_out_path = os.path.join(
                output_dir,
                os.path.basename(video_path).replace(video_extension, ".npy"),
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            npy_results = get_npy_results(results, images)

            np.save(npy_out_path, npy_results)
        elif save_mode == "json":
            json_out_path = os.path.join(
                output_dir,
                os.path.basename(video_path).replace(video_extension, ".json"),
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            json_results = get_json_results(results)
            with open(json_out_path, "w") as f:
                json.dump(json_results, f)


if __name__ == "__main__":
    fire.Fire(main)
