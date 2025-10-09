import os
import glob
import cv2
import json
import numpy as np
import threading
from queue import Queue
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .solution import Wholebody
from ..visualization.draw import draw_skeleton, draw_bbox


class RTMWRunner:
    """
    RTMW 2D pose estimation runner for video processing.
    
    This class provides functionality to process videos and extract 2D pose keypoints
    using the RTMW (Real-Time Multi-person Wholebody) model.
    """
    
    def __init__(
        self,
        extract_mode: str = "balanced",
        backend: str = "onnxruntime", 
        device: str = "cuda",
        openpose_skeleton: bool = False
    ):
        """
        Initialize the RTMW runner.
        
        Args:
            extract_mode: Model mode - 'performance', 'lightweight', 'balanced'
            backend: Backend to use - 'opencv', 'onnxruntime', 'openvino'
            device: Device to run on - 'cpu', 'cuda', 'mps'
            openpose_skeleton: True for openpose-style, False for mmpose-style
        """
        self.wholebody = Wholebody(
            to_openpose=openpose_skeleton,
            mode=extract_mode,
            backend=backend,
            device=device,
        )
        
    def process_images(
        self, 
        images: List[np.ndarray], 
        output_dir: str, 
        vis: bool = True
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Process a list of images and return raw model outputs.

        Args:
            images: List of images as numpy arrays
            output_dir: Directory to save results
            vis: Whether to save visualization results

        Returns:
            Tuple of (results, bbox_out) containing keypoints data and bounding boxes
        """
        results = []
        bbox_out = np.array([]).reshape(0, 5)

        for frame_idx, image in enumerate(tqdm(images, desc="Processing images")):
            keypoints, scores_raw, bbox, bbox_scores_raw = self.wholebody(image)

            scores = scores_raw[:, :, np.newaxis]
            out_data = np.concatenate([keypoints, scores], axis=-1)

            if len(bbox) == 0 and (len(bbox_scores_raw) == 0):
                bbox = np.array([0, 0, 0, 0]).reshape(1, 4)
                bbox_scores_raw = np.array([0.0]).reshape(1,)

            bbox_out = np.concatenate([bbox, bbox_scores_raw[:, np.newaxis]], axis=-1)
            
            # Sort by bbox confidence score (descending order)
            if len(bbox_out) > 0:
                sort_indices = np.argsort(bbox_scores_raw)[::-1]
                bbox_out = bbox_out[sort_indices]
                out_data = out_data[sort_indices]

            if vis:
                vis_bbox = draw_bbox(image, bbox_out)
                
                vis_out = draw_skeleton(
                    vis_bbox, keypoints, scores_raw, kpt_thr=0.5, radius=1, line_width=1
                )
                vis_out_path = os.path.join(
                    output_dir, "vis_RTMW", f"pose_{frame_idx:05d}.jpg"
                )
                Path(os.path.join(output_dir, "vis_RTMW")).mkdir(
                    parents=True, exist_ok=True
                )
                cv2.imwrite(vis_out_path, vis_out)

            results.append(out_data)

        return results, bbox_out

    def read_frames_threaded(self, video_path: str, max_workers: int = 4) -> List[np.ndarray]:
        """
        Read video frames using threading for better performance.
        
        Args:
            video_path: Path to the video file
            max_workers: Number of worker threads (currently not used)
            
        Returns:
            List of video frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_queue = Queue(maxsize=100)
        images = []
        
        def frame_reader():
            while True:
                ret, frame = cap.read()
                if not ret:
                    frame_queue.put(None)
                    break
                frame_queue.put(frame.copy())
        
        reader_thread = threading.Thread(target=frame_reader)
        reader_thread.start()
        
        print(f"Reading {total_frames} frames from video...")
        with tqdm(total=total_frames, desc="Reading video frames") as pbar:
            while True:
                frame = frame_queue.get()
                if frame is None:
                    break
                images.append(frame)
                pbar.update(1)
        
        reader_thread.join()
        cap.release()
        return images

    def get_bbox_from_keypoints(
        self, 
        keypoints: np.ndarray, 
        score_thr: float = 0.5
    ) -> List[int]:
        """
        Extract bounding box from keypoints.
        
        Args:
            keypoints: Keypoints array of shape (133, 3)
            score_thr: Score threshold for valid keypoints
            
        Returns:
            Bounding box in xyxy format
        """
        assert keypoints.shape == (133, 3)
        valid_kpts = keypoints[keypoints[:, 2] > score_thr]
        if valid_kpts.shape[0] > 0:
            x_min = int(np.min(valid_kpts[:, 0]))
            y_min = int(np.min(valid_kpts[:, 1]))
            x_max = int(np.max(valid_kpts[:, 0]))
            y_max = int(np.max(valid_kpts[:, 1]))
            bbox = [x_min, y_min, x_max, y_max]
        else:
            bbox = [0, 0, 0, 0]
        return bbox

    def get_npy_results(
        self, 
        results: List[np.ndarray], 
        images: List[np.ndarray]
    ) -> np.ndarray:
        """
        Convert results to numpy format for training data.
        
        Args:
            results: List of pose estimation results
            images: List of input images
            
        Returns:
            Numpy array of shape (frame, 134, 3)
        """
        if not images:
            return np.array([]).reshape(0, 134, 3)
            
        H, W = images[0].shape[:2]
        video_res_info = np.array([W, H, 1], dtype=np.int32)
        video_res_info = np.tile(video_res_info, (len(results), 1, 1))

        npy_results = np.array([result[0] for result in results])  # Take first person
        npy_results = np.concatenate([npy_results, video_res_info], axis=1)
        return npy_results

    def get_json_results(
        self, 
        results: List[np.ndarray], 
        images: List[np.ndarray], 
        bboxes_raw: np.ndarray
    ) -> List:
        """
        Convert results to JSON format.
        
        Args:
            results: List of pose estimation results
            images: List of input images
            bboxes_raw: Raw bounding box data
            
        Returns:
            List of JSON-formatted results
        """
        json_results = []
        if not images:
            return json_results
            
        H, W = images[0].shape[:2]
        for frame in results:
            frame_results = []
            for i, person in enumerate(frame):
                bbox = self.get_bbox_from_keypoints(person, score_thr=0.5)
                person_dict = {
                    "personID": i,
                    "video_resolution": [W, H],
                    "bbox": bbox,
                    "bbox_confidence": bboxes_raw[i][4] if len(bboxes_raw) > i else 0.0,
                    "keypoints": person.tolist(),
                    "isKeyFrame": False,
                }
                frame_results.append(person_dict)
            json_results.append(frame_results)
        return json_results

    def run_from_json(self, video_path_json: str):
        """
        Process videos listed in a JSON file.
        
        Args:
            video_path_json: Path to JSON file containing video paths
        """
        with open(video_path_json, "r") as f:
            video_list = json.load(f)

        for video_path in tqdm(video_list):
            output_path = video_path.replace('videos', 'json_rtwm').replace('.mp4', '.json')
            if os.path.exists(output_path):
                print(f"Skip existing: {output_path}")
                continue
            print(video_path, '->', output_path)
            self.run(video_path=video_path, output_dir=os.path.dirname(output_path))

    def run(
        self,
        video_path: str = "./test_video.mp4",
        output_dir: str = "./output/RTMW",
        save_mode: str = "json",
        vis: bool = False,
    ):
        """
        Run pose estimation on video(s).
        
        Args:
            video_path: Path to video file or directory containing videos
            output_dir: Directory to save output files
            save_mode: Output format - 'json' or 'npy'
            vis: Whether to save visualization images
        """
        video_extension = os.path.splitext(video_path)[1]

        if os.path.isfile(video_path):
            # Process single video file
            images = self.read_frames_threaded(video_path)
            results, bboxes = self.process_images(images, output_dir, vis)
            
            if save_mode == "npy":
                npy_out_path = os.path.join(
                    output_dir,
                    os.path.basename(video_path).replace(video_extension, ".npy"),
                )
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                npy_results = self.get_npy_results(results, images)
                np.save(npy_out_path, npy_results)
                
            elif save_mode == "json":
                json_out_path = os.path.join(
                    output_dir,
                    os.path.basename(video_path).replace(video_extension, ".json"),
                )
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                json_results = self.get_json_results(results, images, bboxes)
                with open(json_out_path, "w") as f:
                    json.dump(json_results, f)
                    
        elif os.path.isdir(video_path):
            # Process all video files in directory
            video_list = glob.glob(os.path.join(video_path, "**", "*.mp4"), recursive=True)
            print(f"Found {len(video_list)} video files in {video_path}")

            for video_file in video_list:
                images = self.read_frames_threaded(video_file)
                results, bboxes = self.process_images(images, output_dir, vis)
                
                if save_mode == "npy":
                    npy_out_path = os.path.join(
                        output_dir,
                        os.path.basename(video_file).replace(".mp4", ".npy"),
                    )
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    npy_results = self.get_npy_results(results, images)
                    np.save(npy_out_path, npy_results)
                    
                elif save_mode == "json":
                    json_out_path = os.path.join(
                        output_dir,
                        os.path.basename(video_file).replace(".mp4", ".json"),
                    )
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    json_results = self.get_json_results(results, images, bboxes)
                    with open(json_out_path, "w") as f:
                        json.dump(json_results, f)


# Convenience function for backward compatibility
def run_rtmw_2d_pose(
    video_path: str = "./test_video.mp4",
    output_dir: str = "./output/RTMW", 
    extract_mode: str = "balanced",
    save_mode: str = "json",
    device: str = "cuda",
    vis: bool = False,
    backend: str = "onnxruntime",
    openpose_skeleton: bool = False
):
    """
    Convenience function to run RTMW 2D pose estimation.
    
    Args:
        video_path: Path to video file or directory
        output_dir: Output directory for results
        extract_mode: Model mode - 'performance', 'lightweight', 'balanced'
        save_mode: Output format - 'json' or 'npy'
        device: Device to run on - 'cpu', 'cuda', 'mps'
        vis: Whether to save visualization images
        backend: Backend to use - 'opencv', 'onnxruntime', 'openvino'
        openpose_skeleton: True for openpose-style, False for mmpose-style
    """
    runner = RTMWRunner(
        extract_mode=extract_mode,
        backend=backend,
        device=device,
        openpose_skeleton=openpose_skeleton
    )
    
    runner.run(
        video_path=video_path,
        output_dir=output_dir,
        save_mode=save_mode,
        vis=vis
    )