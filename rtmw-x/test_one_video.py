import time

import cv2

from rtmlib import PoseTracker, Wholebody, draw_skeleton

import numpy as np

# import numpy as np

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

input_path = r"C:\Users\32529\Downloads\11月12日(1).mp4"
output_path = r"C:\Users\32529\Downloads\11月12日(1)_vis.mp4"

cap = cv2.VideoCapture(input_path)
openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = PoseTracker(
    Wholebody,
    det_frequency=7,
    to_openpose=openpose_skeleton,
    mode='balanced',  # balanced, performance, lightweight
    backend=backend,
    device=device)

frame_idx = 0
writer = None
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()
    keypoints, scores = wholebody(frame)
    det_time = time.time() - s
    print('det: ', det_time)

    img_show = frame.copy()

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)
    # img_show = np.full(img_show.shape, 255, dtype=np.uint8)
    # print(scores)
    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.5)

    if writer is None:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器: {output_path}")

    writer.write(img_show)
    cv2.imshow('img', img_show)
    cv2.waitKey(10)

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
