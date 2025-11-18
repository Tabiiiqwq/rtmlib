import os
import glob
from rtmlib import Wholebody

# 从我们修改好的核心脚本中，导入 run 函数
from get_2dpose_RTMW import run

# ====================================================================================
# --- 1. 用户配置区域 ---
# --- 请在这里修改为您自己的路径 ---
# ====================================================================================

# 输入路径：存放您所有待处理视频的文件夹
INPUT_VIDEO_DIRECTORY = "C:/Users/admin/Desktop/WechatVedio"

# 输出路径：用于保存所有处理结果的根文件夹
OUTPUT_DIRECTORY = "E:/rtmlib/batch"

# ====================================================================================

def batch_processor():
    """
    批量处理视频的主函数
    """
    print("Initializing Wholebody model once for batch processing (using GPU)...")
    # 只在最开始初始化一次模型，然后传递给每个run调用，避免重复加载
    wholebody_model = Wholebody(
        to_openpose=False,
        mode='balanced',
        backend='onnxruntime',
        device='cuda'  # <--- 修改点 1
    )
    print("Model initialized.")

    # 查找所有支持的视频格式
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv']
    video_list = []
    for ext in video_extensions:
        video_list.extend(glob.glob(os.path.join(INPUT_VIDEO_DIRECTORY, '**', ext), recursive=True))

    if not video_list:
        print(f"Error: No video files found in {INPUT_VIDEO_DIRECTORY}.")
        return

    print(f"Found {len(video_list)} videos to process.")

    # 遍历并处理每个视频
    for video_path in video_list:
        print(f"\n--- Processing video: {video_path} ---")

        # 为每个视频创建独立的输出文件夹
        video_filename = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_filename)[0]
        video_output_folder = os.path.join(OUTPUT_DIRECTORY, video_name_without_ext)
        os.makedirs(video_output_folder, exist_ok=True)
        print(f"Results will be saved in: {video_output_folder}")

        # 调用核心脚本的run函数进行处理
        run(
            video_path=video_path,
            output_dir=video_output_folder,
            device='cuda',  # <--- 修改点 2
            vis=True,
            save_mode='json',
            wholebody_model=wholebody_model  # 传入已经初始化好的模型
        )

    print("\n--- All videos processed successfully! ---")


if __name__ == "__main__":
    batch_processor()