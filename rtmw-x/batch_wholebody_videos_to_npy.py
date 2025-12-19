import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rtmlib import Wholebody

# 每个进程的模型缓存，避免重复加载
_WORKER_MODEL_CACHE = {}
_WORKER_MODEL_INIT = False


def get_worker_models(modes, device, backend, openpose_skeleton):
    """
    在单个进程内缓存 Wholebody 模型，避免重复加载
    """
    global _WORKER_MODEL_CACHE, _WORKER_MODEL_INIT

    cache_key = (device, backend, openpose_skeleton)
    cached = _WORKER_MODEL_CACHE.get(cache_key)

    if cached is None:
        cached = {}
        _WORKER_MODEL_CACHE[cache_key] = cached

    missing_modes = [m for m in modes if m not in cached]
    if missing_modes:
        if not _WORKER_MODEL_INIT:
            print(f"[进程 {os.getpid()}] 正在加载模型: {missing_modes}")
        for mode in missing_modes:
            cached[mode] = Wholebody(
                mode=mode,
                to_openpose=openpose_skeleton,
                backend=backend,
                device=device,
            )
        _WORKER_MODEL_INIT = True

    return cached

# 默认配置
DEFAULT_VIDEO_DIR = Path(r"Z:\DDDataLang\raw_data\How2Sign\test_rgb_front_clips")
DEFAULT_OUTPUT_DIR = Path(r"Z:\DDDataLang\raw_data\How2Sign\test_rgb_front_clips\output_train")
MODELS_TO_COMPARE = [
    "rtmw-x_256*192",
    "rtmw-x_384*288",
]


def run_one_video_one_model(video_path: Path, wholebody_model: Wholebody):
    """使用已加载的模型处理单个视频"""
    # 优化：使用更高效的视频读取方式
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    # 设置视频捕获属性以优化性能
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区，避免内存积累

    frames_data = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        keypoints, scores, bboxes, bbox_scores = wholebody_model(frame)

        # 处理多人情况：选择置信度最高的人（按 bbox_scores 排序）
        if len(keypoints) == 0:
            # 如果没有检测到人，创建一个全零的关键点数组
            # 假设是 133 个关键点（wholebody 标准）
            num_keypoints = 133
            frame_kpts = np.zeros((num_keypoints, 3), dtype=np.float64)
        else:
            # 如果有多个检测到的人，选择 bbox_scores 最高的
            if len(bbox_scores) > 0:
                best_person_idx = np.argmax(bbox_scores)
            else:
                best_person_idx = 0
            
            # keypoints 形状: (num_person, num_keypoints, 2)
            # scores 形状: (num_person, num_keypoints)
            # 合并为 (num_keypoints, 3) 格式: [x, y, score]
            kpts = keypoints[best_person_idx]  # (num_keypoints, 2)
            scrs = scores[best_person_idx]  # (num_keypoints,)
            frame_kpts = np.concatenate([kpts, scrs[:, np.newaxis]], axis=-1)  # (num_keypoints, 3)
        
        frames_data.append(frame_kpts)

    cap.release()

    if not frames_data:
        return None

    # 转换为 numpy 数组: (num_frames, num_keypoints, 3)
    result_array = np.array(frames_data, dtype=np.float64)

    return result_array


def check_network_path(path: Path) -> bool:
    """检查网络路径是否可访问"""
    try:
        # 尝试访问路径的父目录
        if path.is_absolute() and len(path.parts) > 0:
            # 检查根路径（如 Z:\）
            root = Path(path.parts[0])
            if root.exists():
                return True
        return path.exists() or path.parent.exists()
    except (OSError, PermissionError):
        return False


def gather_videos(video_dir: Path):
    # 检查网络路径是否可访问
    if not check_network_path(video_dir):
        print(f"[警告] 无法访问视频目录: {video_dir}")
        print("   可能原因:")
        print("   1. 网络驱动器 Z: 已断开连接（电脑息屏/睡眠可能导致）")
        print("   2. 网络连接不稳定")
        print("   3. 路径不存在")
        print("\n   解决方案:")
        print("   1. 检查网络驱动器是否已连接: 在文件资源管理器中查看 Z: 盘")
        print("   2. 如果断开，请重新连接网络驱动器")
        print("   3. 或者使用 --video-dir 参数指定可访问的路径")
        sys.exit(1)
    
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    videos = sorted(video_dir.rglob("*.mp4"))
    if not videos:
        print(f"目录内没有 mp4 视频: {video_dir}")
    return videos


def save_video_data(video_name: str, output_dir: Path, mode: str, data: np.ndarray):
    # 检查输出目录的父目录是否可访问
    if not check_network_path(output_dir.parent):
        print(f"[错误] 无法访问输出目录的父路径: {output_dir.parent}")
        print("   可能原因: 网络驱动器 Z: 已断开连接（电脑息屏/睡眠可能导致）")
        print("   请检查网络连接或重新连接网络驱动器后重试")
        raise FileNotFoundError(f"网络路径不可访问: {output_dir.parent}")
    
    try:
        # 使用 os.makedirs 创建目录，对网络路径更友好
        os.makedirs(str(output_dir), exist_ok=True)
    except (OSError, PermissionError) as e:
        error_msg = (
            f"无法创建输出目录: {output_dir}\n"
            f"错误: {e}\n\n"
            f"可能原因:\n"
            f"1. 网络驱动器 Z: 已断开连接（电脑息屏/睡眠可能导致）\n"
            f"2. 网络连接不稳定\n"
            f"3. 权限不足\n\n"
            f"解决方案:\n"
            f"1. 检查网络驱动器是否已连接\n"
            f"2. 如果断开，请重新连接网络驱动器\n"
            f"3. 或者使用 --output-dir 参数指定本地路径"
        )
        raise FileNotFoundError(error_msg)
    
    # 为每个模型创建单独的文件
    out_path = output_dir / f"{video_name}_{mode.replace('*', 'x').replace('-', '_')}.npy"
    try:
        np.save(out_path, data, allow_pickle=False)
        return True, str(out_path), data.shape
    except (OSError, PermissionError) as e:
        error_msg = (
            f"无法保存文件: {out_path}\n"
            f"错误: {e}\n\n"
            f"可能原因: 网络驱动器 Z: 在保存过程中断开连接\n"
            f"请检查网络连接后重试"
        )
        return False, str(out_path), None


def process_single_video(
    video_path_str: str,
    output_dir_str: str,
    modes: list,
    device: str,
    backend: str,
    openpose_skeleton: bool,
    worker_id: int = 0,
) -> Tuple[bool, str, Optional[Dict]]:
    """
    处理单个视频的所有模型（用于多进程）
    
    Returns:
        (success, video_name, results_dict)
    """
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    video_name = video_path.stem
    
    try:
        # 每个进程复用已加载的模型，避免重复加载
        try:
            models = get_worker_models(modes, device, backend, openpose_skeleton)
        except AttributeError as e:
            if "InferenceSession" in str(e) or "onnxruntime" in str(e).lower():
                error_msg = (
                    f"onnxruntime 模块问题: {str(e)}\n"
                    f"请尝试重新安装: pip install --upgrade onnxruntime"
                )
                return False, video_name, {"error": error_msg}
            else:
                raise
        except Exception as e:
            return False, video_name, {"error": f"模型加载失败: {str(e)}"}

        results = {}
        for mode in modes:
            result = run_one_video_one_model(video_path, models[mode])
            if result is None:
                results[mode] = {"success": False, "error": "推理失败或无有效帧"}
                continue
            
            # 保存结果
            success, out_path, shape = save_video_data(video_name, output_dir, mode, result)
            results[mode] = {
                "success": success,
                "out_path": out_path,
                "shape": shape,
            }
        
        return True, video_name, results
        
    except AttributeError as e:
        if "InferenceSession" in str(e) or "onnxruntime" in str(e).lower():
            error_msg = (
                f"onnxruntime 模块问题: {str(e)}\n"
                f"请尝试重新安装: pip install --upgrade onnxruntime"
            )
            return False, video_name, {"error": error_msg}
        else:
            return False, video_name, {"error": str(e)}
    except Exception as e:
        return False, video_name, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="批量跑 Wholebody 并将 keypoints 和 scores 保存为 npy")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="包含视频的文件夹")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备 (cpu/cuda)")
    parser.add_argument("--backend", type=str, default="onnxruntime", help="推理后端")
    parser.add_argument("--openpose-skeleton", action="store_true", help="是否使用 openpose skeleton 格式")
    parser.add_argument("--workers", type=int, default=None, help="并行处理的进程数 (默认: CPU核心数)")
    args = parser.parse_args()

    videos = gather_videos(args.video_dir)
    if not videos:
        return

    # 检查 CUDA 是否真的可用（对于 onnxruntime）
    actual_device = args.device
    if args.device == "cuda":
        try:
            import onnxruntime as ort
            # 检查 get_available_providers 方法是否存在
            if hasattr(ort, 'get_available_providers'):
                try:
                    available_providers = ort.get_available_providers()
                    if "CUDAExecutionProvider" not in available_providers:
                        print("[警告] CUDAExecutionProvider 不可用，将使用 CPU")
                        print(f"   可用的 providers: {available_providers}")
                        print("   如果需要使用 GPU，请安装: pip install onnxruntime-gpu")
                        actual_device = "cpu"
                    else:
                        print("[OK] CUDAExecutionProvider 可用")
                except (AttributeError, TypeError, Exception) as e:
                    print(f"警告: 无法检查 CUDA 可用性 ({e})，将使用 CPU")
                    actual_device = "cpu"
            else:
                print("警告: onnxruntime 版本过旧，无法检查 CUDA 支持，将使用 CPU")
                print("   建议升级: pip install --upgrade onnxruntime 或 onnxruntime-gpu")
                actual_device = "cpu"
        except ImportError:
            print("警告: 无法导入 onnxruntime，将使用 CPU")
            print("   如果使用 onnxruntime backend，请安装: pip install onnxruntime")
            actual_device = "cpu"
    
    # 确定工作进程数
    if args.workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = args.workers
    
    # 如果使用CUDA，限制进程数避免GPU内存不足
    if actual_device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                # 单GPU时，可以使用2-4个进程（onnxruntime 支持多进程）
                if num_gpus == 1:
                    # 对于 onnxruntime，可以适当增加进程数，但要注意显存
                    max_workers = min(4, num_workers)  # 最多4个进程
                    if args.workers is None:
                        num_workers = max_workers
                        print(f"[提示] 单GPU模式，使用 {num_workers} 个进程以提升吞吐量")
                    else:
                        num_workers = min(num_workers, max_workers)
                        print(f"[提示] 单GPU模式，限制最大进程数为 {num_workers}")
                else:
                    # 多GPU时，每个GPU最多2个进程
                    max_workers = num_gpus * 2
                    num_workers = min(num_workers, max_workers)
                    print(f"检测到 {num_gpus} 个GPU，限制最大进程数为 {num_workers}")
        except ImportError:
            # 如果没有torch，但onnxruntime支持CUDA，可以使用2-4个进程
            if actual_device == "cuda":
                if args.workers is None:
                    num_workers = min(4, multiprocessing.cpu_count())
                else:
                    num_workers = min(args.workers, 4)
                print(f"[提示] 无法检测GPU数量，使用 {num_workers} 个进程")
    
    # 如果使用CPU，可以根据CPU核心数调整，但不要太多
    if actual_device == "cpu":
        # CPU模式下，建议使用CPU核心数的一半，避免过度竞争
        num_workers = min(num_workers, max(1, num_workers // 2))
        print(f"CPU模式，使用 {num_workers} 个进程（避免过度竞争）")

    # 验证 onnxruntime 是否可用（如果使用 onnxruntime backend）
    if args.backend == "onnxruntime":
        try:
            import onnxruntime as ort
            # 检查 InferenceSession 是否存在
            if not hasattr(ort, 'InferenceSession'):
                print("❌ 错误: onnxruntime 模块缺少 InferenceSession")
                print("   请重新安装 onnxruntime:")
                print("   pip uninstall onnxruntime onnxruntime-gpu")
                print("   pip install onnxruntime")
                sys.exit(1)
            # 尝试创建一个简单的 session 来验证
            try:
                # 这里不实际创建 session，只检查模块是否完整
                pass
            except Exception as e:
                print(f"[警告] onnxruntime 可能有问题: {e}")
        except ImportError:
            print("❌ 错误: 无法导入 onnxruntime")
            print("   请安装: pip install onnxruntime")
            sys.exit(1)
        except Exception as e:
            print(f"[警告] 检查 onnxruntime 时出错: {e}")

    print(f"\n共找到 {len(videos)} 个视频，将依次处理: {args.video_dir}")
    print(f"实际使用设备: {actual_device}")
    print(f"并行进程数: {num_workers}")
    print(f"将使用 {len(MODELS_TO_COMPARE)} 个模型: {MODELS_TO_COMPARE}\n")
    
    start_time = time.time()
    processed_count = 0
    failed_count = 0
    
    # 准备参数
    video_paths_str = [str(v) for v in videos]
    output_dir_str = str(args.output_dir)
    
    # 创建进度条
    pbar = tqdm(
        total=len(videos),
        desc="处理视频",
        unit="视频",
        ncols=120,
        mininterval=0.1,  # 最小更新间隔（更频繁更新）
        maxinterval=1.0,  # 最大更新间隔
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        initial=0,  # 初始值
    )
    pbar.set_postfix_str("初始化中...")
    pbar.refresh()
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        tqdm.write("正在提交任务...")
        future_to_video = {
            executor.submit(
                process_single_video,
                video_path_str,
                output_dir_str,
                MODELS_TO_COMPARE,
                actual_device,  # 使用实际检测到的设备
                args.backend,
                args.openpose_skeleton,
                i % num_workers,
            ): video_path_str
            for i, video_path_str in enumerate(video_paths_str)
        }
        tqdm.write(f"已提交 {len(future_to_video)} 个任务，开始处理...")
        pbar.set_postfix_str("等待第一个任务完成...")
        pbar.refresh()
        
        # 处理完成的任务
        for future in as_completed(future_to_video):
            video_path_str = future_to_video[future]
            video_name = Path(video_path_str).name
            
            try:
                # 获取结果（这里会阻塞直到任务完成）
                success, video_name_result, results = future.result()
                processed_count += 1
                
                if success:
                    # 检查是否有失败的模式
                    mode_failed = False
                    for mode, result_info in results.items():
                        if not result_info.get("success"):
                            failed_count += 1
                            mode_failed = True
                    
                    # 更新进度条
                    elapsed = time.time() - start_time
                    success_count = processed_count - failed_count
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    
                    status = f"成功:{success_count} 失败:{failed_count}"
                    pbar.set_postfix_str(status)
                    pbar.update(1)
                    pbar.refresh()  # 强制刷新
                    
                    # 详细日志（可选，可以通过参数控制）
                    if mode_failed:
                        tqdm.write(f"[{processed_count}/{len(videos)}] [警告] {video_name} - 部分模型失败")
                        for mode, result_info in results.items():
                            if not result_info.get("success"):
                                tqdm.write(f"  [{mode}] {result_info.get('error', '处理失败')}")
                else:
                    failed_count += 1
                    elapsed = time.time() - start_time
                    success_count = processed_count - failed_count
                    rate = processed_count / elapsed if elapsed > 0 else 0

                    status = f"成功:{success_count} 失败:{failed_count}"
                    pbar.set_postfix_str(status)
                    pbar.update(1)
                    pbar.refresh()  # 强制刷新
                    tqdm.write(f"[{processed_count}/{len(videos)}] [失败] {video_name}: {results.get('error', '处理失败')}")

            except Exception as e:
                processed_count += 1
                failed_count += 1
                elapsed = time.time() - start_time
                success_count = processed_count - failed_count
                rate = processed_count / elapsed if elapsed > 0 else 0
                
                status = f"成功:{success_count} 失败:{failed_count}"
                pbar.set_postfix_str(status)
                pbar.update(1)
                pbar.refresh()  # 强制刷新
                tqdm.write(f"[{processed_count}/{len(videos)}] [失败] {video_name}: 异常 - {str(e)}")
    
    pbar.close()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"全部完成！")
    print(f"总视频数: {len(videos)}")
    print(f"成功: {processed_count - failed_count}, 失败: {failed_count}")
    print(f"总用时: {total_time/3600:.2f} 小时 ({total_time:.0f} 秒)")
    print(f"平均每个视频: {total_time/len(videos):.2f} 秒")
    print(f"处理速度: {len(videos)/total_time*3600:.1f} 视频/小时")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

