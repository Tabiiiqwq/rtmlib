#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用OpenCV重新编码视频为H.264格式（不依赖FFmpeg）
"""
import sys
import io
from pathlib import Path
import cv2
import subprocess

# 设置UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

source_dir = Path('output_viz')
output_dir = Path('output_viz_transformed')
backup_dir = Path('output_viz_backup')

def _estimate_target_bitrate(width: int, height: int, fps: float) -> int:
    """根据分辨率与帧率估算一个较高的码率，尽量减少可见损失."""
    # 经验：bits_per_pixel ≈ 0.1（高质量） -> bps = w*h*fps*0.1
    bpp = 0.10
    return int(width * height * fps * bpp)

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(['ffmpeg', '-version'],
                       capture_output=True,
                       text=True,
                       timeout=5)
        return True
    except Exception:
        return False

def _ffmpeg_transcode_to_h264(src: Path, dst: Path, fps: float) -> bool:
    """使用 ffmpeg 转码为 H.264 yuv420p faststart，尽量保留帧率。"""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(src),
        '-vcodec', 'libx264',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-r', f'{fps:.3f}',
        '-an',
        str(dst)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print('ffmpeg 转码失败：', result.stderr[:300])
            return False
        return True
    except subprocess.TimeoutExpired:
        print('ffmpeg 转码超时')
        return False
    except Exception as e:
        print('ffmpeg 调用异常：', e)
        return False


def convert_video_opencv(input_path, output_path):
    """使用OpenCV重新编码视频（保持原分辨率与帧率）"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False, "无法打开输入视频"
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_bitrate = _estimate_target_bitrate(width, height, fps)
    
    # 尝试不同的H.264编码器
    codecs_to_try = [
        ('H264', 'H264'),
        ('avc1', 'avc1'),
        ('X264', 'X264'),
        ('mp4v', 'mp4v'),  # 如果H264不可用，至少保持原格式
    ]
    
    writer = None
    codec_used = None
    
    for codec_name, fourcc_str in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if writer.isOpened():
            codec_used = codec_name
            print(f"  使用编码器: {codec_name}")
            # 尝试设置更高质量（不同后端可能不支持，忽略异常）
            try:
                if hasattr(cv2, 'VIDEOWRITER_PROP_QUALITY'):
                    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
                if hasattr(cv2, 'CAP_PROP_BITRATE'):
                    writer.set(cv2.CAP_PROP_BITRATE, target_bitrate)
            except Exception:
                pass
            break
        else:
            writer.release()
            writer = None
    
    if writer is None:
        cap.release()
        return False, "无法创建视频写入器，所有编码器都失败"
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            writer.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"  进度: {frame_count}/{total_frames} ({progress:.1f}%)", end='\r')
        
        print()  # 换行
        cap.release()
        writer.release()
        # 若未成功使用 H.264/avc1，且可用 ffmpeg，则二次转码为网页友好格式
        if codec_used not in ('H264', 'avc1') and _ffmpeg_available():
            h264_path = Path(output_path).with_name(f"{Path(output_path).stem}_h264.mp4")
            ok = _ffmpeg_transcode_to_h264(Path(output_path), h264_path, fps)
            if ok:
                try:
                    Path(output_path).unlink(missing_ok=True)
                    h264_path.rename(output_path)
                    return True, f"成功（OpenCV:{codec_used} → FFmpeg:H.264）"
                except Exception as e:
                    return False, f"转码后替换失败: {e}"
            else:
                return True, f"成功（编码器: {codec_used}），但未能转码为H.264"
        return True, f"成功，使用编码器: {codec_used}"
    except Exception as e:
        cap.release()
        if writer:
            writer.release()
        return False, str(e)

def main():
    if not source_dir.exists():
        print(f"❌ 源目录不存在: {source_dir.absolute()}")
        return
    
    print("=" * 60)
    print("视频编码转换工具 (OpenCV版本)")
    print("=" * 60)
    print(f"源目录: {source_dir.absolute()}")
    print(f"输出目录: {output_dir.absolute()}")
    print()
    
    # 查找所有mp4文件
    video_files = list(source_dir.glob('*.mp4'))
    if len(video_files) == 0:
        print("未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print()
    
    # 创建备份目录
    backup_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理: {video_file.name}")
        
        # 备份原文件
        backup_path = backup_dir / video_file.name
        try:
            import shutil
            if not backup_path.exists():  # 只备份一次
                shutil.copy2(video_file, backup_path)
        except Exception as e:
            print(f"  ⚠️  备份失败: {e}")
        
        # 转换视频 -> 输出到 output_viz_transformed
        temp_output = output_dir / f"{video_file.stem}_temp.mp4"
        final_output = output_dir / f"{video_file.stem}.mp4"
        success, message = convert_video_opencv(video_file, temp_output)
        
        if success:
            # 重命名为最终输出文件，不覆盖源文件
            try:
                if final_output.exists():
                    final_output.unlink()
                temp_output.rename(final_output)
                print(f"  ✓ {message} -> {final_output.name}")
                success_count += 1
            except Exception as e:
                print(f"  ❌ 替换文件失败: {e}")
                fail_count += 1
                if temp_output.exists():
                    temp_output.unlink()
        else:
            print(f"  ❌ 转换失败: {message}")
            fail_count += 1
            if temp_output.exists():
                temp_output.unlink()
        
        print()
    
    print("=" * 60)
    print(f"转换完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    print(f"备份文件保存在: {backup_dir.absolute()}")
    print("=" * 60)
    print("\n注意: 如果H264编码器不可用，视频可能仍使用mp4v编码。")
    print("      这种情况下，建议使用FFmpeg转换或重新生成视频。")

if __name__ == "__main__":
    main()

