#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将现有的mp4v编码视频转换为H.264编码，以便浏览器播放
"""
import sys
import io
from pathlib import Path
import subprocess

# 设置UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

output_dir = Path('output_viz')
backup_dir = Path('output_viz_backup')

def check_ffmpeg():
    """检查FFmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def convert_video(input_path, output_path, quality='high'):
    """使用FFmpeg转换视频为H.264编码
    
    Args:
        quality: 'high' (CRF 18, 接近无损), 'medium' (CRF 23, 平衡), 'low' (CRF 28, 较小文件)
    """
    # 根据质量选择CRF值
    crf_map = {
        'high': '18',    # 接近无损，文件较大
        'medium': '23',  # 平衡质量和文件大小
        'low': '28'      # 较小文件，质量较低
    }
    crf = crf_map.get(quality, '18')
    
    # 使用高质量FFmpeg命令
    # 先尝试使用高质量设置，如果失败则使用更简单的命令
    cmd_high_quality = [
        'ffmpeg',
        '-i', str(input_path),
        '-vcodec', 'libx264',
        '-crf', crf,  # 高质量
        '-preset', 'slow',  # 慢速编码以获得更好质量（如果支持）
        '-pix_fmt', 'yuv420p',
        '-acodec', 'copy',
        '-y',
        str(output_path)
    ]
    
    cmd_simple = [
        'ffmpeg',
        '-i', str(input_path),
        '-vcodec', 'libx264',
        '-crf', crf,  # 使用指定的质量
        '-pix_fmt', 'yuv420p',
        '-acodec', 'copy',
        '-y',
        str(output_path)
    ]
    
    # 如果简单命令失败，尝试更基础的命令
    cmd_basic = [
        'ffmpeg',
        '-i', str(input_path),
        '-vcodec', 'libx264',
        '-b:v', '5M',  # 使用固定码率5Mbps保证质量
        '-pix_fmt', 'yuv420p',
        '-y',
        str(output_path)
    ]
    
    # 按优先级尝试：高质量 -> 简单 -> 基础
    for cmd in [cmd_high_quality, cmd_simple, cmd_basic]:
        try:
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=300)  # 5分钟超时
            
            if result.returncode == 0:
                return True, None
            else:
                # 如果当前命令失败，尝试下一个
                if cmd != cmd_basic:  # 不是最后一个命令，继续尝试
                    continue
                else:
                    return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "转换超时"
        except Exception as e:
            if cmd != cmd_basic:  # 不是最后一个命令，继续尝试
                continue
            else:
                return False, str(e)
    
    return False, "所有转换方法都失败"

def main():
    import argparse
    parser = argparse.ArgumentParser(description='转换视频为H.264编码')
    parser.add_argument('--quality', choices=['high', 'medium', 'low'], 
                       default='high',
                       help='视频质量: high(CRF18,接近无损), medium(CRF23,平衡), low(CRF28,较小文件)')
    args = parser.parse_args()
    
    if not output_dir.exists():
        print(f"❌ 目录不存在: {output_dir.absolute()}")
        return
    
    # 检查FFmpeg
    if not check_ffmpeg():
        print("❌ 未找到FFmpeg，请先安装FFmpeg")
        print("   下载地址: https://ffmpeg.org/download.html")
        print("   或使用: conda install ffmpeg")
        return
    
    crf_value = {'high': '18', 'medium': '23', 'low': '28'}[args.quality]
    print("=" * 60)
    print("视频编码转换工具")
    print("=" * 60)
    print(f"源目录: {output_dir.absolute()}")
    print(f"质量设置: {args.quality} (CRF: {crf_value})")
    print()
    
    # 查找所有mp4文件
    video_files = list(output_dir.glob('*.mp4'))
    if len(video_files) == 0:
        print("未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print()
    
    # 创建备份目录
    backup_dir.mkdir(exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理: {video_file.name}")
        
        # 备份原文件
        backup_path = backup_dir / video_file.name
        try:
            import shutil
            shutil.copy2(video_file, backup_path)
        except Exception as e:
            print(f"  ⚠️  备份失败: {e}")
        
        # 转换视频
        temp_output = video_file.parent / f"{video_file.stem}_temp.mp4"
        success, error = convert_video(video_file, temp_output, quality=args.quality)
        
        if success:
            # 替换原文件
            try:
                video_file.unlink()  # 删除原文件
                temp_output.rename(video_file)  # 重命名临时文件
                print(f"  ✓ 转换成功")
                success_count += 1
            except Exception as e:
                print(f"  ❌ 替换文件失败: {e}")
                fail_count += 1
        else:
            print(f"  ❌ 转换失败: {error}")
            fail_count += 1
            # 删除临时文件
            if temp_output.exists():
                temp_output.unlink()
        
        print()
    
    print("=" * 60)
    print(f"转换完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    print(f"备份文件保存在: {backup_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()

