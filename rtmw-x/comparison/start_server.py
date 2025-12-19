#!/usr/bin/env python3
"""
简单的HTTP服务器，用于运行视频对比网页
"""
import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # 添加CORS头，允许跨域请求
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS, HEAD')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # 添加视频文件的Content-Type
        if self.path.endswith('.mp4'):
            self.send_header('Content-Type', 'video/mp4')
        elif self.path.endswith('.json'):
            self.send_header('Content-Type', 'application/json')
        super().end_headers()

    def log_message(self, format, *args):
        # 输出请求日志，方便调试
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_HEAD(self):
        # 支持HEAD请求用于检测文件是否存在
        f = self.send_head()
        if f:
            f.close()

def main():
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/video_comparison.html"
        print("=" * 60)
        print(f"🚀 服务器已启动!")
        print(f"📺 打开浏览器访问: {url}")
        print("=" * 60)
        print(f"按 Ctrl+C 停止服务器")
        print("=" * 60)
        
        # 自动打开浏览器
        try:
            webbrowser.open(url)
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n服务器已停止")

if __name__ == "__main__":
    main()

