#!/usr/bin/env python3
"""PartnerFM local server — serves static files, persists state, proxies LLM calls."""

import json
import os
import re
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, '.partnerfm-state.json')
MODELS_FILE = os.path.join(BASE_DIR, '.partnerfm-models.json')
CLI_FILE = os.path.join(BASE_DIR, '.partnerfm-cli.json')
MCP_FILE = os.path.join(BASE_DIR, '.partnerfm-mcp.json')
CHAT_FILE = os.path.join(BASE_DIR, '.partnerfm-chats.json')
HOST = '127.0.0.1'
PORT = 8765

# Default CLI registry — what CLIs PartnerFM knows about
DEFAULT_CLI = {
    "cursor-agent": {
        "name": "Cursor Agent",
        "path": "/Applications/Cursor.app/Contents/Resources/app/bin/cursor agent",
        "icon": "🖱️",
        "description": "Cursor 编辑器的 AI 编程代理，支持 print 模式和交互模式",
        "tutorial": "## 使用方式\n\n**Print 模式（推荐）：**\n```bash\ncursor agent -p \"你的任务\" --workspace ~/project\n```\n\n**交互模式：**\n```bash\ncursor agent \"你的任务\"\n```\n\n**在 PartnerFM 中调用：**\n在聊天窗口用自然语言描述任务，我来帮你调 CLI。"
    },
    "hermes-agent": {
        "name": "Hermes Agent",
        "path": "hermes",
        "icon": "⚡",
        "description": "多平台 AI 代理，支持 20+ 大模型提供商，消息平台网关",
        "tutorial": "## 使用方式\n\n**单次查询：**\n```bash\nhermes chat -q \"你的问题\"\n```\n\n**交互模式：**\n```bash\nhermes\n```\n\n**在 PartnerFM 中调用：**\n在聊天窗口直接对话即可，或用 `hermes chat -q` 做单次任务。"
    }
}

# Default MCP registry
DEFAULT_MCP = {
    "filesystem": {
        "name": "文件系统",
        "icon": "📁",
        "description": "安全的本地文件系统访问，读/写/搜索文件",
        "command": "npx @modelcontextprotocol/server-filesystem ~/Desktop",
        "tutorial": "## 功能\n\n- 读取本地文件\n- 写入文件\n- 搜索文件\n\n**在 PartnerFM 中：** 文件管理模块已直接调用了浏览器文件系统 API，不需要额外配置 MCP。"
    },
    "fetch": {
        "name": "网页获取",
        "icon": "🌐",
        "description": "获取网页内容、搜索信息、提取数据",
        "command": "uvx mcp-server-fetch",
        "tutorial": "## 功能\n\n- 获取网页内容\n- 网页搜索\n\n**在 PartnerFM 中：** 在聊天窗口直接问，我会用 web_search 工具帮你查。"
    }
}

# Default chat history
DEFAULT_CHATS = {
    "chats": []
}


def _load_json(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _save_json(path, default)
        return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/state':
            return self._serve_json(_load_json(STATE_FILE, {}))
        if self.path == '/api/models':
            return self._serve_json(_load_json(MODELS_FILE, {"models": []}))
        if self.path == '/api/cli':
            data = _load_json(CLI_FILE, {"items": DEFAULT_CLI, "enabled": list(DEFAULT_CLI.keys())})
            return self._serve_json(data)
        if self.path == '/api/mcp':
            data = _load_json(MCP_FILE, {"items": DEFAULT_MCP, "enabled": list(DEFAULT_MCP.keys())})
            return self._serve_json(data)
        if self.path == '/api/chats':
            return self._serve_json(_load_json(CHAT_FILE, DEFAULT_CHATS))
        if self.path == '/api/health':
            return self._serve_json({'ok': True})
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/state':
            return self._save_json_endpoint(STATE_FILE)
        if self.path == '/api/models':
            return self._save_json_endpoint(MODELS_FILE)
        if self.path == '/api/cli':
            return self._save_json_endpoint(CLI_FILE)
        if self.path == '/api/mcp':
            return self._save_json_endpoint(MCP_FILE)
        if self.path == '/api/chats':
            return self._save_json_endpoint(CHAT_FILE)
        if self.path == '/api/chat':
            return self._proxy_chat()
        self.send_error(404)

    def _proxy_chat(self):
        """Proxy LLM chat request — forwards to the target provider's OpenAI-compatible API."""
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            req_data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        api_key = req_data.get('api_key', '')
        base_url = req_data.get('base_url', '')
        model = req_data.get('model', '')
        messages = req_data.get('messages', [])
        stream = req_data.get('stream', True)

        if not api_key or not base_url:
            self._serve_json({'error': '请先配置模型和 API Key'}, 400)
            return

        url = base_url.rstrip('/') + '/chat/completions'
        payload = json.dumps({
            'model': model,
            'messages': messages,
            'stream': stream
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            method='POST'
        )

        try:
            if stream:
                # Streaming response — forward chunks as SSE
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                with urllib.request.urlopen(req, timeout=300) as resp:
                    while True:
                        chunk = resp.readline()
                        if not chunk:
                            break
                        line = chunk.decode('utf-8', errors='ignore')
                        self.wfile.write(line.encode('utf-8') if isinstance(line, str) else chunk)
                        self.wfile.flush()
            else:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode('utf-8'))
                self._serve_json(result)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='ignore')
            self._serve_json({'error': f'API 错误 {e.code}: {error_body}'}, e.code)
        except Exception as e:
            self._serve_json({'error': str(e)}, 500)

    def _save_json_endpoint(self, path):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        _save_json(path, data)
        self._serve_json({'ok': True})

    def _serve_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    # Initialize default files on first run
    _load_json(STATE_FILE, {})
    _load_json(MODELS_FILE, {"models": []})
    _load_json(CLI_FILE, {"items": DEFAULT_CLI, "enabled": list(DEFAULT_CLI.keys())})
    _load_json(MCP_FILE, {"items": DEFAULT_MCP, "enabled": list(DEFAULT_MCP.keys())})
    _load_json(CHAT_FILE, DEFAULT_CHATS)

    os.chdir(BASE_DIR)
    print(f'PartnerFM → http://localhost:{PORT}')
    HTTPServer((HOST, PORT), Handler).serve_forever()
