#!/usr/bin/env python3
"""PartnerFM local server — serves static files, persists state, proxies LLM calls."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, '.partnerfm-state.json')
MODELS_FILE = os.path.join(BASE_DIR, '.partnerfm-models.json')
CLI_FILE = os.path.join(BASE_DIR, '.partnerfm-cli.json')
MCP_FILE = os.path.join(BASE_DIR, '.partnerfm-mcp.json')
CHAT_FILE = os.path.join(BASE_DIR, '.partnerfm-chats.json')
PROMPTS_FILE = os.path.join(BASE_DIR, '.partnerfm-prompts.json')
WORKSPACES_FILE = os.path.join(BASE_DIR, '.partnerfm-workspaces.json')
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

# Default system prompts
DEFAULT_PROMPTS = {
    "prompts": [
        {"id": "none",      "name": "无模板",     "prompt": ""},
        {"id": "code",      "name": "帮我写代码", "prompt": "你是一个资深的软件工程师。请根据用户提供的上下文和需求，写出高质量、可运行的代码。使用中文回复，代码部分保持英文。"},
        {"id": "translate", "name": "翻译文档",   "prompt": "你是一个专业的技术文档翻译。请将用户提供的内容翻译成中文，保留代码块和技术术语的原文，确保翻译准确流畅。"},
        {"id": "summary",   "name": "总结内容",   "prompt": "你是一个高效的文档分析助手。请用简洁的中文总结用户提供的内容要点，使用分条列举的方式，突出关键信息。"},
        {"id": "explain",   "name": "解释概念",   "prompt": "你是一个耐心的技术导师。请用通俗易懂的中文解释用户提出的概念或代码，从基础到深入，逐步展开。"},
        {"id": "review",    "name": "代码审查",   "prompt": "你是一个严格的代码审查员。请审查用户提供的代码，指出潜在问题、性能瓶颈、安全隐患，并给出改进建议。使用中文回复。"},
        {"id": "write",     "name": "写作助手",   "prompt": "你是一个优秀的中文写作助手。请帮助用户润色、改写或创作内容，保持原意，提升表达质量。"},
        {"id": "custom",    "name": "自定义提示", "prompt": ""}
    ]
}

DEFAULT_WORKSPACES = {"workspaces": {}}


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
        if self.path == '/api/prompts':
            return self._serve_json(_load_json(PROMPTS_FILE, DEFAULT_PROMPTS))
        if self.path == '/api/workspaces':
            return self._serve_json(_load_json(WORKSPACES_FILE, DEFAULT_WORKSPACES))
        if self.path == '/api/agent-config':
            return self._serve_json({
                'tools': ['list_dir', 'read_file', 'write_file', 'search_files'],
                'max_iterations': 10
            })
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
        if self.path == '/api/prompts':
            return self._save_json_endpoint(PROMPTS_FILE)
        if self.path == '/api/workspaces':
            return self._save_json_endpoint(WORKSPACES_FILE)
        if self.path.startswith('/api/convert-office'):
            return self._convert_office()
        if self.path == '/api/chat':
            return self._proxy_chat()
        if self.path == '/api/agent':
            return self._agent_loop()
        self.send_error(404)

    def _find_libreoffice(self):
        """Find the LibreOffice executable."""
        for path in [
            '/Applications/LibreOffice.app/Contents/MacOS/soffice',
            'soffice', 'libreoffice',
        ]:
            if shutil.which(path):
                return path
        return None

    def _convert_office(self):
        """Convert uploaded Office file to PDF using LibreOffice."""
        # Read binary body
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            self._serve_json({'error': '未收到文件'}, 400)
            return

        # Extract filename from query string
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        filename = qs.get('name', ['document'])[0]
        if not filename:
            self._serve_json({'error': '缺少文件名'}, 400)
            return

        body = self.rfile.read(length)

        # Check LibreOffice availability
        lo_path = self._find_libreoffice()
        if not lo_path:
            self._serve_json({
                'error': '未找到 LibreOffice。请运行：brew install --cask libreoffice'
            }, 500)
            return

        # Save to temp directory and convert
        tmpdir = tempfile.mkdtemp(prefix='partnerfm-')
        try:
            input_path = os.path.join(tmpdir, filename)
            with open(input_path, 'wb') as f:
                f.write(body)

            # Run LibreOffice headless conversion
            result = subprocess.run(
                [lo_path, '--headless', '--convert-to', 'pdf', '--outdir', tmpdir, input_path],
                capture_output=True, text=True, timeout=60
            )

            # Find the output PDF
            base = os.path.splitext(filename)[0]
            pdf_path = os.path.join(tmpdir, base + '.pdf')
            if not os.path.exists(pdf_path):
                # Try glob
                pdfs = [f for f in os.listdir(tmpdir) if f.endswith('.pdf')]
                if pdfs:
                    pdf_path = os.path.join(tmpdir, pdfs[0])
                else:
                    self._serve_json({
                        'error': f'转换失败：{result.stderr.strip() or "未生成 PDF 文件"}'
                    }, 500)
                    return

            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Length', str(len(pdf_data)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(pdf_data)

        except subprocess.TimeoutExpired:
            self._serve_json({'error': '转换超时，文件可能过大'}, 500)
        except Exception as e:
            self._serve_json({'error': f'转换出错：{e}'}, 500)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

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

    def _agent_loop(self):
        """Agent loop: think → act → observe → think, with tool calling."""
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        api_key = req.get('api_key', '')
        base_url = req.get('base_url', '')
        model = req.get('model', '')
        messages = req.get('messages', [])
        workspace = req.get('workspace', '')
        max_iter = req.get('max_iterations', 10)

        if not api_key or not base_url:
            self._serve_json({'error': '请先配置模型和 API Key'}, 400)
            return
        wpath = os.path.expanduser(workspace) if workspace else None
        if wpath and not os.path.isdir(wpath):
            self._serve_json({'error': f'工作区路径不存在：{wpath}'}, 400)
            return

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "列出指定目录下的所有文件和子目录。path 为空或 '.' 时列出工作区根目录。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "相对于工作区根目录的路径，如 '.' 或 '产出' 或 'sop'"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取指定文件的内容。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "相对于工作区根目录的文件路径，如 'sop/01-素材入库.md'"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "新建或覆盖写入一个文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "相对于工作区根目录的文件路径，如 '产出/新文件.md'"},
                            "content": {"type": "string", "description": "要写入的完整文件内容"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "在工作区中递归搜索包含指定关键词的文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "要搜索的关键词"},
                            "path": {"type": "string", "description": "搜索的起始子目录，留空则搜索整个工作区"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        def _resolve(p):
            """Resolve a relative path to absolute, preventing escape from workspace."""
            if not wpath:
                raise ValueError('未设置工作区。请在聊天中添加文件或文件夹到对话上下文。')
            full = os.path.normpath(os.path.join(wpath, p))
            if not full.startswith(os.path.normpath(wpath)):
                raise ValueError(f'不允许访问工作区之外的路径：{p}')
            return full

        def _exec_tool(call):
            name = call['function']['name']
            try:
                args = json.loads(call['function'].get('arguments', '{}'))
            except json.JSONDecodeError:
                return f'参数解析失败：{call["function"].get("arguments", "")}'

            try:
                if name == 'list_dir':
                    p = _resolve(args.get('path', '.'))
                    if not os.path.isdir(p):
                        return f'目录不存在：{args.get("path", ".")}'
                    items = []
                    for entry in sorted(os.listdir(p)):
                        if entry.startswith('.'):
                            continue
                        ep = os.path.join(p, entry)
                        tag = '📁' if os.path.isdir(ep) else '📄'
                        size = ''
                        if os.path.isfile(ep):
                            s = os.path.getsize(ep)
                            size = f' ({s}B)' if s < 1024 else f' ({s//1024}KB)'
                        items.append(f'{tag} {entry}{size}')
                    return '\n'.join(items) if items else '(空目录)'

                elif name == 'read_file':
                    p = _resolve(args['path'])
                    if not os.path.isfile(p):
                        return f'文件不存在：{args["path"]}'
                    with open(p, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    if len(content) > 8000:
                        content = content[:8000] + '\n\n... (文件过长，已截断。建议用 search_files 搜索关键词定位)'
                    return content

                elif name == 'write_file':
                    p = _resolve(args['path'])
                    parent = os.path.dirname(p)
                    if not os.path.isdir(parent):
                        return f'目录不存在：{os.path.relpath(parent, wpath) if wpath else parent}。请先用 list_dir 确认父目录，或直接写在工作区根目录（path 只写文件名，如 "测试.md"）。'
                    os.makedirs(parent, exist_ok=True)
                    with open(p, 'w', encoding='utf-8') as f:
                        f.write(args['content'])
                    return f'文件已写入：{args["path"]}'

                elif name == 'search_files':
                    query = args['query'].lower()
                    start = _resolve(args.get('path', '.'))
                    if not os.path.isdir(start):
                        start = wpath
                    results = []
                    for root, dirs, files in os.walk(start):
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        for fname in files:
                            if fname.startswith('.'):
                                continue
                            fp = os.path.join(root, fname)
                            try:
                                with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                                    content = f.read().lower()
                            except Exception:
                                continue
                            if query in content:
                                rel = os.path.relpath(fp, wpath)
                                count = content.count(query)
                                results.append(f'{rel} ({count}处匹配)')
                    if not results:
                        return f'未找到包含 "{args["query"]}" 的文件'
                    return '\n'.join(results[:20])

                else:
                    return f'未知工具：{name}'
            except ValueError as e:
                return str(e)
            except Exception as e:
                return f'执行出错：{e}'

        # --- Agent loop ---
        iteration = 0
        system_msg = next((m for m in messages if m['role'] == 'system'), None)
        if not system_msg:
            base_prompt = '你是一个智能助手。'
            if wpath:
                base_prompt += f'当前工作区根目录：{wpath}。所有文件路径相对于此目录（"."=根目录）。当用户说"在XX文件夹里面"操作而XX正好是根目录的名字时，直接在根目录操作，不要创建同名子目录。根目录下的文件和文件夹是用户的直接内容，像管理自己的文件夹一样管理它们。你可以使用工具读取、写入、搜索文件。操作前先用 list_dir(".") 看一眼。请用中文回复。'
            else:
                base_prompt += '你可以根据对话中的文件内容回答用户问题。如果你需要操作文件但工作区未设置，请告诉用户将文件夹添加到对话上下文。请用中文回复。'
            system_msg = {'role': 'system', 'content': base_prompt}
            messages.insert(0, system_msg)

        try:
            while iteration < max_iter:
                iteration += 1
                url = base_url.rstrip('/') + '/chat/completions'
                payload = json.dumps({
                    'model': model,
                    'messages': messages,
                    'tools': tools,
                    'tool_choice': 'auto'
                }).encode('utf-8')

                r = urllib.request.Request(url, data=payload, headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }, method='POST')

                with urllib.request.urlopen(r, timeout=120) as resp:
                    result = json.loads(resp.read().decode('utf-8'))

                choice = result.get('choices', [{}])[0]
                msg = choice.get('message', {})
                finish = choice.get('finish_reason', '')

                if finish == 'tool_calls' or msg.get('tool_calls'):
                    tool_calls = msg.get('tool_calls', [])
                    # Add assistant tool call message
                    messages.append({
                        'role': 'assistant',
                        'content': msg.get('content'),
                        'tool_calls': tool_calls
                    })
                    # Execute each tool and add results
                    for tc in tool_calls:
                        result_text = _exec_tool(tc)
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': tc['id'],
                            'content': result_text
                        })
                    continue  # Next iteration

                # Final text response
                final_content = msg.get('content', '')
                self._serve_json({
                    'content': final_content,
                    'iterations': iteration,
                    'model': model
                })
                return

            # Max iterations reached
            self._serve_json({
                'content': '达到最大执行轮次，但任务可能未完成。请检查结果或简化指令。',
                'iterations': iteration,
                'model': model
            })

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='ignore')
            self._serve_json({'error': f'API 错误 {e.code}: {error_body}'}, e.code)
        except Exception as e:
            self._serve_json({'error': str(e)}, 500)

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    # Initialize default files on first run
    _load_json(STATE_FILE, {})
    _load_json(MODELS_FILE, {"models": []})
    _load_json(CLI_FILE, {"items": DEFAULT_CLI, "enabled": list(DEFAULT_CLI.keys())})
    _load_json(MCP_FILE, {"items": DEFAULT_MCP, "enabled": list(DEFAULT_MCP.keys())})
    _load_json(CHAT_FILE, DEFAULT_CHATS)
    _load_json(PROMPTS_FILE, DEFAULT_PROMPTS)
    _load_json(WORKSPACES_FILE, DEFAULT_WORKSPACES)

    os.chdir(BASE_DIR)
    print(f'PartnerFM → http://localhost:{PORT}')
    HTTPServer((HOST, PORT), Handler).serve_forever()
