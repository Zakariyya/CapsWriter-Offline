# coding: utf-8
"""
Minimal MCP stdio client (JSON-RPC over stdin/stdout).
"""
from __future__ import annotations

import json
import subprocess
import threading
import queue
from typing import Any, Dict, Optional

from util import get_logger
from util.llm.llm_exceptions import APIConnectionError, APIResponseError

logger = get_logger('client')


class MCPStdioClient:
    """Simple stdio MCP client with a long-running subprocess."""

    def __init__(self, command: str, args: Optional[list[str]] = None, env: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args or []
        self.env = env
        self._proc: Optional[subprocess.Popen] = None
        self._req_id = 0
        self._resp_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._err_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _start(self):
        if self._proc and self._proc.poll() is None:
            return

        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
                bufsize=1,
            )
        except Exception as e:
            raise APIConnectionError("mcp-stdio", self.command, str(e)) from e

        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        self._err_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._err_thread.start()

    def _read_stdout(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                self._resp_queue.put(data)
            except Exception:
                logger.debug(f"MCP stdio: {line}")

    def _read_stderr(self):
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            line = line.strip()
            if not line:
                continue
            logger.info(f"MCP stdio: {line}")

    def _send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._start()
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params or {}
        }
        msg = json.dumps(payload, ensure_ascii=False)

        assert self._proc and self._proc.stdin
        with self._lock:
            self._proc.stdin.write(msg + "\n")
            self._proc.stdin.flush()

        while True:
            resp = self._resp_queue.get()
            if resp.get("id") == self._req_id:
                if "error" in resp:
                    raise APIResponseError("mcp-stdio", None, str(resp["error"]))
                return resp.get("result")

    def initialize(self) -> Any:
        params = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "CapsWriter-Offline",
                "version": "2.4"
            }
        }
        return self._send("initialize", params)

    def list_tools(self) -> Any:
        try:
            self.initialize()
        except Exception:
            pass
        return self._send("tools/list", {})

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        return self._send("tools/call", {"name": name, "arguments": arguments or {}})
