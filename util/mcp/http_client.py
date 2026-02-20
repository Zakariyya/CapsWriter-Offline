# coding: utf-8
"""
Minimal MCP StreamableHTTP client for tools discovery and invocation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from util.llm.llm_exceptions import APIConnectionError, APIResponseError
from util import get_logger

logger = get_logger('client')


class MCPHttpClient:
    """Simple MCP-over-HTTP client."""

    def __init__(self, base_url: str, auth_token: str = "", timeout: float = 30.0, headers: Optional[Dict[str, str]] = None):
        base = base_url.rstrip("/")
        # Some MCP hosts expose JSON-RPC directly at /mcp-servers/<name>
        if "/mcp-servers/" in base or base.endswith("/mcp"):
            self.endpoint = base
        else:
            self.endpoint = f"{base}/mcp"
        self.auth_token = auth_token
        self.timeout = timeout
        self.extra_headers = headers or {}
        self._session_id: Optional[str] = None
        self._req_id = 0

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self.extra_headers:
            headers.update(self.extra_headers)
        return headers

    def _post(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params or {}
        }
        try:
            resp = httpx.post(self.endpoint, json=payload, headers=self._headers(), timeout=self.timeout)
        except Exception as e:
            raise APIConnectionError("mcp", self.endpoint, str(e)) from e

        if resp.status_code >= 400:
            logger.error(f"MCP HTTP {resp.status_code} {self.endpoint} method={method} body={resp.text}")
            raise APIResponseError("mcp", resp.status_code, f"{self.endpoint} ({method}) {resp.text}")

        # MCP session id header (case-insensitive)
        session_id = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise APIResponseError("mcp", resp.status_code, str(data.get("error")))
        return data.get("result") if isinstance(data, dict) else data

    def initialize(self) -> Any:
        params = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "CapsWriter-Offline",
                "version": "2.4"
            }
        }
        return self._post("initialize", params)

    def list_tools(self) -> List[Dict[str, Any]]:
        if not self._session_id:
            try:
                self.initialize()
            except Exception:
                # Some servers allow tools/list without initialize
                pass

        result = self._post("tools/list", {})
        if isinstance(result, dict) and "tools" in result:
            return result["tools"]
        if isinstance(result, list):
            return result
        return []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._session_id:
            try:
                self.initialize()
            except Exception:
                pass

        return self._post("tools/call", {"name": name, "arguments": arguments or {}})
