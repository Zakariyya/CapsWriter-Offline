"""
LLM 处理引擎

功能：
1. 执行 LLM API 调用
2. 处理流式输出
3. 更新上下文历史
4. 统一的错误处理和包装
5. 精确的生成时间统计（从第一个 token 开始）
"""
import time
import json
import re
from typing import Callable, Optional, Dict, Any, List, Tuple
from util.llm.llm_role_config import RoleConfig
from util.llm.llm_interfaces import IContextManager
from util.llm.llm_client_pool import ClientPool
from util.mcp import MCPHttpClient
from util.llm.llm_exceptions import (
    APIException,
    wrap_openai_error, OpenAIErrorWrapper,
    TimeoutErrorWrapper
)
from . import logger


class LLMProcessor:
    """LLM 处理引擎 - 负责 API 调用和流式输出"""

    def __init__(self, client_pool: ClientPool):
        """
        Args:
            client_pool: 客户端池
        """
        self.client_pool = client_pool

    def process(
        self,
        role_config: RoleConfig,
        messages: List[Dict[str, str]],
        callback: Optional[Callable[[str], None]] = None,
        should_stop_check: Optional[Callable[[], bool]] = None,
        context_manager: Optional[IContextManager] = None
    ) -> Tuple[str, int, float]:
        """
        执行 LLM 处理

        Args:
            role_config: 角色配置
            messages: 消息列表
            callback: 流式输出回调函数
            should_stop_check: 检查是否应该停止的函数
            context_manager: 上下文管理器（用于更新历史）

        Returns:
            (处理后的文本, 输出token数, 生成时间秒)
        """
        logger.info(f"开始 LLM 处理，模型: {role_config.model}")

        # 获取客户端
        logger.debug(f"获取 LLM 客户端，提供商: {role_config.provider}, API: {role_config.api_url}")
        client = self.client_pool.get_client(
            provider=role_config.provider,
            api_url=role_config.api_url,
            api_key=role_config.api_key
        )

        # 构建请求参数
        request_params = self._build_request_params(role_config, messages)
        logger.debug(f"请求参数: model={role_config.model}, stream=True")

        try:
            logger.debug("开始调用 LLM API（流式）")

            # MCP 工具调用（仅对启用的角色生效）
            if role_config.enable_mcp and (role_config.mcp_base_url or role_config.mcp_servers):
                tools, tool_registry = self._build_mcp_tools(role_config)
                if tools:
                    request_params['tools'] = tools
                    request_params['tool_choice'] = 'auto'
                    return self._stream_request_with_tools(
                        client,
                        request_params,
                        callback,
                        should_stop_check,
                        role_config,
                        context_manager,
                        messages,
                        tool_registry
                    )

            return self._stream_request(
                client,
                request_params,
                callback,
                should_stop_check,
                role_config,
                context_manager,
                messages
            )

        except OpenAIErrorWrapper:
            # 已包装的 OpenAI 异常，直接重新抛出
            raise
        except APIException:
            # 其他 API 相关异常，直接重新抛出
            raise
        except Exception as e:
            # 捕获 OpenAI SDK 原生异常并包装
            import openai

            # 处理 httpx 超时异常（在流式读取时发生）
            if 'httpx' in str(type(e).__module__):
                error_type = type(e).__name__
                if 'Timeout' in error_type or 'timeout' in error_type:
                    # httpx.ReadTimeout 或 httpx.TimeoutException
                    wrapped_error = TimeoutErrorWrapper(e, role_config.provider)
                    logger.error(f"LLM API 请求超时: {wrapped_error}")
                    raise wrapped_error from e

            if isinstance(e, (
                openai.AuthenticationError,
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.APIError
            )):
                wrapped_error = wrap_openai_error(e, role_config.provider)
                logger.error(f"LLM API 调用失败: {wrapped_error}")
                raise wrapped_error from e

            # 其他未预期的异常，包装为 APIException 并抛出
            import traceback
            error_msg = f"LLM 处理失败: {e}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise APIException(error_msg, role_config.provider) from e

    def _build_request_params(
        self,
        role_config: RoleConfig,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """构建请求参数"""
        request_params = {
            'model': role_config.model,
            'messages': messages,
        }

        # 添加生成参数
        if role_config.temperature is not None:
            request_params['temperature'] = role_config.temperature

        if role_config.top_p is not None:
            request_params['top_p'] = role_config.top_p

        if role_config.max_tokens > 0:
            request_params['max_tokens'] = role_config.max_tokens
            logger.debug(f"最大tokens: {role_config.max_tokens}")

        # 处理停止序列
        stop = role_config.stop
        if stop:
            if isinstance(stop, str):
                request_params['stop'] = [s.strip() for s in stop.split(',')]
            else:
                request_params['stop'] = stop
            logger.debug(f"停止序列: {request_params['stop']}")

        # 合并额外选项
        if role_config.extra_options:
            request_params.update(role_config.extra_options)

        return request_params

    def _build_mcp_tools(self, role_config: RoleConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """从多个 MCP 服务获取工具，并转换为 OpenAI tools 结构。

        Returns:
            (tool_specs, tool_registry) where tool_registry maps tool_name -> {client, original_name}
        """
        servers = role_config.mcp_servers or []
        if not servers and role_config.mcp_base_url:
            servers = [{
                "name": "mcp",
                "base_url": role_config.mcp_base_url,
                "auth_token": role_config.mcp_auth_token,
                "timeout": role_config.mcp_timeout,
                "tool_whitelist": role_config.mcp_tool_whitelist or []
            }]

        tool_specs: List[Dict[str, Any]] = []
        tool_registry: Dict[str, Dict[str, Any]] = {}
        tool_names = []

        for idx, server in enumerate(servers):
            base_url = server.get("base_url") or ""
            server_name = server.get("name") or f"mcp{idx+1}"
            whitelist = server.get("tool_whitelist") or []

            if not base_url:
                logger.warning(f"MCP 配置缺少 base_url: {server_name}")
                continue
            try:
                mcp_client = MCPHttpClient(
                    base_url=base_url,
                    auth_token=server.get("auth_token", ""),
                    timeout=server.get("timeout", role_config.mcp_timeout),
                    headers=server.get("headers")
                )
                tools = mcp_client.list_tools()
            except Exception as e:
                logger.error(f"MCP 工具列表获取失败: {server_name} {e}")
                continue

            for tool in tools:
                name = tool.get("name", "")
                if not name:
                    continue
                if whitelist and name not in whitelist:
                    continue

                final_name = name
                if final_name in tool_registry:
                    final_name = f"{server_name}::{name}"
                    if final_name in tool_registry:
                        logger.warning(f"MCP 工具名冲突: {name}, 跳过")
                        continue

                tool_registry[final_name] = {
                    "client": mcp_client,
                    "original_name": name
                }
                tool_names.append(final_name)

                params = tool.get("inputSchema") or tool.get("schema") or {"type": "object", "properties": {}}
                desc = tool.get("description", "") or ""
                if server_name:
                    desc = f"[{server_name}] {desc}".strip()

                tool_specs.append({
                    "type": "function",
                    "function": {
                        "name": final_name,
                        "description": desc,
                        "parameters": params
                    }
                })

        if tool_specs:
            logger.info(f"MCP 工具已加载: {len(tool_specs)}")
            logger.debug(f"MCP 工具列表: {tool_names}")
        else:
            logger.warning("MCP 工具列表为空")

        return tool_specs, tool_registry

    def _stream_request_with_tools(
        self,
        client: Any,
        request_params: Dict[str, Any],
        callback: Optional[Callable[[str], None]],
        should_stop_check: Optional[Callable[[], bool]],
        role_config: RoleConfig,
        context_manager: Optional[IContextManager],
        messages: List[Dict[str, str]],
        tool_registry: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[str, int, float]:
        """流式请求（支持一次 MCP 工具调用）"""
        # 允许有限轮次的工具调用
        max_rounds = 2
        round_idx = 0
        current_messages = messages
        current_params = dict(request_params)
        current_params['stream'] = True

        while True:
            round_idx += 1
            full_response, total_tokens, generation_time, tool_calls = self._stream_collect(
                client, current_params, callback, should_stop_check
            )

            if not tool_calls:
                logger.info("模型未触发 MCP 工具调用")
                if role_config.enable_history and context_manager:
                    user_msg = next((m for m in reversed(current_messages) if m.get('role') == 'user'), None)
                    if user_msg:
                        context_manager.add_message('user', user_msg.get('content', ''))
                    context_manager.add_message('assistant', full_response)
                    logger.debug("已更新历史记录")
                return (full_response.strip(), total_tokens, generation_time)

            if round_idx > max_rounds:
                logger.warning("达到 MCP 工具调用最大轮次，停止继续调用")
                return (full_response.strip(), total_tokens, generation_time)

            default_mcp_client = MCPHttpClient(
                base_url=role_config.mcp_base_url,
                auth_token=role_config.mcp_auth_token,
                timeout=role_config.mcp_timeout
            ) if role_config.mcp_base_url else None

            tool_messages = []
            for call in tool_calls:
                name = call["function"].get("name", "")
                raw_args = call["function"].get("arguments", "")
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    args = {"_raw": raw_args}

                try:
                    result = self._call_mcp_tool(default_mcp_client, name, args, tool_registry)
                except Exception as e:
                    result = {"content": [{"type": "text", "text": f"工具调用失败: {e}"}]}

                content_text = self._normalize_mcp_result(result)
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content_text
                })

            current_messages = current_messages + [{
                "role": "assistant",
                "tool_calls": tool_calls,
                "content": ""
            }] + tool_messages

            current_params = self._build_request_params(role_config, current_messages)
            current_params['stream'] = True
            if 'tools' in request_params:
                current_params['tools'] = request_params['tools']
                current_params['tool_choice'] = request_params.get('tool_choice', 'auto')

    @staticmethod
    def _normalize_mcp_result(result: Any) -> str:
        """将 MCP 返回值规范化为文本"""
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                if parts:
                    return "\n".join(parts).strip()
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    def _call_mcp_tool(
        self,
        mcp_client: Optional[MCPHttpClient],
        name: str,
        args: Dict[str, Any],
        tool_registry: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Any:
        """Try calling MCP tool with name variants."""
        if not name:
            raise ValueError("empty tool name")
        logger.info(f"MCP 调用工具: {name} args_keys={list(args.keys())}")

        client = mcp_client
        original_name = name
        if tool_registry and name in tool_registry:
            entry = tool_registry[name]
            client = entry.get("client") or client
            original_name = entry.get("original_name") or name

        if client is None:
            raise ValueError("MCP client not configured")

        try:
            return client.call_tool(original_name, args)
        except Exception as e:
            alt = None
            if "_" in original_name:
                alt = original_name.replace("_", "-")
            elif "-" in original_name:
                alt = original_name.replace("-", "_")
            if alt and alt != original_name:
                logger.info(f"MCP 工具名尝试替换: {original_name} -> {alt}")
                return client.call_tool(alt, args)
            raise e

    def _stream_collect(
        self,
        client: Any,
        request_params: Dict[str, Any],
        callback: Optional[Callable[[str], None]],
        should_stop_check: Optional[Callable[[], bool]],
    ) -> Tuple[str, int, float, List[Dict[str, Any]]]:
        """Stream once and collect tool calls if present."""
        stream = client.chat.completions.create(**request_params)

        full_response = ""
        total_tokens = 0
        chunk_count = 0

        first_token_time = None
        generation_start_time = None

        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}

        for chunk in stream:
            chunk_count += 1
            if should_stop_check and should_stop_check():
                logger.debug(f"收到停止信号，当前已接收 {chunk_count} 个 chunks")
                try:
                    stream.close()
                except:
                    pass
                break

            delta = chunk.choices[0].delta

            if getattr(delta, 'tool_calls', None):
                for tc in delta.tool_calls:
                    idx = getattr(tc, 'index', 0) or 0
                    entry = tool_calls_by_index.setdefault(idx, {
                        "id": f"mcp_call_{idx}",
                        "type": "function",
                        "function": {"name": "", "arguments": ""}
                    })

                    if getattr(tc, 'id', None):
                        entry["id"] = tc.id
                    if getattr(tc, 'type', None):
                        entry["type"] = tc.type
                    if getattr(tc, 'function', None):
                        fn = tc.function
                        if getattr(fn, 'name', None):
                            entry["function"]["name"] = fn.name
                        if getattr(fn, 'arguments', None):
                            entry["function"]["arguments"] += fn.arguments

            if getattr(delta, 'function_call', None):
                fc = delta.function_call
                entry = tool_calls_by_index.setdefault(0, {
                    "id": "mcp_call_0",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                })
                if getattr(fc, 'name', None):
                    entry["function"]["name"] = fc.name
                if getattr(fc, 'arguments', None):
                    entry["function"]["arguments"] += fc.arguments

            if delta.content:
                content_chunk = delta.content
                full_response += content_chunk

                if first_token_time is None:
                    first_token_time = time.time()
                    generation_start_time = first_token_time

                if callback:
                    callback(content_chunk)

            if hasattr(chunk, 'usage') and chunk.usage:
                if hasattr(chunk.usage, 'completion_tokens'):
                    tokens = chunk.usage.completion_tokens or 0
                    if tokens > 0:
                        total_tokens = tokens

        generation_time = 0.0
        if generation_start_time is not None:
            generation_time = time.time() - generation_start_time

        if total_tokens == 0 and full_response:
            from util.llm.llm_constants import estimate_tokens
            total_tokens = estimate_tokens(full_response)

        tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index.keys())]
        if not tool_calls:
            tool_calls = self._extract_dsml_tool_calls(full_response)

        return (full_response.strip(), total_tokens, generation_time, tool_calls)
    @staticmethod
    def _extract_dsml_tool_calls(text: str) -> List[Dict[str, Any]]:
        """Parse DeepSeek DSML tool calls including args blocks."""
        if "<｜DSML｜invoke" not in text:
            return []

        # Match each invoke block and optional function_args JSON
        pattern = r"<｜DSML｜invoke\\s+name=\"([^\"]+)\"\\s*>\\s*(?:<｜DSML｜function_args>\\s*(.*?)\\s*</｜DSML｜function_args>)?\\s*</｜DSML｜invoke>"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if not matches:
            return []

        tool_calls: List[Dict[str, Any]] = []
        for idx, (name, args_text) in enumerate(matches):
            args_text = (args_text or "").strip()
            tool_calls.append({
                "id": f"mcp_call_dsml_{idx}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_text
                }
            })
        return tool_calls

    def _stream_request(
        self,
        client: Any,
        request_params: Dict[str, Any],
        callback: Optional[Callable[[str], None]],
        should_stop_check: Optional[Callable[[], bool]],
        role_config: RoleConfig,
        context_manager: Optional[IContextManager],
        messages: List[Dict[str, str]]
    ) -> Tuple[str, int, float]:
        """执行流式请求

        Returns:
            (响应文本, token数, 生成时间秒)
        """
        request_params['stream'] = True
        stream = client.chat.completions.create(**request_params)

        full_response = ""
        total_tokens = 0
        chunk_count = 0

        # 计时：从第一个 token 开始
        first_token_time = None
        generation_start_time = None

        for chunk in stream:
            chunk_count += 1
            # 检查是否应该停止
            if should_stop_check and should_stop_check():
                logger.debug(f"收到停止信号，当前已接收 {chunk_count} 个 chunks")
                # 关闭流式响应，终止模型继续生成
                try:
                    stream.close()
                except:
                    pass
                # 中断循环，返回已生成的部分
                break

            if chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                full_response += content_chunk

                # 记录第一个 token 到达时间
                if first_token_time is None:
                    first_token_time = time.time()
                    generation_start_time = first_token_time

                if callback:
                    callback(content_chunk)

            # 统计 token 数（在最后一个 chunk 中获取）
            # 注意：某些提供商（如 Ollama）的流式响应不包含 usage
            if hasattr(chunk, 'usage') and chunk.usage:
                if hasattr(chunk.usage, 'completion_tokens'):
                    tokens = chunk.usage.completion_tokens or 0
                    if tokens > 0:
                        # 使用最后一个非零的 token 数
                        total_tokens = tokens

        # 计算生成时间（从第一个 token 到最后一个 token）
        generation_time = 0.0
        if generation_start_time is not None:
            generation_end_time = time.time()
            generation_time = generation_end_time - generation_start_time

        # 如果 API 没有返回 token 数，使用估算（针对 Ollama 等不返回 usage 的提供商）
        if total_tokens == 0 and full_response:
            from util.llm.llm_constants import estimate_tokens
            total_tokens = estimate_tokens(full_response)
            logger.debug(f"API 未返回 token 数，使用估算值: {total_tokens}")

        logger.debug(f"LLM 响应完成，接收 {chunk_count} 个 chunks, 输出tokens: {total_tokens}, 响应长度: {len(full_response)}, 生成时间: {generation_time:.3f}秒")
        # 记录响应内容（截断过长内容）
        preview_len = min(len(full_response), 500)
        logger.debug(f"LLM 响应内容: {full_response[:preview_len]}{'...' if len(full_response) > preview_len else ''}")

        # 更新历史
        if role_config.enable_history and context_manager:
            # 保存完整的用户提示词（包含剪贴板、热词等）
            user_msg = next((m for m in reversed(messages) if m.get('role') == 'user'), None)
            if user_msg:
                context_manager.add_message('user', user_msg.get('content', ''))
            context_manager.add_message('assistant', full_response)
            logger.debug(f"已更新历史记录")

        return (full_response.strip(), total_tokens, generation_time)
