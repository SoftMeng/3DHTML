#!/usr/bin/env python3
"""多模态大模型客户端。凭据只从环境变量读取，不落盘、不进日志。

用途边界：只做两件事 —— 结构化问答（JSON 输出）与图像问答。
不做：度量深度估计、逐帧精确角度推断（见 docs/constraint/vlm_boundary.md 的实测边界）。
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

ENV_URL = "MY_OPENAI_PROXY_URL"
ENV_KEY = "MY_OPENAI_API_KEY"
ENV_MODEL = "MY_OPENAI_MODEL_NAME"

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3


class VLMError(RuntimeError):
    """传输层或协议层失败。重试用尽后抛出，调用方必须显式处理。"""


def _env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise VLMError(f"环境变量 {name} 未设置")
    return v


class VLM:
    def __init__(self, model: str | None = None, timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES):
        self.base = _env(ENV_URL).rstrip("/")
        self.key = _env(ENV_KEY)
        self.model = model or _env(ENV_MODEL)
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ---- 协议 ----

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key},
            method="POST",
        )
        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self.calls += 1
                usage = body.get("usage") or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                return body
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8")[:180]
                except Exception:
                    pass
                last = f"HTTP {e.code} {detail}"
                if e.code in (400, 401, 403, 404):
                    break                       # 请求本身有问题，重试无意义
                time.sleep(2.0 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
        raise VLMError(f"{self.retries} 次尝试后仍失败 → {last}")

    @staticmethod
    def _content(body: dict) -> str:
        try:
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise VLMError(f"响应缺少 choices[0].message：{json.dumps(body)[:200]}") from e
        text = msg.get("content")
        if isinstance(text, list):  # 少数端点返回内容块数组
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        if not text or not str(text).strip():
            raise VLMError(f"响应内容为空：{json.dumps(body)[:200]}")
        return str(text)

    # ---- 能力 ----

    def ask(self, prompt: str, image: bytes | None = None, max_tokens: int = 400,
            json_mode: bool = False, temperature: float = 0.0) -> str:
        """单轮问答。image 为 PNG/JPEG 原始字节。返回模型文本。"""
        if image is None:
            content = prompt
        else:
            b64 = base64.b64encode(image).decode("ascii")
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return self._content(self._post(payload))

    def ask_json(self, prompt: str, image: bytes | None = None, max_tokens: int = 400,
                 temperature: float = 0.0, required: list[str] | None = None) -> dict:
        """要求模型返回 JSON 对象。解析失败会从文本里抠出最外层 {...} 再试一次；
        缺少 required 里的键 → 再问一次并附上缺哪个键。"""
        p = prompt if "json" in prompt.lower() else prompt + "\n只输出 JSON 对象。"
        text = self.ask(p, image=image, max_tokens=max_tokens, json_mode=True, temperature=temperature)
        obj = None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            text2 = self.ask(p + "\n上一次输出不是合法 JSON。现在只输出一个合法 JSON 对象，不要任何解释文字。",
                             image=image, max_tokens=max_tokens, json_mode=False, temperature=0.0)
            obj = self._parse_json(text2)
        if required:
            missing = [k for k in required if k not in obj]
            if missing:
                text3 = self.ask(p + f"\n上一次输出缺少键 {'、'.join(missing)}。补齐全部键后只输出 JSON 对象。",
                                 image=image, max_tokens=max_tokens, json_mode=True, temperature=0.0)
                try:
                    obj2 = json.loads(text3)
                    if all(k in obj2 for k in missing):
                        obj = obj2
                except json.JSONDecodeError:
                    pass
        return obj

    @staticmethod
    def _parse_json(text: str) -> dict:
        s = text.strip()
        if s.startswith("```"):
            s = s.split("```")[1] if "```" in s[3:] else s.strip("`")
            s = s[4:] if s.lower().startswith("json") else s
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError as e:
                raise VLMError(f"无法从响应中解析 JSON：{text[:200]}") from e
        raise VLMError(f"响应不含 JSON 对象：{text[:200]}")


def load_image(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
