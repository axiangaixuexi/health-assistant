"""
DashScope ``qwen3-rerank`` 精排封装。

在「BM25 + 稠密向量」召回之后，用 cross-encoder 式 rerank 模型对候选
``(query, document)`` 打分重排，取 top-k。当 ``DASHSCOPE_API_KEY`` 未配置时
降级为 no-op（保持召回顺序），使应用在没有密钥时仍可运行。

环境变量：
    RERANK_MODEL       精排模型名（默认 qwen3-rerank）
    DASHSCOPE_API_KEY  百炼/DashScope API key（留空则跳过精排）
    RERANK_BASE_URL    精排接口地址（默认 DashScope text-rerank 端点）
"""
from __future__ import annotations

import os
from typing import Sequence

import httpx

from backend._log import fmt_args

DEFAULT_MODEL = "qwen3-rerank"
DEFAULT_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


class DashScopeReranker:
    """调用 DashScope text-rerank 接口，对召回候选做相关性精排。"""

    def __init__(self, model: str, api_key: str, base_url: str, timeout: float = 30.0) -> None:
        print(f"--- [reranker] DashScopeReranker.__init__({fmt_args(model=model, api_key=api_key, base_url=base_url, timeout=timeout)}) ---", flush=True)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = httpx.Client(timeout=timeout)
        if not api_key:
            print(
                "--- [reranker] DASHSCOPE_API_KEY 未配置，精排降级为 no-op（保持召回顺序） ---",
                flush=True,
            )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[int]:
        """返回按相关性从高到低排序的文档下标列表（最多 top_n 个）。"""
        print(f"--- [reranker] DashScopeReranker.rerank({fmt_args(query=query, documents=documents, top_n=top_n)}) ---", flush=True)
        if not documents:
            return []
        n = top_n or len(documents)
        if not self.api_key:
            return list(range(min(n, len(documents))))

        payload = {
            "model": self.model,
            "input": {"query": query, "documents": list(documents)},
            "parameters": {"top_n": n, "return_documents": False},
        }
        resp = self._client.post(
            self.base_url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        results = resp.json().get("output", {}).get("results", [])
        results = sorted(
            results,
            key=lambda item: item.get("relevance_score", 0.0),
            reverse=True,
        )
        return [int(item["index"]) for item in results if "index" in item][:n]


def get_reranker() -> DashScopeReranker:
    """按环境变量构建 qwen3-rerank 精排器。"""
    print("--- [reranker] get_reranker() ---", flush=True)
    model = os.getenv("RERANK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("RERANK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    return DashScopeReranker(model=model, api_key=api_key, base_url=base_url)
