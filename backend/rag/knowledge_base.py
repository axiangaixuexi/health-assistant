from backend._log import fmt_args
import asyncio
import re
from pathlib import Path
from typing import List

from anyio.functools import lru_cache
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.rag.embeddings import get_embeddings
from backend.rag.reranker import get_reranker, DashScopeReranker
from backend.rag.vectorstores import get_vectorstore

DOCS_DIR=Path(__file__).parent / "documents"

_CHINESE_SEPARATORS = ["。", "！", "？", "；", "\n\n", "\n", "，", " ", ""]

# 精排前召回的候选数（rerank 需要比最终返回的 k 更大的候选池）。
_RERANK_CANDIDATES = 12


class HealthKnowledgeBase:

    def __init__(self) -> None:
        print("--- [knowledge_base] HealthKnowledgeBase.__init__() ---", flush=True)
        self._retriever: EnsembleRetriever | None = None
        self._reranker: DashScopeReranker | None = None
        self._initialized = False
        self._lock = asyncio.Lock()


    async def _ensure_initialized(self) -> None:
        print("--- [knowledge_base] HealthKnowledgeBase._ensure_initialized() ---", flush=True)
        if self._initialized:
            return
        async  with self._lock:
            if self._initialized:
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None,self._sync_init)

    def _sync_init(self, *, rebuild: bool = False) -> None:
        """阻塞式初始化——通过 run_in_executor 调用，或直接从 CLI 调用。"""
        print(f"--- [knowledge_base] HealthKnowledgeBase._sync_init({fmt_args(rebuild=rebuild)}) ---", flush=True)
        embeddings = get_embeddings()
        documents = self._load_documents()
        vectorstore = get_vectorstore(embeddings, documents, rebuild=rebuild)

        dense_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": _RERANK_CANDIDATES, "fetch_k": _RERANK_CANDIDATES * 3},
        )
        sparse_retriever = BM25Retriever.from_documents(documents, k=_RERANK_CANDIDATES)

        self._retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
        )
        self._reranker = get_reranker()
        self._initialized = True

    def rebuild(self) -> None:
        """从源文档删除并重建向量存储（同步）。"""
        print("--- [knowledge_base] HealthKnowledgeBase.rebuild() ---", flush=True)
        self._initialized = False
        self._retriever = None
        self._sync_init(rebuild=True)


    def _load_documents(self) -> List[Document]:
        """
        从 documents/ 加载 Markdown 文件，按 ## 章节标题切分，然后
        使用中文感知的分隔符应用 RecursiveCharacterTextSplitter。
        每个文本块都带有 {source, section} 元数据。
        """
        print("--- [knowledge_base] HealthKnowledgeBase._load_documents() ---", flush=True)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=_CHINESE_SEPARATORS,
        )
        docs: List[Document] = []
        for md_file in sorted(DOCS_DIR.glob("*.md")):
            source = md_file.stem
            text = md_file.read_text(encoding="utf-8")
            # 按以 "## " 开头的行切分，以提取具名章节
            sections = re.split(r"\n(?=## )", text)
            for section in sections:
                lines = section.strip().splitlines()
                if not lines:
                    continue
                heading = lines[0].lstrip("# ").strip()
                body = "\n".join(lines[1:]).strip()
                if not body:
                    continue
                chunks = splitter.create_documents(
                    [body],
                    metadatas=[{"source": source, "section": heading}],
                )
                docs.extend(chunks)
        return docs




    async def _rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """用 qwen3-rerank 对召回候选做精排；无密钥时保持召回顺序。"""
        if not docs or self._reranker is None or not self._reranker.api_key:
            return docs[:k]
        loop = asyncio.get_event_loop()
        order = await loop.run_in_executor(
            None,
            self._reranker.rerank,
            query,
            [doc.page_content for doc in docs],
            k,
        )
        return [docs[i] for i in order if 0 <= i < len(docs)][:k]


    async def aretrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        异步混合检索——可以安全地从 FastAPI 处理器调用。
        先 BM25 + 稠密向量召回并 RRF 融合，再用 qwen3-rerank 精排，返回 top-k。
        """
        print(f"--- [knowledge_base] HealthKnowledgeBase.aretrieve({fmt_args(query=query, k=k)}) ---", flush=True)
        await self._ensure_initialized()
        assert self._retriever is not None
        candidates = await self._retriever.ainvoke(query)
        return await self._rerank(query, candidates, k)


    async def amulti_query_retrieve(self, queries: List[str], k: int = 5) -> List[Document]:
        """
        多查询检索（RAG 实现最佳实践）。

        并行地为每个查询运行混合检索器，然后合并并
        去重结果。对于复杂的医学问题，当单一查询措辞会漏掉相关文本块时，
        该方法能提高召回率。

        Args:
            queries: 对同一信息需求的 2-4 种不同措辞。
            k: 去重后返回的最大唯一文档数。
        """
        print(f"--- [knowledge_base] HealthKnowledgeBase.amulti_query_retrieve({fmt_args(queries=queries, k=k)}) ---", flush=True)
        await self._ensure_initialized()
        assert self._retriever is not None

        # 并发运行所有查询
        tasks = [self._retriever.ainvoke(q) for q in queries]
        per_query_results = await asyncio.gather(*tasks)

        # 按 page_content 去重，同时保留首次出现的顺序；池子比最终 k 更大，供精排使用
        seen: set[str] = set()
        merged: List[Document] = []
        pool_size = max(k * 4, _RERANK_CANDIDATES)
        for docs in per_query_results:
            for doc in docs:
                if len(merged) >= pool_size:
                    break
                key = doc.page_content
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
            if len(merged) >= pool_size:
                break

        # 用第一个查询（通常为原始用户问题）做精排
        return await self._rerank(queries[0], merged, k)

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        同步检索——用于 LangChain @tool 函数内部，那里可能
        已经在运行事件循环。
        """
        print(f"--- [knowledge_base] HealthKnowledgeBase.retrieve({fmt_args(query=query, k=k)}) ---", flush=True)
        if not self._initialized:
            self._sync_init()
        assert self._retriever is not None
        candidates = self._retriever.invoke(query)
        if self._reranker is not None and self._reranker.api_key:
            order = self._reranker.rerank(query, [d.page_content for d in candidates], k)
            candidates = [candidates[i] for i in order if 0 <= i < len(candidates)]
        return candidates[:k]

    @staticmethod
    def format_context(docs: List[Document]) -> str:
        """
        把检索到的文本块渲染为带标签的上下文块，用于注入
        智能体系统提示词。

        示例输出：
            [medical_knowledge · 高血压（Hypertension）]
            高血压是指动脉血压持续升高…

            ---

            [lab_reference · 血常规参考范围]
            红细胞计数（RBC）…
        """
        print(f"--- [knowledge_base] HealthKnowledgeBase.format_context({fmt_args(docs=docs)}) ---", flush=True)
        if not docs:
            return ""
        parts: List[str] = []
        for doc in docs:
            src = doc.metadata.get("source", "")
            sec = doc.metadata.get("section", "")
            label = f"[{src} · {sec}]" if sec else f"[{src}]"
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)



@lru_cache(maxsize=1)
def get_knowledge_base() -> HealthKnowledgeBase:
    print("--- [knowledge_base] get_knowledge_base() ---", flush=True)
    return HealthKnowledgeBase()