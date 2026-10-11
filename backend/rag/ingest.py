"""
知识库摄取脚本。

从 rag/documents/ 中的文档构建（或重建）向量存储。
安装完成后运行一次，之后每当文档变化时再运行。

用法：
    python -m rag.ingest              # 不存在则构建，否则跳过
    python -m rag.ingest --rebuild    # 强制从文档重建

环境变量（从 .env 读取）：
    EMBEDDING_PROVIDER  huggingface | openai | ark（默认：huggingface）
    EMBEDDING_MODEL     模型名 / 路径（默认：BAAI/bge-small-zh-v1.5）
    VECTOR_STORE        chroma | faiss | qdrant | pgvector（默认：chroma）
    CHROMA_PERSIST_DIR  Chroma 存储路径（默认：rag/chroma_db）
    FAISS_INDEX_PATH    FAISS 索引路径（默认：rag/faiss_index）
    QDRANT_URL          Qdrant 服务器 URL（默认：http://localhost:6333）
    PGVECTOR_CONNECTION_STRING  PostgreSQL DSN
"""
import argparse
import sys
import time

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # 在导入 rag 模块之前加载 .env，使环境变量可用


def main() -> None:
    print("--- [ingest] main() ---", flush=True)
    parser = argparse.ArgumentParser(
        description="Build the health knowledge base vector store."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if the store already exists",
    )
    args = parser.parse_args()

    import os

    print("═" * 46)
    print("  Health Knowledge Base — Ingestion")
    print("═" * 46)
    print(f"  Embedding provider : {os.getenv('EMBEDDING_PROVIDER', 'huggingface')}")
    print(f"  Embedding model    : {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')}")
    print(f"  Vector store       : {os.getenv('VECTOR_STORE', 'chroma')}")
    print(f"  Mode               : {'REBUILD' if args.rebuild else 'BUILD (skip if exists)'}")
    print()

    from backend.rag.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()
    t0 = time.time()
    try:
        if args.rebuild:
            print("Rebuilding vector store from documents …")
            kb.rebuild()
        else:
            print("Building vector store (first run may download embedding model) …")
            kb._sync_init()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  ✓  Vector store is ready.")


if __name__ == "__main__":
    main()
