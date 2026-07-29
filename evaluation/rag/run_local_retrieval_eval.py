#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不依赖 Milvus，用本地 BGE 和原始文档复现稠密检索评估。

运行：python3 evaluation/rag/run_local_retrieval_eval.py
"""

from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAG_CONFIG
from evaluation.rag.rag_evaluator import RAGEvaluator, load_cases


def split_text(text: str, max_chars: int = 600, overlap: int = 100):
    """与知识库初始化脚本保持一致的段落优先切分逻辑。"""
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
        if len(candidate) <= max_chars:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk.strip())
        if len(paragraph) > max_chars:
            remaining = paragraph
            while len(remaining) > max_chars:
                chunks.append(remaining[:max_chars])
                remaining = remaining[max_chars - overlap:]
            current_chunk = remaining
        else:
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


class LocalDenseRetriever:
    """使用 NumPy 余弦相似度复现当前 Milvus 稠密检索逻辑。"""

    def __init__(self):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.top_k = int(RAG_CONFIG.get("top_k", 4))
        self.similarity_threshold = float(RAG_CONFIG.get("similarity_threshold", 0.50))
        self.candidate_multiplier = int(RAG_CONFIG.get("candidate_multiplier", 3))
        self.dedupe_similarity = float(RAG_CONFIG.get("dedupe_similarity", 0.92))

        documents_dir = (
            PROJECT_ROOT / ".claude" / "skills" / "ask-question" / "data" / "documents"
        )
        self.documents = []
        for path in sorted(documents_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            title = text.splitlines()[0].strip() if text else path.stem
            for index, content in enumerate(split_text(text), 1):
                self.documents.append({
                    "id": f"{path.stem}_{index}",
                    "content": content,
                    "metadata": {
                        "title": f"{title} (Part {index})",
                        "parent_doc": path.name,
                        "file_path": str(path),
                    },
                })

        model_path = PROJECT_ROOT / RAG_CONFIG["embedding_model"]
        self.model = SentenceTransformer(str(model_path))
        self.document_vectors = self.model.encode(
            [document["content"] for document in self.documents],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @staticmethod
    def _normalize(content: str) -> str:
        return "".join(content.lower().split())

    def search_knowledge(self, query: str):
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = self.document_vectors @ query_vector
        candidate_limit = self.top_k * self.candidate_multiplier
        accepted = []
        accepted_contents = []

        for index in self.np.argsort(scores)[::-1][:candidate_limit]:
            score = float(scores[index])
            if score < self.similarity_threshold:
                continue
            document = self.documents[int(index)]
            normalized = self._normalize(document["content"])
            if any(
                normalized == existing
                or SequenceMatcher(None, normalized, existing).ratio() >= self.dedupe_similarity
                for existing in accepted_contents
            ):
                continue

            accepted.append({**document, "score": score, "distance": score})
            accepted_contents.append(normalized)
            if len(accepted) >= self.top_k:
                break
        return accepted


def main() -> int:
    retriever = LocalDenseRetriever()
    report = RAGEvaluator(load_cases()).evaluate_retrieval(retriever)
    report["configuration"] = {
        "embedding_model": RAG_CONFIG["embedding_model"],
        "document_chunks": len(retriever.documents),
        "top_k": retriever.top_k,
        "similarity_threshold": retriever.similarity_threshold,
        "candidate_multiplier": retriever.candidate_multiplier,
        "dedupe_similarity": retriever.dedupe_similarity,
        "note": "开发集结果，仅用于参数初调，不代表独立测试集或生产准确率。",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
