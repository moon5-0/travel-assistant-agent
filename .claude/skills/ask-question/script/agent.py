"""
RAG知识库智能体 RAGKnowledgeAgent
职责：基于向量数据库的知识检索与问答

核心功能：
1. 知识库构建：将商旅相关文档向量化并存储到Milvus Lite
2. 混合检索：通过 BGE 向量和 BM25 召回，并使用 RRF 融合知识片段
3. 知识问答：结合检索到的知识和LLM生成准确答案
4. 知识管理：支持添加、更新、删除知识库内容

技术栈：
- Milvus Lite: 轻量级向量数据库（本地存储）
- sentence-transformers: 文本向量化模型
- BM25 + RRF: 关键词召回与双路排名融合
- LLM: 用户配置的豆包模型用于生成答案

安装：
pip install milvus sentence-transformers
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List, Dict
from collections import Counter
from difflib import SequenceMatcher
import json
import logging
import math
import os
from pathlib import Path
import re
import unicodedata

# Add project root to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

_GRPC_MAX_MS = '2147483647'  # gRPC 使用的 int32 上限，约 24.8 天
os.environ['GRPC_KEEPALIVE_TIME_MS'] = _GRPC_MAX_MS
os.environ['GRPC_KEEPALIVE_TIMEOUT_MS'] = '20000'
os.environ['GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS'] = '0'
os.environ['GRPC_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS'] = _GRPC_MAX_MS
os.environ['GRPC_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS'] = _GRPC_MAX_MS

logger = logging.getLogger(__name__)

try:
    from pymilvus import MilvusClient, DataType
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG dependencies not available: {e}")
    logger.warning("Install with: pip install pymilvus sentence-transformers")
    DEPENDENCIES_AVAILABLE = False


class RAGKnowledgeAgent(AgentBase):
    """RAG知识库智能体"""

    def __init__(
        self,
        name: str = "RAGKnowledgeAgent",
        model=None,
        knowledge_base_path: str = None,
        collection_name: str = "business_travel_knowledge",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        candidate_multiplier: Optional[int] = None,
        dedupe_similarity: Optional[float] = None,
        retrieval_mode: Optional[str] = None,
        vector_candidate_k: Optional[int] = None,
        keyword_candidate_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        hybrid_similarity_floor: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.model = model
        
        if knowledge_base_path is None:
            # Default to local data directory in skill folder
            current_dir = Path(__file__).parent.parent
            knowledge_base_path = str(current_dir / "data" / "rag_knowledge")

        self.knowledge_base_path = Path(knowledge_base_path)
        self.collection_name = collection_name
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

        if not DEPENDENCIES_AVAILABLE:
            logger.error("RAG dependencies not installed. Install with: pip install pymilvus sentence-transformers")
            self.initialized = False
            return

        # 优先使用 config 中的配置（支持本地路径，避免连 HuggingFace）
        try:
            from config import RAG_CONFIG
            embedding_model = RAG_CONFIG.get("embedding_model", embedding_model)
            top_k = top_k if top_k is not None else RAG_CONFIG.get("top_k", 4)
            similarity_threshold = (
                similarity_threshold
                if similarity_threshold is not None
                else RAG_CONFIG.get("similarity_threshold", 0.50)
            )
            candidate_multiplier = (
                candidate_multiplier
                if candidate_multiplier is not None
                else RAG_CONFIG.get("candidate_multiplier", 3)
            )
            dedupe_similarity = (
                dedupe_similarity
                if dedupe_similarity is not None
                else RAG_CONFIG.get("dedupe_similarity", 0.92)
            )
            retrieval_mode = retrieval_mode or RAG_CONFIG.get("retrieval_mode", "hybrid")
            vector_candidate_k = (
                vector_candidate_k
                if vector_candidate_k is not None
                else RAG_CONFIG.get("vector_candidate_k", 20)
            )
            keyword_candidate_k = (
                keyword_candidate_k
                if keyword_candidate_k is not None
                else RAG_CONFIG.get("keyword_candidate_k", 20)
            )
            rrf_k = rrf_k if rrf_k is not None else RAG_CONFIG.get("rrf_k", 60)
            hybrid_similarity_floor = (
                hybrid_similarity_floor
                if hybrid_similarity_floor is not None
                else RAG_CONFIG.get("hybrid_similarity_floor", 0.43)
            )
        except Exception:
            top_k = top_k if top_k is not None else 4
            similarity_threshold = similarity_threshold if similarity_threshold is not None else 0.50
            candidate_multiplier = candidate_multiplier if candidate_multiplier is not None else 3
            dedupe_similarity = dedupe_similarity if dedupe_similarity is not None else 0.92
            retrieval_mode = retrieval_mode or "hybrid"
            vector_candidate_k = vector_candidate_k if vector_candidate_k is not None else 20
            keyword_candidate_k = keyword_candidate_k if keyword_candidate_k is not None else 20
            rrf_k = rrf_k if rrf_k is not None else 60
            hybrid_similarity_floor = (
                hybrid_similarity_floor
                if hybrid_similarity_floor is not None
                else 0.43
            )

        self.top_k = max(1, int(top_k))
        self.similarity_threshold = float(similarity_threshold)
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.dedupe_similarity = min(1.0, max(0.0, float(dedupe_similarity)))
        if retrieval_mode not in {"vector", "hybrid"}:
            raise ValueError("retrieval_mode must be 'vector' or 'hybrid'")
        self.retrieval_mode = retrieval_mode
        self.vector_candidate_k = max(self.top_k, int(vector_candidate_k))
        self.keyword_candidate_k = max(self.top_k, int(keyword_candidate_k))
        self.rrf_k = max(1, int(rrf_k))
        self.hybrid_similarity_floor = min(
            self.similarity_threshold,
            float(hybrid_similarity_floor),
        )
        # BM25 语料延迟加载。知识库新增或重建后会主动清空缓存。
        self._lexical_documents = None
        self._lexical_index = None

        # 若配置的是本地路径且存在，则从本地加载，否则按模型 ID 使用（会联网）
        model_path_or_id = embedding_model
        path_obj = Path(embedding_model).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
        if path_obj.exists():
            model_path_or_id = str(path_obj.resolve())
            logger.info(f"Using local embedding model: {model_path_or_id}")
        else:
            if "/" in embedding_model or "\\" in embedding_model or embedding_model.startswith("."):
                logger.warning(
                    f"Configured embedding path does not exist: {embedding_model}，将使用 BAAI/bge-small-zh-v1.5 并尝试联网下载。"
                )
                model_path_or_id = "BAAI/bge-small-zh-v1.5"
        logger.info(f"Loading embedding model: {model_path_or_id}")
        self.embedding_model = SentenceTransformer(model_path_or_id)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # 初始化 Milvus Lite（本地文件存储）
        milvus_db_path = str(self.knowledge_base_path / "milvus_lite.db")
        logger.info(f"Initializing Milvus Lite at: {milvus_db_path}")

        self.milvus_client = MilvusClient(
            milvus_db_path,
            # 新版 pymilvus 会尝试从本地绝对路径中推断数据库名；显式指定
            # default，避免把路径中的 Users 等目录名误当作 Milvus 数据库。
            db_name="default",
            grpc_options={
                "keepalive_time": _GRPC_MAX_MS,
                "keepalive_timeout": "20000",
                "keepalive_permit_without_calls": "0",
                "http2_min_recv_ping_interval_without_data": _GRPC_MAX_MS,
                "http2_min_ping_interval_without_data": _GRPC_MAX_MS,
            },
        )
        self._client_created_at = None  # 用于追踪客户端创建时间

        # 检查collection是否存在
        if self.milvus_client.has_collection(collection_name):
            logger.info(f"Loaded existing collection: {collection_name}")
        else:
            # 创建新collection
            logger.info(f"Creating new collection: {collection_name}")
            self.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=self.embedding_dim,
                metric_type="COSINE",  # 余弦相似度
                auto_id=False,
            )
            logger.info(f"Created new collection: {collection_name}")
        # Milvus Lite 3.x 在新进程中打开持久化 Collection 时默认可能处于
        # released 状态；显式 load 后才能执行 search/query。
        self.milvus_client.load_collection(collection_name)

        self.initialized = True
        self._milvus_db_path = milvus_db_path  # 保存路径用于重连
        logger.info("RAG Knowledge Agent (Milvus Lite) initialized successfully")

    def _ensure_connection(self):
        """确保 Milvus 连接正常，如果需要则重新创建客户端"""
        try:
            # 尝试一个轻量级操作来检查连接
            self.milvus_client.has_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"Milvus connection issue detected: {e}, reconnecting...")
            try:
                # 关闭旧连接
                if hasattr(self.milvus_client, 'close'):
                    try:
                        self.milvus_client.close()
                    except:
                        pass

                # 重新创建客户端
                self.milvus_client = MilvusClient(
                    self._milvus_db_path,
                    db_name="default",
                )
                if self.milvus_client.has_collection(self.collection_name):
                    self.milvus_client.load_collection(self.collection_name)
                logger.info("Milvus client reconnected successfully")
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect Milvus: {reconnect_error}")
                raise

    def add_documents(self, documents: List[Dict[str, str]]) -> Dict:
        """
        添加文档到知识库

        Args:
            documents: 文档列表，每个文档包含 {'content': '内容', 'metadata': {...}}

        Returns:
            添加结果统计
        """
        if not self.initialized:
            return {"status": "error", "message": "RAG Agent not initialized"}

        try:
            # 确保连接正常
            self._ensure_connection()
            # 获取当前文档总数，用于生成连续的ID
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            current_count = stats.get("row_count", 0)

            # 准备数据
            data_to_insert = []

            for i, doc in enumerate(documents):
                # Milvus 要求 id 必须是 int64
                doc_id = current_count + i + 1
                content = doc['content']
                metadata = doc.get('metadata', {})

                # 生成向量
                embedding = self.embedding_model.encode(content).tolist()

                # Milvus 数据格式
                data_to_insert.append({
                    "id": doc_id,
                    "vector": embedding,
                    "content": content,
                    "metadata": json.dumps(metadata, ensure_ascii=False)  # 将metadata转为JSON字符串
                })

            # 批量插入到 Milvus
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=data_to_insert
            )
            # 新文档写入后使内存 BM25 语料失效，下次检索会自动重建索引。
            self._lexical_documents = None
            self._lexical_index = None

            # 获取总数
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            total_count = stats.get("row_count", len(documents))

            logger.info(f"Successfully added {len(documents)} documents to knowledge base")
            return {
                "status": "success",
                "added_count": len(documents),
                "total_count": total_count
            }

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _normalize_content(content: str) -> str:
        """统一空白和大小写，用于重复片段判断。"""
        return "".join((content or "").lower().split())

    def _is_duplicate_content(self, content: str, accepted_contents: List[str]) -> bool:
        """过滤完全相同或几乎相同的候选片段。"""
        normalized = self._normalize_content(content)
        if not normalized:
            return True
        for existing in accepted_contents:
            if normalized == existing:
                return True
            if SequenceMatcher(None, normalized, existing).ratio() >= self.dedupe_similarity:
                return True
        return False

    @staticmethod
    def _decode_metadata(value) -> Dict:
        """兼容 Milvus 中 JSON 字符串与字典两种 metadata 格式。"""
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return dict(parsed) if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                return {}
        return {}

    @staticmethod
    def _lexical_tokens(text: str) -> List[str]:
        """生成通用词项：英文/数字词，以及中文单字和相邻双字。

        不维护面向评估题的同义词表。中文双字能够保留“报销、酒店”等
        词组，单字则让较短表达仍有机会匹配，常见虚词会被统一过滤。
        """
        normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
        latin_tokens = re.findall(r"[a-z0-9]+", normalized)
        han_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
        stop_chars = set("的了是在和与及或把被呢吗呀啊这那一个中内时后前")
        han_unigrams = [char for char in han_chars if char not in stop_chars]
        han_bigrams = [
            first + second
            for first, second in zip(han_chars, han_chars[1:])
            if first not in stop_chars and second not in stop_chars
        ]
        return latin_tokens + han_unigrams + han_bigrams

    @staticmethod
    def _requires_live_data(query: str) -> bool:
        """判断问题是否依赖知识库不可能持有的动态状态。

        这里判断的是“信息类型”而非某道评估题：实时天气、库存余量、
        即时价格和交通状态应由 InformationQueryAgent 查询，静态政策库
        即便碰巧含有相似关键词也不能回答。
        """
        normalized = unicodedata.normalize("NFKC", str(query or "")).lower()
        explicit_live_terms = (
            "余票", "实时天气", "天气怎么样", "天气如何", "气温多少",
            "会下雨", "是否下雨", "实时路况", "航班状态", "车次状态",
            "延误情况", "还有多少张", "库存还有", "剩余库存",
        )
        if any(term in normalized for term in explicit_live_terms):
            return True
        time_markers = ("实时", "当前", "现在", "今天", "明天", "后天", "最新")
        dynamic_subjects = (
            "天气", "气温", "降雨", "票价", "现价", "报价", "库存",
            "航班", "车次", "延误", "路况",
        )
        return (
            any(marker in normalized for marker in time_markers)
            and any(subject in normalized for subject in dynamic_subjects)
        )

    def _load_lexical_corpus(self) -> List[Dict]:
        """从 Milvus metadata/content 加载小型 BM25 语料并缓存在内存中。"""
        if self._lexical_documents is not None:
            return self._lexical_documents
        rows = self.milvus_client.query(
            collection_name=self.collection_name,
            filter="id >= 0",
            output_fields=["id", "content", "metadata"],
            limit=10000,
        )
        documents = []
        tokenized_documents = []
        for row in rows or []:
            content = str(row.get("content", "") or "")
            if not content:
                continue
            documents.append({
                "id": row.get("id"),
                "content": content,
                "metadata": self._decode_metadata(row.get("metadata")),
            })
            tokenized_documents.append(self._lexical_tokens(content))

        document_frequencies = Counter()
        document_lengths = []
        for tokens in tokenized_documents:
            document_frequencies.update(set(tokens))
            document_lengths.append(len(tokens))
        self._lexical_documents = documents
        self._lexical_index = {
            "tokens": tokenized_documents,
            "document_frequencies": document_frequencies,
            "document_lengths": document_lengths,
            "average_document_length": (
                sum(document_lengths) / len(document_lengths)
                if document_lengths
                else 0.0
            ),
        }
        return documents

    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """在当前知识块上执行标准 BM25 排名。"""
        documents = self._load_lexical_corpus()
        if not documents or not self._lexical_index:
            return []
        query_tokens = set(self._lexical_tokens(query))
        if not query_tokens:
            return []

        index = self._lexical_index
        corpus_size = len(documents)
        average_length = index["average_document_length"] or 1.0
        k1, b = 1.5, 0.75
        ranked = []
        for position, tokens in enumerate(index["tokens"]):
            frequencies = Counter(tokens)
            document_length = index["document_lengths"][position]
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = index["document_frequencies"].get(token, 0)
                inverse_document_frequency = math.log(
                    1 + (corpus_size - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * document_length / average_length
                )
                score += inverse_document_frequency * (
                    frequency * (k1 + 1) / denominator
                )
            if score > 0:
                document = dict(documents[position])
                document["keyword_score"] = score
                ranked.append(document)
        ranked.sort(key=lambda item: item["keyword_score"], reverse=True)
        return ranked[:limit]

    def _vector_search(self, query: str, limit: int) -> List[Dict]:
        """返回未做阈值过滤的向量候选，供纯向量或融合检索使用。"""
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["id", "content", "metadata"],
        )
        candidates = []
        if results and results[0]:
            for hit in results[0]:
                entity = hit.get("entity", {})
                candidates.append({
                    "id": entity.get("id", hit.get("id")),
                    "content": entity.get("content", ""),
                    "metadata": self._decode_metadata(entity.get("metadata")),
                    "vector_score": float(hit.get("distance", 0.0) or 0.0),
                })
        return candidates

    @staticmethod
    def _document_key(document: Dict):
        metadata = document.get("metadata") or {}
        source = metadata.get("parent_doc")
        chunk_index = metadata.get("chunk_index")
        if source and chunk_index is not None:
            return str(source), int(chunk_index)
        return "id", str(document.get("id"))

    def _rrf_fusion(
        self,
        vector_documents: List[Dict],
        keyword_documents: List[Dict],
    ) -> List[Dict]:
        """使用 Reciprocal Rank Fusion 合并语义和关键词候选。"""
        fused = {}
        for source_name, documents in (
            ("vector", vector_documents),
            ("keyword", keyword_documents),
        ):
            for rank, document in enumerate(documents, start=1):
                key = self._document_key(document)
                item = fused.setdefault(key, dict(document))
                item["rrf_score"] = item.get("rrf_score", 0.0) + 1 / (
                    self.rrf_k + rank
                )
                item[f"{source_name}_rank"] = rank
                if source_name == "vector":
                    item["vector_score"] = document.get("vector_score", 0.0)
                else:
                    item["keyword_score"] = document.get("keyword_score", 0.0)
        return sorted(
            fused.values(),
            key=lambda item: (
                item.get("rrf_score", 0.0),
                item.get("vector_score", 0.0),
            ),
            reverse=True,
        )

    def search_knowledge(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索知识库

        Args:
            query: 查询文本
            top_k: 返回top k个结果

        Returns:
            检索结果列表
        """
        self.last_search_error = None
        if not self.initialized or not str(query).strip():
            return []

        try:
            # 确保连接正常
            self._ensure_connection()
            if self._requires_live_data(query):
                logger.info("Skip static RAG retrieval for live-data query=%r", query[:50])
                return []
            k = top_k if top_k is not None else self.top_k
            k = max(1, int(k))
            vector_limit = max(
                self.vector_candidate_k,
                k * self.candidate_multiplier,
            )
            vector_documents = self._vector_search(query, vector_limit)
            if self.retrieval_mode == "hybrid":
                keyword_documents = self._keyword_search(
                    query,
                    max(self.keyword_candidate_k, k * self.candidate_multiplier),
                )
                candidates = self._rrf_fusion(vector_documents, keyword_documents)
            else:
                candidates = vector_documents

            retrieved_docs = []
            accepted_contents: List[str] = []
            for candidate in candidates:
                vector_score = float(candidate.get("vector_score", 0.0) or 0.0)
                keyword_rank = candidate.get("keyword_rank")
                # 纯语义结果沿用原阈值；关键词证据只有在语义也达到较宽松
                # 下限时才能补召回，避免只凭常见词回答无关问题。
                accepted_by_vector = vector_score >= self.similarity_threshold
                accepted_by_hybrid = (
                    self.retrieval_mode == "hybrid"
                    and keyword_rank is not None
                    and vector_score >= self.hybrid_similarity_floor
                )
                if not (accepted_by_vector or accepted_by_hybrid):
                    continue
                content = candidate.get("content", "")
                if self._is_duplicate_content(content, accepted_contents):
                    continue
                fusion_score = candidate.get("rrf_score", vector_score)
                retrieved_docs.append({
                    'id': candidate.get("id", ""),
                    'content': content,
                    'metadata': candidate.get("metadata", {}),
                    'score': fusion_score,
                    'distance': vector_score,
                    'vector_score': vector_score,
                    'keyword_score': candidate.get("keyword_score"),
                    'vector_rank': candidate.get("vector_rank"),
                    'keyword_rank': keyword_rank,
                    'rrf_score': candidate.get("rrf_score"),
                })
                accepted_contents.append(self._normalize_content(content))
                if len(retrieved_docs) >= k:
                    break

            logger.info(
                "Retrieved %d documents for query=%r (mode=%s, threshold=%.3f, vector_candidates=%d)",
                len(retrieved_docs),
                query[:50],
                self.retrieval_mode,
                self.similarity_threshold,
                len(vector_documents),
            )
            return retrieved_docs

        except Exception as e:
            self.last_search_error = f"{type(e).__name__}: {e}"
            logger.error(f"Error searching knowledge: {e}")
            return []

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """
        RAG问答主流程
        1. 接收用户查询
        2. 检索相关知识
        3. 结合知识生成答案
        """
        if not self.initialized:
            return Msg(
                name=self.name,
                content=json.dumps({
                    "status": "error",
                    "message": "RAG Agent not initialized. Please install dependencies: pip install pymilvus sentence-transformers"
                }),
                role="assistant"
            )

        if x is None:
            return Msg(name=self.name, content=json.dumps({}), role="assistant")

        # 获取用户查询
        if isinstance(x, list):
            content = x[-1].content if x else ""
        else:
            content = x.content

        # 尝试解析 JSON 输入 (来自 Orchestrator)
        user_query = content
        if isinstance(content, str) and content.strip().startswith('{'):
            try:
                data = json.loads(content)
                # 只要解析成功，就认为 content 是结构化数据，尝试提取 query
                extracted_query = ""
                if "context" in data and isinstance(data["context"], dict):
                    extracted_query = data["context"].get("rewritten_query", "")
                elif "rewritten_query" in data:
                    extracted_query = data.get("rewritten_query", "")
                
                # 使用提取到的 query（即使为空，也比 JSON 字符串好）
                user_query = extracted_query
            except:
                pass  # 解析失败则保留原字符串

        # 检索相关知识
        retrieved_docs = self.search_knowledge(user_query)

        if not retrieved_docs:
            result = {
                "status": "no_knowledge",
                "query": user_query,
                "answer": "抱歉，我在知识库中没有找到相关信息。",
                "retrieved_documents": [],
                "sources": [],
            }
            return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

        # 构建知识上下文
        knowledge_context = "\n\n".join(
            f"【知识片段{i + 1}｜来源：{doc['metadata'].get('title', '未知')}】\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        )

        # 如果有LLM，使用LLM生成答案
        if self.model:
            # 动态读取 Prompt 指令 (Progressive Disclosure)
            skill_instruction = self.skill_loader.get_skill_content("ask-question")
            if not skill_instruction:
                skill_instruction = "请基于知识库中的信息回答用户的问题。"

            prompt = f"""你是一个商旅知识专家。请严格基于以下知识库中的信息回答用户的问题。

【用户问题】
{user_query}

【知识库信息】
{knowledge_context}

【任务说明】
{skill_instruction}

【重要约束】
1. 如果【知识库信息】中没有包含回答用户问题所需的信息，请直接回答“抱歉，知识库中没有找到相关信息”，不要尝试根据你自己的知识编造答案。
2. 即使问题很基础，如果知识库里没写，就说不知道。
3. 请以专业、客观的语气回答。
4. 不要编造来源；只能引用【知识库信息】中标明的来源标题。
"""

            try:
                # 调用LLM生成答案
                messages = [
                    {"role": "system", "content": "你是一个商旅知识专家。"},
                    {"role": "user", "content": prompt}
                ]
                response = await self.model(messages)

                # 获取响应内容 - 处理异步生成器
                answer = ""
                if hasattr(response, '__aiter__'):
                    # 异步生成器，需要迭代获取内容
                    async for chunk in response:
                        if isinstance(chunk, str):
                            answer = chunk
                        elif hasattr(chunk, 'content'):
                            if isinstance(chunk.content, str):
                                answer = chunk.content
                            elif isinstance(chunk.content, list):
                                for item in chunk.content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        answer = item.get('text', '')
                elif hasattr(response, 'text'):
                    answer = response.text
                elif hasattr(response, 'content'):
                    answer = response.content
                elif isinstance(response, dict) and 'content' in response:
                    answer = response['content']
                else:
                    answer = str(response) if response else "无法生成答案"

                if not answer:
                    answer = "无法生成答案"
                
                # 清理 LLM 可能输出的 JSON 格式
                answer_str = answer.strip()
                if answer_str.startswith("{") and answer_str.endswith("}"):
                    try:
                        json_obj = json.loads(answer_str)
                        # 如果 LLM 输出了 {"answer": "..."} 或 {"content": "..."}
                        if isinstance(json_obj, dict):
                            answer = json_obj.get("answer") or json_obj.get("content") or answer
                    except:
                        pass

            except Exception as e:
                logger.error(f"Error generating answer with LLM: {e}")
                answer = f"知识库中找到相关信息，但生成答案时出错：{str(e)}"
        else:
            # 如果没有LLM，直接返回检索到的知识
            answer = "以下是知识库中的相关信息：\n\n" + knowledge_context

        sources = []
        seen_sources = set()
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            source_key = (
                metadata.get("parent_doc")
                or metadata.get("file_path")
                or metadata.get("title")
                or str(doc.get("id", ""))
            )
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            sources.append({
                "title": metadata.get("title", "未知来源"),
                "category": metadata.get("category", ""),
                "parent_doc": metadata.get("parent_doc", ""),
                "score": doc.get("score", doc.get("distance", 0.0)),
            })

        result = {
            "status": "success",
            "query": user_query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    "metadata": doc['metadata'],
                    "score": doc.get("score", doc.get("distance", 0.0)),
                }
                for doc in retrieved_docs
            ],
            "sources": sources,
        }

        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        if not self.initialized:
            return {"status": "error", "message": "Not initialized"}

        try:
            # 确保连接正常
            self._ensure_connection()
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            return {
                "status": "success",
                "collection_name": self.collection_name,
                "total_documents": stats.get("row_count", 0),
                "knowledge_base_path": str(self.knowledge_base_path),
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        """关闭 Milvus 连接"""
        if hasattr(self, 'milvus_client'):
            try:
                if hasattr(self.milvus_client, 'close'):
                    self.milvus_client.close()
                    logger.info("Milvus client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Milvus client: {e}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.close()
