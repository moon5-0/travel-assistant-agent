"""
Configuration for the Aligo Multi-Agent System
"""

import os

# LLM Configuration
LLM_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "model_name": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com",
    "temperature": 0.7,
    "max_tokens": 8192,
}

# System Configuration
SYSTEM_CONFIG = {
    "enable_llm": True,  # Set to True to use LLM (recommended), False for rule-based
    "log_level": "INFO",
    "max_retries": 3,
    "timeout": 60,  # Increased timeout for better stability
}

# RAG 知识库：嵌入模型（本地路径，无需连 HuggingFace）
RAG_CONFIG = {
    "embedding_model": "data/models/bge-small-zh-v1.5",
    # 语义向量与 BM25 分别召回候选，再使用 RRF 按排名融合。
    "retrieval_mode": "hybrid",
    "top_k": 4,
    # COSINE 相似度越高越相关；该初始阈值将在新版独立 RAG 评估集上重新校准。
    "similarity_threshold": 0.50,
    # 先多召回候选，再做阈值和重复片段过滤，最终最多保留 top_k 条。
    "candidate_multiplier": 3,
    "vector_candidate_k": 20,
    "keyword_candidate_k": 20,
    "rrf_k": 60,
    # BM25 命中时允许语义分略低于主阈值，但仍保留语义下限，避免关键词误召回。
    "hybrid_similarity_floor": 0.43,
    "dedupe_similarity": 0.92,
}

# 连接与可用性：重试、熔断、健康检查
RESILIENCE_CONFIG = {
    "max_retries": 3,              # 单次请求最大重试次数（与 SYSTEM_CONFIG 对齐）
    "retry_base_delay_sec": 1.0,   # 重试退避基数（秒）
    "retry_max_delay_sec": 30.0,   # 重试退避上限（秒）
    "circuit_failure_threshold": 5, # 连续失败多少次后熔断
    "circuit_recovery_timeout_sec": 60.0,  # 熔断后多少秒进入半开
    "circuit_half_open_successes": 2,      # 半开状态下连续成功多少次后关闭
    "health_check_timeout_sec": 10.0,      # 健康检查请求超时（秒）
}

# Redis 短期会话存储。Redis 是正式运行依赖；单元测试通过注入测试客户端隔离。
REDIS_SESSION_CONFIG = {
    "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "ttl_seconds": int(os.getenv("REDIS_SESSION_TTL_SECONDS", "3600")),
    "key_prefix": os.getenv("REDIS_KEY_PREFIX", "travel-agent"),
    "socket_connect_timeout_sec": float(
        os.getenv("REDIS_CONNECT_TIMEOUT_SECONDS", "2")
    ),
    "socket_timeout_sec": float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "2")),
}
