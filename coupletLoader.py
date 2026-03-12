import os
import redis
from dotenv import load_dotenv  # 补全了环境变量加载模块
from langchain_community.embeddings import DashScopeEmbeddings  # 修复了导入路径
from langchain_redis import RedisConfig, RedisVectorStore

# 1. 加载 .env 中的 DASHSCOPE_API_KEY
load_dotenv()

# 2. 初始化阿里云通义千问的向量化模型
embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

# 3. 连接本地 Redis
redis_url = "redis://localhost:6379"
redis_client = redis.from_url(redis_url)
print("Redis 连接状态 (Ping):", redis_client.ping())

# 4. 配置 Redis 向量库
redis_config = RedisConfig(
    redis_url=redis_url,
    index_name="couplet"
)

vector_store = RedisVectorStore(
    config=redis_config,
    embeddings=embedding_model
)

# 5. 读取对联文件
lines = []
with open("resource/couplet.csv", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip(): 
            lines.append(line.strip())

        if len(lines) >= 1000:
            break

print(f"总共读取了 {len(lines)} 条数据。")
print("准备写入的前10条数据:", lines[:10])

# 6. 核心步骤：调用大模型将文本转换为向量，并存入 Redis
# 注意：这一步会消耗 dashscope 的 Token
vector_store.add_texts(lines)

print("Done! 对联数据已成功向量化并存入 Redis。")