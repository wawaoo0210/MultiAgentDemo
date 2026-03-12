# 🤖 LangGraph Multi-Agent Assistant

这是一个基于 **LangGraph** 和 **通义千问 (Qwen)** 构建的多智能体 AI 助手。项目采用了先进的 Multi-Agent 架构，通过路由智能体（Supervisor）将用户请求精准分发给不同的专业智能体处理，并支持跨轮次的全局记忆对话。

## ✨ 核心特性 / Features

- 🧭 **智能路由 (Supervisor)**: 准确识别用户意图，将任务分发给最合适的后台 Agent。
- 🗺️ **MCP 工具调用 (Travel Agent)**: 集成高德地图 MCP 服务，动态查询经纬度并规划步行/驾车路线。
- 📜 **RAG 向量检索 (Couplet Agent)**: 结合 **Redis Stack** 向量数据库，根据用户的上联，检索海量经典对联作为参考，由大模型生成绝佳下联。
- 🎭 **幽默生成 (Joke Agent)**: 结合多轮对话上下文，生成短小精悍的笑话。
- 🧠 **全局持久化记忆**: 使用 `InMemorySaver`，各个智能体共享多轮对话历史，支持连环追问。
- 🎨 **Web UI交互**: 基于 `Gradio` 构建了丝滑的现代网页聊天界面，实时展示多智能体的工作流状态。

## 🛠️ 技术栈 / Tech Stack

- **框架**: LangChain, LangGraph
- **大模型**: 阿里云通义千问 (`qwen-turbo`)
- **向量数据库**: Redis Stack (`langchain-redis`)
- **工具协议**: MCP (Model Context Protocol)
- **前端界面**: Gradio

## 🚀 快速启动 / Quick Start

### 1. 环境准备
确保你的本地已安装 `Python 3.10+` 和 `Node.js` (用于运行高德 MCP)。
还需要安装并启动 **Redis Stack**：

```bash
brew tap redis-stack/redis-stack
brew install redis-stack-server
redis-stack-server
```

### 2. 安装依赖
建议在虚拟环境中运行：

```Bash
pip install -r requirements.txt
```

### 3. 配置环境变量
在项目根目录创建一个 `.env` 文件，并填入你的 API Keys：

```
DASHSCOPE_API_KEY="你的阿里云通义千问Key"
AMAP_MAPS_API_KEY="你的高德地图Key"
```

### 4. 初始化向量数据库 (可选)
如果你需要使用对联功能，请先准备好 `resource/couplet.csv` 数据集，然后运行：
``` Bash
python coupletLoader.py
```
### 5. 启动 Web 界面
```Bash
python web_ui.py
```
访问终端输出的本地链接（通常为 `http://127.0.0.1:7860`）即可开始体验！

## 📄 License
MIT License