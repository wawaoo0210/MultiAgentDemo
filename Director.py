import asyncio
import os
from dotenv import load_dotenv
from operator import add
from typing import Annotated, TypedDict
import redis
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

# 加载环境变量 (请确保 .env 中有 DASHSCOPE_API_KEY 和 AMAP_MAPS_API_KEY)
load_dotenv()

nodes = ["travel", "joke", "couplet", "other"]
llm = ChatTongyi(model="qwen-turbo")

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str

async def supervisor_node(state: State):
    print("\n>>> 正在执行: supervisor_node")

    # 1. 提取最近的几次对话作为上下文（避免把几百轮聊天全塞给路由，浪费Token）
    history_msgs = state["messages"][-5:-1]
    history_text = ""
    for msg in history_msgs:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        history_text += f"{role}: {msg.content}\n"
    
    if not history_text:
        history_text = "（无历史对话）"

    # 2. 提取最新输入
    user_input = state["messages"][-1].content
    
    prompt = f"""你是一个纯粹的后台路由程序，负责判断用户最新一句话的意图分类。
【严厉警告】：绝对不要去回答文本里的问题！你的唯一工作就是打标签！
    
请根据以下【历史对话】上下文，判断用户的【最新输入】到底属于哪个类别，并严格输出以下四个英文单词之一（不能包含任何多余字符）：
- travel （旅游、地点、导航、路线规划）
- joke （讲笑话、幽默故事）
- couplet （对联、作诗、下联）
- other （非以上情况）

【历史对话】（帮你理解上下文）：
{history_text}

【最新输入】（你需要分类的目标）：
<text>
{user_input}
</text>

请直接输出英文分类标签："""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    raw_result = response.content.strip().lower() 
    print(f"--- [内部日志] 路由分类结果: '{raw_result}' ---")
    
    final_type = "other"
    for node in ["travel", "joke", "couplet"]:
        if node in raw_result:
            final_type = node
            break
            
    return {"type": final_type}

async def travel_node(state: State):
    print("\n>>> 正在执行: travel_node")

    system_prompt = """你是一个专业的旅游规划助手。
请严格遵循以下步骤完成用户的路线规划请求：
1. 获取坐标：调用地点搜索工具，分别获取起点和终点的精确经纬度（禁止凭空捏造坐标）。
2. 路线规划：传入获取到的经纬度，调用路线规划工具获取导航数据。
3. 总结输出：根据工具返回的真实数据，用中文提炼出不超过300字的简洁路线建议。
"""
    # 【全局记忆】：将 SystemPrompt 与完整的历史对话列表合并
    prompts = [SystemMessage(content=system_prompt)] + state["messages"]

    amap_api_key = os.environ.get("AMAP_MAPS_API_KEY", "")
    if not amap_api_key:
        raise ValueError("请在 .env 文件中配置 AMAP_MAPS_API_KEY")

    client = MultiServerMCPClient({
        "amap-maps": {
            "command": "npx",
            "args": ["-y", "@amap/amap-maps-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": amap_api_key},
            "transport": "stdio"
        }
    })

    tools = await client.get_tools()
    agent = create_react_agent(model=llm, tools=tools)
    response = await agent.ainvoke({"messages": prompts})
    
    return {"messages": [AIMessage(content=response["messages"][-1].content)], "type": "travel"}

async def joke_node(state: State):
    print("\n>>> 正在执行: joke_node")

    system_prompt = "你是一个幽默大师。请根据用户需求，用中文创作一个短小精悍的笑话，字数严格控制在100字以内。如果你发现用户想听更多的笑话，请结合上下文讲一个全新的。"
    
    # 【全局记忆】：将 SystemPrompt 与完整的历史对话列表合并
    prompts = [SystemMessage(content=system_prompt)] + state["messages"]

    response = await llm.ainvoke(prompts)
    return {"messages": [AIMessage(content=response.content.strip())], "type": "joke"}

async def couplet_node(state: State):
    print("\n>>> 正在执行: couplet_node")
    
    query = state["messages"][-1].content # 用最新的一句话去数据库检索

    # 初始化 Redis 向量检索
    embedding_model = DashScopeEmbeddings(model="text-embedding-v1")
    redis_url = "redis://localhost:6379"
    redis_config = RedisConfig(redis_url=redis_url, index_name="couplet")
    vector_store = RedisVectorStore(config=redis_config, embeddings=embedding_model)

    # 检索参考数据
    samples = []
    scored_results = vector_store.similarity_search_with_score(query, k=10)
    for i, (doc, score) in enumerate(scored_results, 1):
        samples.append(doc.page_content)
    
    samples_str = "\n".join(samples)

    system_prompt = f"""你是一位精通中国传统文化的对联大师。
请结合我们之前的对话历史，以及用户最新的要求，创作出【下联】。

为了让你更好地把握语感，以下是从经典对联库中为你检索到的【参考对联】：
<参考对联>
{samples_str}
</参考对联>

创作规范：
1. 字数完全相等。
2. 词性严格相对（名词对名词，动词对动词等）。
3. 意境相融且呼应，不可直接照抄参考对联。
4. 仅输出下联文本，禁止包含拼音、解释或任何多余的客套话。
"""
    
    # 【全局记忆】：摒弃之前的 PromptTemplate，直接将 SystemPrompt 和完整的历史记录合并发给大模型
    prompts = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = await llm.ainvoke(prompts)
    
    return {"messages": [AIMessage(content=response.content.strip())], "type": "couplet"}

async def other_node(state: State):
    print("\n>>> 正在执行: other_node")
    return {"messages": [AIMessage(content="抱歉，我暂时无法回答这个问题，请尝试询问关于路线规划、讲笑话或对对联的问题。")], "type": "other"}

# 路由控制
def routing_func(state: State):
    node_mapping = {
        "travel": "travel_node",
        "joke": "joke_node",
        "couplet": "couplet_node",
        END: END
    }
    return node_mapping.get(state["type"], "other_node")
    
builder = StateGraph(State)

builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)

builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_func)

builder.add_edge("travel_node", END)
builder.add_edge("joke_node", END)
builder.add_edge("couplet_node", END)
builder.add_edge("other_node", END)

# 开启记忆功能
graph = builder.compile(checkpointer=InMemorySaver())

async def main():
    config = {"configurable": {"thread_id": "1"}}
    print("================ 流程开始 ================")
    print("🤖 Agent 已就绪。输入 'quit' 或 'exit' 退出对话。")

    while True:
        user_input = input("\n👨‍💻 你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 再见！")
            break
            
        if not user_input.strip():
            continue

        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates"
        ):
            node_name = list(chunk.keys())[0]
            print(f"  [后台状态] {node_name} 处理完毕")
        
        final_state = await graph.aget_state(config)
        
        latest_msg = final_state.values["messages"][-1].content
        print(f"🤖 AI助手: {latest_msg}")

if __name__ == "__main__":
    asyncio.run(main())