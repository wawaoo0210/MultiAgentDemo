import gradio as gr
from langchain_core.messages import HumanMessage
# 从你写好的后端文件中直接导入编译好的图
from Director import graph  

async def chat_stream(message, history):
    # 锁定同一个 thread_id，保证该网页用户的多轮对话记忆连贯
    config = {"configurable": {"thread_id": "web_user_1"}}
    
    # 1. 占位提示，告诉用户系统已经收到请求
    yield "⏳ 正在思考中..."
    
    # 2. 调用我们强大的多智能体图引擎
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="updates"
    ):
        # 获取当前正在运行的节点名称
        node_name = list(chunk.keys())[0]
        
        # 🌟 酷炫功能：在界面上实时显示哪个 Agent 正在干活
        if node_name == "supervisor_node":
            yield "🧭 前台主管正在分析您的意图..."
        elif node_name == "travel_node":
            yield "🗺️ 旅游专家正在调用高德地图..."
        elif node_name == "joke_node":
            yield "🎭 幽默大师正在为您构思笑话..."
        elif node_name == "couplet_node":
            yield "📜 对联大师正在 Redis 库中检索灵感..."
        elif node_name == "other_node":
            yield "🤔 正在转交默认助手..."

    # 3. 流程走完后，提取最终的 AI 完整回复
    final_state = await graph.aget_state(config)
    latest_msg = final_state.values["messages"][-1].content
    
    # 4. 把最终的回答输出到界面上，覆盖掉之前的过程提示
    yield latest_msg

# 构建并启动 Gradio 聊天界面
demo = gr.ChatInterface(
    fn=chat_stream,
    title="🤖 多智能体 AI 助手",
    description="支持功能：高德地图路线规划 🗺️ | 向量数据库对联检索 📜 | 幽默笑话 🎭",
    examples=[
        "讲一个笑话", 
        "花好月圆夜的下联是什么", 
        "我想从重庆大学虎溪校区前往重庆龙湖时代天街"
    ],
)

if __name__ == "__main__":
    # 启动网页服务
    demo.launch()