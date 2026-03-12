import random
from Director import graph

config = {
    "configurable": {
        "thread_id": random.randint(1, 10000)
    }
}

query="请给我讲一个郭德纲的笑话"

res = graph.invoke(
    {"messages": [query]},
    config=config,
    stream_mode="values"
)
print(res["messages"][-1].content)
