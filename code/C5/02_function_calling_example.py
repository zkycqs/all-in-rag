from openai import OpenAI
import os
import json
import requests

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ★ 新增：真实的天气查询函数（用 wttr.in，免 API key，支持中文城市名）
def get_weather(location: str) -> str:
    """真实调用天气接口

    模型的参数形如 "杭州市, 浙江省"，而 wttr.in 只需要城市名，所以先按逗号取第一段。
    """
    city = location.split(",")[0].strip()
    try:
        resp = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1", "lang": "zh"},
            timeout=15,
        )
        resp.raise_for_status()
        c = resp.json()["current_condition"][0]
        desc = c["weatherDesc"][0]["value"]
        return (f"{city}：{desc}，气温 {c['temp_C']}℃（体感 {c['FeelsLikeC']}℃），"
                f"湿度 {c['humidity']}%，观测时间 {c['observation_time']}（UTC）")
    except Exception as e:
        return f"天气查询失败：{type(e).__name__}: {e}"


# 定义一个函数，用于发送消息并获取模型的响应
def send_messages(messages, tools=None):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # 让模型自主决定是否调用工具
    )
    return response.choices[0].message

# 1. 定义工具（函数）的 Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和省份，例如：杭州市, 浙江省",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

# 1. 用户提问，模型决策调用工具
messages = [{"role": "user", "content": "杭州今天天气怎么样？"}]
print(f"User> {messages[0]['content']}\n")
message = send_messages(messages, tools=tools)

# 2. 执行工具，并将结果返回模型
if message.tool_calls:
    print("--- 模型发起了工具调用 ---")
    messages.append(message)  # 将模型的回复（含 tool_calls）加入历史

    # ★ 修复：遍历「所有」工具调用（原版只取 tool_calls[0]，模型一次返回多个时会漏）
    for tool_call in message.tool_calls:
        function_info = tool_call.function
        print(f"工具名称: {function_info.name}")
        print(f"工具参数(JSON字符串): {function_info.arguments}")

        # ★ 修复：arguments 是 JSON 字符串，必须先 json.loads 成 dict 才能当参数用
        args = json.loads(function_info.arguments)
        print(f"工具参数(已解析): {args}")

        # ★ 替换原来的写死数据，改为真实调用天气接口
        print("--- 真实调用天气接口 ---")
        tool_output = get_weather(**args)
        print(f"接口返回: {tool_output}\n")

        # 将工具的执行结果作为一个新的消息添加到历史中
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_output})

    # 3. 第二次调用：将工具结果返回给模型，获取最终回答
    print("--- 将工具结果返回给模型，获取最终答案 ---")
    final_message = send_messages(messages, tools=tools)
    print(f"Model> {final_message.content}")
else:
    # 如果模型没有调用工具，直接打印其回答
    print(f"Model> {message.content}")
