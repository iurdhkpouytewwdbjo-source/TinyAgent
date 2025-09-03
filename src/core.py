from openai import OpenAI
import json
from typing import List, Dict, Any
from src.utils import function_to_json
from src.tools import get_current_datetime, add, compare, count_letter_in_string, search_wikipedia

import pprint

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose
        # 将可用工具构建为名称到函数的安全映射
        self.tool_map = {t.__name__: t for t in self.tools}

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments or "{}"
        function_id = tool_call.id

        # 安全解析参数并调用映射中的函数
        try:
            args = json.loads(function_args)
        except Exception:
            args = {}
        func = self.tool_map.get(function_name)
        if func is None:
            result = f"Unknown tool: {function_name}"
        else:
            try:
                result = func(**args)
            except Exception as e:
                result = f"Tool execution error in {function_name}: {e}"

        return {
            "role": "tool",
            "content": str(result),
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        if response.choices[0].message.tool_calls:
            response_message = response.choices[0].message
            # 先把包含 tool_calls 的 assistant 消息（包括其 tool_calls）加入消息历史
            assistant_msg = {
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response_message.tool_calls
                ],
            }
            self.messages.append(assistant_msg)

            # 执行每个工具并将结果加入历史
            tool_list = []
            for tool_call in response_message.tool_calls:
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("调用工具：", response_message.content, tool_list)

            # 带着工具结果再次请求模型
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 将模型的完成响应添加到消息列表中
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content


    

