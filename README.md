# TinyAgent

一个基于 OpenAI 接口的轻量级工具调用 Agent 示例。通过 tool_calls 机制，模型可在对话中动态调用预定义的 Python 工具函数来完成任务。该项目已集成到 Happy-LLM（Chapter 7.3），后续更新请关注 Happy-LLM。

- 仓库地址（当前）：本项目
- 集成位置：https://github.com/datawhalechina/happy-llm

## 功能概览

- 多轮对话管理：维护 system/user/assistant/tool 角色消息
- 动态工具调用：提供工具 JSON Schema，模型触发 tool_calls 后由 Agent 执行并回传结果，再次补全生成最终答复
- 可扩展工具集：通过新增带注释的函数即可扩展
- 语言自适应：输出与用户语言保持一致（中文优先）

当前内置工具
- 时间：get_current_datetime
- 数学：add、mul、compare
- 文本：count_letter_in_string
- 检索：search_wikipedia（返回前三条摘要）

## 代码结构

- src/core.py：Agent 核心逻辑（消息管理、两阶段补全、工具调用处理）
- src/utils.py：function_to_json，将函数签名与注释转换为 OpenAI tools 所需 JSON Schema
- src/tools.py：工具函数实现
- demo.py：命令行交互示例
- web_demo.py：Streamlit Web 演示界面
- images/：示意图与动图

## 工作流程

1) 接收用户输入并加入历史
2) 携带工具 Schema 调用模型
3) 如触发 tool_calls：执行对应 Python 工具函数，并将结果以 tool 消息加入历史
4) 再次调用模型生成最终回复
5) 返回给用户并记录

示意图

<div style="display: flex; justify-content: center;">
  <img src="./images/Tiny_Agent.jpg" style="width: 80%;" />
</div>

## 快速开始

准备
- Python 3.9+
- 申请并配置兼容 OpenAI 的 API Key（示例使用 SiliconFlow）

安装依赖

```bash
pip install -r requirements.txt
```

命令行示例

```bash
# 设置你的 API Key（示例）
export OPENAI_API_KEY="your_siliconflow_api_key"
# 或在 demo.py 中直接替换 api_key

python demo.py
```

Web 示例（Streamlit）

```bash
# 在 web_demo.py 中替换 api_key 后运行
streamlit run web_demo.py
```

## 配置说明

OpenAI 客户端（示例使用 SiliconFlow 端点）

```python
from openai import OpenAI
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.siliconflow.cn/v1",
)
```

Agent 构造

```python
from src.core import Agent
from src.tools import get_current_datetime, add, compare, count_letter_in_string, search_wikipedia

agent = Agent(
    client=client,
    model="Qwen/Qwen2.5-32B-Instruct",
    tools=[get_current_datetime, add, compare, count_letter_in_string, search_wikipedia],
)
```

## 安全与注意事项

- 生产环境请避免使用 eval 动态执行函数，建议维护安全的函数映射并进行参数校验
- search_wikipedia 需要外网访问
- 一些工具返回字符串以便直接插入对话，实际集成可改为返回结构化数据

## 扩展工具

新增一个工具的步骤：
1) 在 src/tools.py 中新增带 docstring 的函数（包含参数类型注释）
2) 在创建 Agent 时把该函数加入 tools 列表
3) function_to_json 会自动将其转换为工具 Schema

## 许可证

本项目采用 Apache-2.0 许可证，详见 LICENSE。

## 致谢

- 感谢 Happy-LLM 项目对本仓库的集成与示例展示。
- 如果觉得有帮助，欢迎点个 Star！




### Step 1: 初始化客户端和模型

首先，我们需要一个能够调用大模型的客户端。这里我们使用 `openai` 库，并配置其指向一个兼容 OpenAI API 的服务终端，例如 [SiliconFlow](https://cloud.siliconflow.cn/i/ybUFvmqK)。同时，指定要使用的模型，如 `Qwen/Qwen2.5-32B-Instruct`。

```python
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="YOUR_API_KEY",  # 替换为你的 API Key
    base_url="https://api.siliconflow.cn/v1", # 使用 SiliconFlow 的 API 地址
)

# 指定模型名称
model_name = "Qwen/Qwen2.5-32B-Instruct"
```

> **注意:** 你需要将 `YOUR_API_KEY` 替换为你从 [SiliconFlow](https://cloud.siliconflow.cn/i/ybUFvmqK) 或其他服务商获取的有效 API Key。

### Step 2: 定义工具函数

我们在 `src/tools.py` 文件中定义 Agent 可以使用的工具函数。每个函数都需要有清晰的文档字符串（docstring），描述其功能和参数，因为这将用于自动生成工具的 JSON Schema。

```python
# src/tools.py
from datetime import datetime

# 获取当前日期和时间
def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def add(a: float, b: float):
    """
    计算两个浮点数的和。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 两个浮点数的和。
    """
    return a + b

def compare(a: float, b: float):
    """
    比较两个浮点数的大小。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 比较结果的字符串表示。
    """
    if a > b:
        return f'{a} is greater than {b}'
    elif a < b:
        return f'{b} is greater than {a}'
    else:
        return f'{a} is equal to {b}'

def count_letter_in_string(a: str, b: str):
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    return a.count(b)

# ... (可能还有其他工具函数)
```

为了让 OpenAI API 理解这些工具，我们需要将它们转换成特定的 JSON Schema 格式。这可以通过 `src/utils.py` 中的 `function_to_json` 辅助函数完成。

```python
# src/utils.py (部分)
import inspect

def function_to_json(func) -> dict:
    # ... (函数实现细节)
    # 返回符合 OpenAI tool schema 的字典
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

### Step 3: 构造 Agent 类

我们在 `src/core.py` 文件中定义 `Agent` 类。这个类负责管理对话历史、调用 OpenAI API、处理工具调用请求以及执行工具函数。

```python
# src/core.py (部分)
from openai import OpenAI
import json
from typing import List, Dict, Any
from utils import function_to_json
# 导入定义好的工具函数
from tools import get_current_datetime, add, compare, count_letter_in_string

SYSREM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools # 存储可用的工具函数列表
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSREM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 使用 utils.function_to_json 获取所有工具的 JSON Schema
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理来自模型的工具调用请求
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        # 动态执行工具函数
        # 注意：实际应用中应添加更严格的安全检查
        function_call_content = eval(f"{function_name}(**{function_args})")

        # 返回工具执行结果给模型
        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:
        # 主对话逻辑
        self.messages.append({"role": "user", "content": prompt})

        # 第一次调用模型，传入工具 Schema
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )

        # 检查模型是否请求调用工具
        if response.choices[0].message.tool_calls:
            tool_list = []
            # 处理所有工具调用请求
            for tool_call in response.choices[0].message.tool_calls:
                # 执行工具并将结果添加到消息历史中
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append(tool_call.function.name)
            if self.verbose:
                print("调用工具：", tool_list)
            
            # 第二次调用模型，传入工具执行结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(), # 再次传入 Schema 可能有助于模型理解上下文
                stream=False,
            )

        # 将最终的助手回复添加到消息历史
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
```

这个 Agent 的工作流程如下：
1.  接收用户输入。
2.  调用大模型（如 Qwen），并告知其可用的工具及其 Schema。
3.  如果模型决定调用工具，Agent 会解析请求，执行相应的 Python 函数。
4.  Agent 将工具的执行结果返回给模型。
5.  模型根据工具结果生成最终回复。
6.  Agent 将最终回复返回给用户。

<div style="display: flex; justify-content: center;">
    <img src="./images/Tiny_Agent.jpg" style="width: 80%;">
</div>

### Step 4: 运行 Agent

现在我们可以实例化并运行 Agent。在 `demo.py` 的 `if __name__ == "__main__":` 部分提供了一个简单的命令行交互示例。

```python
# demo.py (部分)
if __name__ == "__main__":
    client = OpenAI(
        api_key="YOUR_API_KEY", # 替换为你的 API Key
        base_url="https://api.siliconflow.cn/v1",
    )

    # 创建 Agent 实例，传入 client、模型名称和工具函数列表
    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
        verbose=True # 设置为 True 可以看到工具调用信息
    )

    # 开始交互式对话循环
    while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答
```

运行 `python src/core.py` 后，你可以开始提问。如果问题需要调用工具，Agent 会自动处理。

**示例交互:**

```bash
User: 你好
Assistant:  你好！有什么可以帮助你的吗？
User: 9.12和9 .2哪个更大？
调用工具： ['compare']
Assistant:  9.2 比 9.12 更大。
User: 为什么？
Assistant:  当我们比较9.12和9.2时，可以将它们看作是9.12和9.20。由于9.20在小数点后第二位是0，而9.12在小数点后第二位是2，所以在小数点后第一位相等的情况下，9.20（即9.2）大于9.12。因此，9.2 比 9.12 更大。
User: strawberry中有几个r？
调用工具： ['count_letter_in_string']
Assistant:  单词 "strawberry" 中有3个字母 'r'。
User: 你确信嘛？
调用工具： ['count_letter_in_string']
Assistant:  是的，我确定。单词 "strawberry" 中确实有3个字母 'r'。让我们再次确认一下，"strawberry" 中的 'r' 确实出现了3次。
User: 好的 你很薄，现在几点 了？
调用工具： ['get_current_datetime']
Assistant:  当前的时间是2025年4月26日17:01:33。不过，我注意到您提到“你很薄”，这似乎是一个打字错误，如果您有任何其他问题或者需要进一步的帮助，请告诉我！
User: exit
```

> ***记得给仓库点个小小的 star 哦~***
