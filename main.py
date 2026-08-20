from agents.simple_agent import SimpleAgent
from core.llm import HelloAgentsLLM
from dotenv import load_dotenv
from tools.builtin.calculator import CalculatorTool

load_dotenv()

llm = HelloAgentsLLM()

# 或手动指定provider（可选）
# llm = HelloAgentsLLM(provider="modelscope")

agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个有用的AI助手"
)

response = agent.run("你好！请介绍一下自己")
print(response)

# 添加工具功能（可选）
calculator = CalculatorTool()
# 需要实现7.4.1的MySimpleAgent进行调用，后续章节会支持此类调用方式
# agent.add_tool(calculator)

response = agent.run("Calculate 2 + 3 * 4")
print(response)

print(f"History Length: {len(agent.get_history())}")
