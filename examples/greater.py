import os

from pydantic import BaseModel
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai_bedrock.bedrock import BedrockModel

from pydantic_ai_deepagent.deepagent import DeepAgentModel
from pydantic_ai_deepagent.reasoning import DeepseekReasoningModel

DEEPSEEK_R1_MODEL_NAME = os.getenv("DEEPSEEK_R1_MODEL_NAME")
DEEPSEEK_R1_API_KEY = os.getenv("DEEPSEEK_R1_API_KEY")
DEEPSEEK_R1_BASE_URL = os.getenv("DEEPSEEK_R1_BASE_URL")

model = DeepAgentModel(
    reasoning_model=DeepseekReasoningModel(
        model_name=DEEPSEEK_R1_MODEL_NAME,
        api_key=DEEPSEEK_R1_API_KEY,
        base_url=DEEPSEEK_R1_BASE_URL,
    ),  # Any model's Textpart is reasoning content
    execution_model=BedrockModel(
        model_name="us.amazon.nova-micro-v1:0"
    ),  # Any other model can use tool call, e.g. OpenAI
)


class BiggerNumber(BaseModel):
    result: float


system_prompt = """
You are the EXECUTION model of an LLM Agent system. Your role is to analyze the reasoning process provided in <Thinking></Thinking> tags and determine the most appropriate tool calls to accomplish the task.

When you receive reasoning output, you should:

1. Parse the thinking process carefully to identify:
   - The specific task requirements
   - Any constraints or conditions
   - The logical steps needed for completion

2. For each identified step that requires tool interaction:
   - Select the most appropriate tool from your available toolkit
   - Format the tool call with the necessary parameters
   - Consider any error handling or fallback options

3. To minimize your response time, you can just select the tool without saying what you're thinking.

Only make tool calls that are directly supported by the reasoning process.
If the reasoning is unclear or insufficient, choose a tool that best meets the needs as much as possible.
"""


agent = Agent(
    model=model,
    result_type=BiggerNumber,  # Execution model will use tool call for this type
    system_prompt=system_prompt,  # This is only given to the execution model.
)

if __name__ == "__main__":

    with capture_run_messages() as messages:
        try:
            result = agent.run_sync("9.11 and 9.8, which is greater?")
            print(result.data)
            print(result.usage())
        except Exception as e:
            print(e)
        finally:
            print(messages)
