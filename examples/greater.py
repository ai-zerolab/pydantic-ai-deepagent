import os

from pydantic import BaseModel
from pydantic_ai import Agent
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
    result: str


agent = Agent(
    model=model,
    result_type=BiggerNumber,  # Execution model will use tool call for this type
    system_prompt="You are a helpful assistant. You muse use a tool.",  # This is only given to the execution model.
)

if __name__ == "__main__":
    result = agent.run_sync("9.11 and 9.8, which is greater?")
    print(result.data)
    print(result.usage)
