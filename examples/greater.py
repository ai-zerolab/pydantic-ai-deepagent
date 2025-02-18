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
        model_name="anthropic.claude-3-5-haiku-20241022-v1:0"
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

"""
Usage(
    requests=1,
    request_tokens=2116,
    response_tokens=1301,
    total_tokens=3417,
    details={
        "reasoning_tokens": 976,
        "cached_tokens": 0,
        "reasoning_request_tokens": 18,
        "reasoning_response_tokens": 1260,
        "reasoning_total_tokens": 1278,
        "execution_request_tokens": 2098,
        "execution_response_tokens": 41,
        "execution_total_tokens": 2139,
    },
)
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content="\nYou are the EXECUTION model of an LLM Agent system. Your role is to analyze the reasoning process provided in <Thinking></Thinking> tags and determine the most appropriate tool calls to accomplish the task.\n\nWhen you receive reasoning output, you should:\n\n1. Parse the thinking process carefully to identify:\n   - The specific task requirements\n   - Any constraints or conditions\n   - The logical steps needed for completion\n\n2. For each identified step that requires tool interaction:\n   - Select the most appropriate tool from your available toolkit\n   - Format the tool call with the necessary parameters\n   - Consider any error handling or fallback options\n\n3. To minimize your response time, you can just select the tool without saying what you're thinking.\n\nOnly make tool calls that are directly supported by the reasoning process.\nIf the reasoning is unclear or insufficient, choose a tool that best meets the needs as much as possible.\n",
                dynamic_ref=None,
                part_kind="system-prompt",
            ),
            UserPromptPart(
                content="9.11 and 9.8, which is greater?",
                timestamp=datetime.datetime(
                    2025, 2, 18, 14, 11, 8, 233695, tzinfo=datetime.timezone.utc
                ),
                part_kind="user-prompt",
            ),
        ],
        kind="request",
    ),
    ModelResponse(
        parts=[
            TextPart(
                content="<Thinking>Okay, so I need to figure out whether 9.11 is greater than 9.8 or if 9.8 is greater than 9.11. Hmm, let's start by comparing these two numbers step by step. \n\nFirst, I remember that when comparing decimals, you look at each place value from left to right. So starting with the ones place. Both numbers have a 9 in the ones place, so they are equal there. That means I need to move to the next digit to the right, which is the tenths place. \n\nIn 9.11, the tenths place is 1, and in 9.8, the tenths place is 8. Wait, so 1 versus 8 in the tenths place. Since 8 is greater than 1, does that mean 9.8 is greater than 9.11? But hold on, 9.8 is the same as 9.80, right? Because you can add a zero at the end of a decimal without changing its value. So maybe if I write both numbers with the same number of decimal places, it will be easier to compare. \n\nLet me write them out:\n\n9.11\n9.80\n\nNow comparing them digit by digit after the decimal. The tenths place: 1 vs. 8. Since 8 is larger, 9.80 is bigger than 9.11. Therefore, 9.8 is greater than 9.11. \n\nBut wait, let me make sure I'm not making a mistake here. Sometimes when there are different numbers of decimal places, it can be confusing. Let me think of another way. If I convert both numbers to fractions, maybe that will help. \n\n9.11 is the same as 9 + 11/100, and 9.8 is the same as 9 + 8/10. To compare 11/100 and 8/10, I can convert them to have the same denominator. The denominators here are 100 and 10. The least common denominator is 100. \n\nSo, 8/10 is equal to 80/100. Now, comparing 11/100 and 80/100, clearly 80/100 is larger. Therefore, 9.8 is greater than 9.11. \n\nAnother way to think about it is by subtracting one from the other. Let's do 9.8 minus 9.11. \n\n9.80\n-9.11\n------\n0.69\n\nSince the result is positive, 9.8 is larger by 0.69. \n\nBut maybe I can also visualize this on a number line. If I imagine the numbers between 9 and 10, 9.11 is just a little past 9.1, whereas 9.8 is much closer to 10. So 9.8 is definitely further to the right on the number line, meaning it's greater. \n\nWait, but sometimes people get confused because 9.11 has two digits after the decimal and 9.8 only has one. But adding a zero to 9.8 to make it 9.80 helps in comparing. Since 9.80 has 8 tenths and 0 hundredths, versus 9.11 which has 1 tenth and 1 hundredth. So even though 9.11 has more digits, the tenths place is way more important in determining the value. \n\nLet me also check with money. If I think of these as amounts of money, $9.11 versus $9.80. Which is more? Well, $9.80 is 9 dollars and 80 cents, while $9.11 is 9 dollars and 11 cents. Clearly, 80 cents is more than 11 cents, so $9.80 is more. \n\nAlternatively, if I convert both to cents. $9.11 is 911 cents, and $9.80 is 980 cents. 980 is greater than 911, so again, 9.8 is greater. \n\nI think I've verified this multiple ways: aligning decimal places, converting to fractions, subtracting, using a number line, and real-life money examples. All of them point to 9.8 being greater than 9.11. \n\nJust to recap, the key point is that the tenths place is more significant than the hundredths place when comparing decimals. So even though 9.11 has a 1 in the hundredths place, the tenths place difference (8 vs. 1) is much more impactful. \n\nTherefore, the conclusion is that 9.8 is greater than 9.11.\n<\\Thinking>\n\n\n\nTo determine whether 9.11 or 9.8 is greater, follow these steps:\n\n1. **Align the decimal places** by rewriting 9.8 as 9.80:\n   - \\(9.11\\)\n   - \\(9.80\\)\n\n2. **Compare digit by digit**:\n   - **Ones place**: Both have \\(9\\), so they are equal.\n   - **Tenths place**: Compare \\(1\\) (in 9.11) vs. \\(8\\) (in 9.80). Since \\(8 > 1\\), \\(9.80\\) is greater here.\n\n3. **Verification**:\n   - Convert to fractions:  \n     \\(9.11 = 9 + \\frac{11}{100}\\) and \\(9.8 = 9 + \\frac{80}{100}\\).  \n     Clearly, \\(\\frac{80}{100} > \\frac{11}{100}\\).\n   - Subtract: \\(9.80 - 9.11 = 0.69\\) (positive result confirms \\(9.80 > 9.11\\)).\n   - Real-world analogy: \\(9.80\\) dollars is 80 cents, which is more than \\(9.11\\) dollars (11 cents).\n\n**Conclusion**: \\(9.8\\) is greater than \\(9.11\\).\n\n\\(\\boxed{9.8}\\)",
                part_kind="text",
            )
        ],
        model_name="deepseek-r1-250120",
        timestamp=datetime.datetime(
            2025, 2, 18, 14, 11, 53, tzinfo=datetime.timezone.utc
        ),
        kind="response",
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content="Please use a tool accroding the reasoning result.",
                timestamp=datetime.datetime(
                    2025, 2, 18, 14, 11, 54, 19986, tzinfo=datetime.timezone.utc
                ),
                part_kind="user-prompt",
            )
        ],
        kind="request",
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="final_result",
                args={"result": 9.8},
                tool_call_id="tooluse_YcdXsLvKSpOxb9WpLqd-6Q",
                part_kind="tool-call",
            )
        ],
        model_name="anthropic.claude-3-5-haiku-20241022-v1:0",
        timestamp=datetime.datetime(
            2025, 2, 18, 14, 11, 56, 264701, tzinfo=datetime.timezone.utc
        ),
        kind="response",
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="final_result",
                content="Final result processed.",
                tool_call_id="tooluse_YcdXsLvKSpOxb9WpLqd-6Q",
                timestamp=datetime.datetime(
                    2025, 2, 18, 14, 11, 56, 267170, tzinfo=datetime.timezone.utc
                ),
                part_kind="tool-return",
            )
        ],
        kind="request",
    ),
]
"""
