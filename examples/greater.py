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

"""
Usage(
    requests=1,
    request_tokens=1992,
    response_tokens=1393,
    total_tokens=3385,
    details={
        "reasoning_tokens": 1065,
        "cached_tokens": 0,
        "reasoning_request_tokens": 18,
        "reasoning_response_tokens": 1263,
        "reasoning_total_tokens": 1281,
        "execution_request_tokens": 1974,
        "execution_response_tokens": 130,
        "execution_total_tokens": 2104,
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
                    2025, 2, 18, 8, 3, 18, 208853, tzinfo=datetime.timezone.utc
                ),
                part_kind="user-prompt",
            ),
        ],
        kind="request",
    ),
    ModelResponse(
        parts=[
            TextPart(
                content="<Thinking>Okay, let's see. I need to figure out whether 9.11 is greater than 9.8 or if 9.8 is greater than 9.11. Hmm, comparing decimals can sometimes be a bit tricky because of the different number of digits after the decimal point. Let me start by writing both numbers down: 9.11 and 9.8. \n\nFirst, I remember that when comparing decimals, it's helpful to make sure they have the same number of decimal places. That way, you can compare them digit by digit. So, 9.11 already has two decimal places, but 9.8 only has one. Maybe I can add a zero to 9.8 to make it 9.80. Does that change the value? No, because adding a zero at the end of a decimal doesn't change its value. So, 9.8 is the same as 9.80.\n\nNow, both numbers are 9.11 and 9.80. Let's compare them starting from the left. The whole number part is 9 for both, so they are equal there. Moving to the tenths place, the first digit after the decimal. For 9.11, the tenths place is 1, and for 9.80, the tenths place is 8. Hmm, so 1 versus 8. Since 8 is greater than 1, that means 9.80 has a larger tenths place than 9.11. \n\nWait, but does that automatically make 9.80 greater than 9.11? Let me think. If the whole numbers are the same, then we compare the tenths. If the tenths are different, the one with the higher tenths digit is greater, regardless of the hundredths or any further digits. So, even though 9.11 has a 1 in the hundredths place and 9.80 has a 0, since the tenths place of 9.80 is higher (8 vs. 1), 9.80 is larger.\n\nTherefore, 9.8 (or 9.80) is greater than 9.11. Let me double-check to make sure I didn't mix up any digits. Let's convert both numbers to fractions to see if that helps. \n\n9.11 can be written as 9 + 11/100, which is 9 + 0.11. 9.8 is 9 + 8/10, which is 9 + 0.8. Comparing 0.11 and 0.8. Clearly, 0.8 is larger than 0.11. So yes, 9.8 is greater.\n\nAlternatively, maybe I can subtract them to see which is larger. Let's subtract 9.11 from 9.8. 9.8 minus 9.11. Let's line them up:\n\n```\n 9.80\n-9.11\n-------\n```\n\nStarting from the hundredths place: 0 - 1. Can't do that, so borrow from the tenths place. The tenths place is 8, so borrow 1, making it 7, and the hundredths place becomes 10. 10 - 1 = 9.\n\nThen tenths place: 7 - 1 = 6.\n\nWhole number: 9 - 9 = 0.\n\nSo the result is 0.69. Since the result is positive, that means 9.8 is larger than 9.11 by 0.69.\n\nAnother way to think about it is in terms of money. If these are amounts of money, $9.11 versus $9.80. Which is more? $9.80 is 9 dollars and 80 cents, while $9.11 is 9 dollars and 11 cents. Definitely, 80 cents is more than 11 cents, so $9.80 is more. That makes sense.\n\nWait, sometimes people get confused because 9.11 has two digits after the decimal and might think it's larger. For example, if you see 9.11 and 9.8, maybe someone could mistakenly think 9.11 is bigger because 11 is more than 8. But that's not how decimals work. It's about the place value. The first digit after the decimal is tenths, so 8 tenths is way more than 1 tenth. The second digit is hundredths, so even if 9.11 has 1 tenth and 1 hundredth, it's still only 0.11, whereas 9.8 is 0.8. So 0.8 is greater than 0.11.\n\nI think that's solid. So 9.8 is greater than 9.11. Yeah, that seems right. All the different methods—expanding as fractions, subtracting, using money examples—lead to the same conclusion. So I can confidently say that 9.8 is greater than 9.11.\n\n**Final Answer**\nThe greater number is \\boxed{9.8}.\n<\\Thinking>\n\n\n\nTo determine whether 9.11 or 9.8 is greater, we can compare them by ensuring they have the same number of decimal places. Converting 9.8 to 9.80 allows for a direct comparison:\n\n- 9.11 (which is 9 + 0.11)\n- 9.80 (which is 9 + 0.80)\n\nComparing the tenths place:\n- The tenths digit of 9.11 is 1.\n- The tenths digit of 9.80 is 8.\n\nSince 8 (in the tenths place) is greater than 1 (in the tenths place), 9.80 is greater than 9.11. This can also be verified by subtracting 9.11 from 9.80, which results in a positive value (0.69), confirming that 9.80 is larger.\n\nThus, the greater number is \\boxed{9.8}.",
                part_kind="text",
            )
        ],
        model_name="deepseek-r1-250120",
        timestamp=datetime.datetime(2025, 2, 18, 8, 4, 5, tzinfo=datetime.timezone.utc),
        kind="response",
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content="Please use a tool accroding the reasoning result.",
                timestamp=datetime.datetime(
                    2025, 2, 18, 8, 4, 5, 234859, tzinfo=datetime.timezone.utc
                ),
                part_kind="user-prompt",
            )
        ],
        kind="request",
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='<thinking> I need to provide the final answer to the question "9.11 and 9.8, which is greater?" based on the reasoning provided earlier. Since the reasoning process has concluded that 9.8 is greater than 9.11, I will use the "final_result" tool to encapsulate this conclusion as the final response to the user\'s query.</thinking>\n',
                part_kind="text",
            ),
            ToolCallPart(
                tool_name="final_result",
                args={"result": 9.8},
                tool_call_id="tooluse_Iwmr778FTvqABH028tf5GA",
                part_kind="tool-call",
            ),
        ],
        model_name="us.amazon.nova-micro-v1:0",
        timestamp=datetime.datetime(
            2025, 2, 18, 8, 4, 7, 289295, tzinfo=datetime.timezone.utc
        ),
        kind="response",
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="final_result",
                content="Final result processed.",
                tool_call_id="tooluse_Iwmr778FTvqABH028tf5GA",
                timestamp=datetime.datetime(
                    2025, 2, 18, 8, 4, 7, 292865, tzinfo=datetime.timezone.utc
                ),
                part_kind="tool-return",
            )
        ],
        kind="request",
    ),
]
"""
