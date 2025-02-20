![](https://img.shields.io/github/license/wh1isper/pydantic-ai-deepagent)
![](https://img.shields.io/github/v/release/wh1isper/pydantic-ai-deepagent)
![](https://img.shields.io/pypi/dm/pydantic_ai_deepagent)
![](https://img.shields.io/github/last-commit/wh1isper/pydantic-ai-deepagent)
![](https://img.shields.io/pypi/pyversions/pydantic_ai_deepagent)

# pydantic-ai-deepagent

[pydantic-ai](https://github.com/pydantic/pydantic-ai)'s model to implement [deepclaude](https://github.com/getAsterisk/deepclaude)-style-agent. Making models such as claude can use deepseek r1's thinking as a reference for tool use. Check the [example](./examples/greater.py).

⚠️ This is not a official project of PydanticAI, And PydanticAI is in early beta, the API is still subject to change and there's a lot more to do. Feedback is very welcome!

## WIP

- [ ] Implement StreamResponse
- [ ] Add tests

## Install

`pip install pydantic_ai_deepagent`

## Usage

```python
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

agent = Agent(model)
```

More examples can be found in [examples](examples)

## Develop

Install pre-commit before commit

```
pip install pre-commit
pre-commit install
```

Install package locally

```
pip install -e .[test]
```

Run unit-test before PR, **ensure that new features are covered by unit tests**

```
pytest -v
```
