![](https://img.shields.io/github/license/wh1isper/pydantic-ai-deepagent)
![](https://img.shields.io/github/v/release/wh1isper/pydantic-ai-deepagent)
![](https://img.shields.io/docker/image-size/wh1isper/pydantic-ai-deepagent)
![](https://img.shields.io/pypi/dm/pydantic_ai_deepagent)
![](https://img.shields.io/github/last-commit/wh1isper/pydantic_ai_deepagent)
![](https://img.shields.io/pypi/pyversions/pydantic_ai_deepagent)

# pydantic-ai-deepagent

This is a pydantic model to implement [reasoning response](https://github.com/pydantic/pydantic-ai/issues/907) with tool use.

⚠️ This is not a official project of PydanticAI, And PydanticAI is in early beta, the API is still subject to change and there's a lot more to do. Feedback is very welcome!

## Install

`pip install pydantic_ai_deepagent`

## Usage

TBD

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
