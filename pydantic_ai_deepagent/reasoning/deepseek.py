from pydantic_ai.models import Model


class DeepseekReasoningModel(Model):
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @property
    def model_name(self) -> str:
        """The model name."""
        return self.model_name

    @property
    def system(self) -> str | None:
        """The system / model provider, ex: openai."""
        return "Deepseek"
