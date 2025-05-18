from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent.parent.as_posix()


class FactExtractorSettings(BaseModel):
    openai_api_key: str | None = None
    api_key : str | None = None
    api_base_url: str | None = None
    model_name: str | None = None
    embedding_model: str | None = None
    chat_model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class RetrievalSettings(BaseModel):
    embed_model_name: str = 'all-MiniLM-L6-v2'
    llm_model_name: str = "phi-4"
    top_k: int = 10
    api_base_url: str = "http://localhost:1234/v1"
    api_key: str | None = "dummy_key"
    wiki_lang_code: str = "en"
    max_tokens: int | str | None = 1024
    temperature: float | str | None = 0.7
    top_p: float | str | None = 0.9


class SpanLabelingSettings(BaseModel):
    api_base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    max_tokens: int | str | None = 2048
    temperature: float | str | None = 0.7
    top_p: float | str | None = 0.9
    seed: int | str | None = 42


class Settings(BaseSettings):
    """
    Load and manage configuration settings for the app and each pipeline step.
    """
    
    project_root: str = PROJECT_ROOT

    fact_extractor: FactExtractorSettings = Field(
        default_factory=FactExtractorSettings,
        description="Settings for the fact extractor step.",
    )
    retrieval: RetrievalSettings = RetrievalSettings(
        default_factory=RetrievalSettings,
        description="Settings for the retrieval step.",
    )
    span_labeling: SpanLabelingSettings = SpanLabelingSettings(
        default_factory=SpanLabelingSettings,
        description="Settings for the span labeling step.",
    )

    class Config:
        extra = "allow"
        env_file =  Path(PROJECT_ROOT) / ".env.example", Path(PROJECT_ROOT) / ".env",
        env_file_encoding = "utf-8"
        case_sensitive = True
        env_nested_delimiter = "__"


settings = Settings()

if __name__ == "__main__":
    print(settings)
