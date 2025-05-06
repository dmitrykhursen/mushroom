from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path


class FactExtractorSettings(BaseModel):
    openai_api_key: str | None = None
    embedding_model: str | None = None
    chat_model: str | None = None
    max_tokens: int | str | None = None


class RetrievalSettings(BaseModel):
    pass


class SpanLabelingSettings(BaseModel):
    pass


class Settings(BaseSettings):
    """
    Load and manage configuration settings for the app and each pipeline step.
    """
    
    project_root: str = Path(__file__).absolute().parent.parent.parent.parent.as_posix()

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
        env_file = ".env.example", ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        env_nested_delimiter = "__"


settings = Settings()

if __name__ == "__main__":
    print(settings)
