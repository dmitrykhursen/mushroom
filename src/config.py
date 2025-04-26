from pydantic import BaseModel
from dotenv import load_dotenv
import os 

load_dotenv()

class Config(BaseModel):
    OPENAI_API_KEY : str = ""

    EMBEDDING_MODEL : str = ""
    CHAT_MODEL : str = ""

    MAX_TOKENS : int | str

    @classmethod
    def load_from_env(cls):
        """Creates an isntance of Config with walues from environment variables."""
        env_values = {
            field: os.getenv(field, "")
            for field in cls.model_fields.keys()
        }
        return cls(**env_values)
    
config = Config.load_from_env()