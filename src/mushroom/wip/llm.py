from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from config import config


def get_embedding_model() -> OpenAIEmbeddings:
    """
    This function initializes and returns an instance of OpenAIEmbeddings.
    It uses the OpenAI API key from the config module.
    """    
    # Create an instance of OpenAIEmbeddings using the API key from config
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )

    return embeddings

def get_chat_model(temperature=0) -> ChatOpenAI:
    """
    This function initializes and returns an instance of ChatOpenAI.
    It uses the OpenAI API key from the config module.
    """
    # Create an instance of ChatOpenAI using the API key from config
    chat = ChatOpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        model=config.CHAT_MODEL,
        temperature=temperature,
        max_tokens=config.MAX_TOKENS,
    )
    
    return chat