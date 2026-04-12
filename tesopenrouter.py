from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
# Example usage
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = model.invoke(messages)
print(ai_msg.content)