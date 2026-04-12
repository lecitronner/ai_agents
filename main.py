from dataclasses import dataclass
from dotenv import load_dotenv
import requests
import os

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

load_dotenv()

### DATACLASSES ###
@dataclass()
class Context:
    user_id: str

@dataclass()
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

### TOOLS ###
@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    #print('DEBUG: sending api, city', city)
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description="ALWAYS call this first to find the user's city when no city is explicitly mentioned in the message.")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return "Vienna"
        case "XYZ456":
            return "London"
        case "HJK111":
            return "Paris"
        case _:
            return "Unknown"

### MODEL ###
model = ChatOpenAI(
    model="google/gemini-flash-1.5",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

### Remembering conversation
checkpointer = InMemorySaver()

### SYSTEM PROMPT ###
system_prompt = """You are a helpful weather assistant.

IMPORTANT: If the user does not explicitly mention a city, you MUST call the 'locate_user' 
tool first to determine their city, then call 'get_weather' with that city.
Never ask the user for their city — always use the tools."""

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt=system_prompt,
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {'configurable': {'thread_id': '1'}}
response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like?'}
    ]},
    config = config,
    context=Context(user_id='ABC123')
)
print(response)
print(response['messages'][-1].content)


