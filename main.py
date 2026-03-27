import requests

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

### TOOLS ###
@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    print('DEBUG: sending api')
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    print('DEBUG: got the answer', response.json())
    return response.json()

### MODEL ###
model = ChatOllama(model="lfm2.5-thinking", temperature=0)

### SYSTEM PROMPT ###
system_prompt = "You are a helpful weather assistant, who always ends up with a joke while remaining helpful."


agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=system_prompt,
)
response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like in trappes?'}
    ]
})
print(response)
print(response['messages'][-1].content)


