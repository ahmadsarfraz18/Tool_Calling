from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
import requests
import random

load_dotenv()
set_tracing_disabled(disabled=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
                                   
    model = "gemini-2.0-flash",
    openai_client = provider,
)
# 1st method of toll calling.
@function_tool
def how_many_jokes():
    """
    Get the random number of jokes
    """
    return random.randint(1, 10)

# 2nd method of toll calling.
@function_tool
def get_weather(city: str) -> str:
    """
    Get the weather of the given city.
    """
    try:
        result = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key=8e3aca2b91dc4342a1162608252604&q={city}"
        )

        data = result.json()
        return f"The weather of {city} is {data["current"]["temp_c"]}ºC with{data["current"]["condition"]["text"]}."
    
    except Exception as e :
        return f"Couldn't fetch weather data due to {e}"



agent = Agent(
    name= "Assistant",
    instructions= """
if the user asks the jokes, first call 'how_many_jokes' function, then tell the jokes with numbers.
if the user asks for weather, call the 'get_weather' function with the city name
""",
    model = model,
    tools=[how_many_jokes, get_weather]
)

result = Runner.run_sync(
    agent,
    input = "Tell me the weather of chak No 18/14L ?"
)

print(result.final_output)

