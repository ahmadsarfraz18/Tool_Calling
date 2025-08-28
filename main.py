from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings, AsyncOpenAI, RunConfig, function_tool
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from .env file
load_dotenv()

# Ensure the GEMINI_API_KEY is set in your environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. ")

# Initialize the AsyncOpenAI client with Gemini API key
external_client = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define the model with the external client
model = OpenAIChatCompletionsModel(
    model= "gemini-2.5-flash",
    openai_client= external_client
)

# Define the run configuration
config= RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled=True
)

@function_tool
def calculator(a: int, b: int) -> int:
    return a * b

@function_tool
def usd_to_pkr(query: str) -> str:
    return "To convert USD to PKR, multiply the amount in USD by 280. "

@function_tool
def get_current_weather(city: str) -> str:
    return f"The current weather in {city} is sunny with a temperature of 25°C. "

# Create the base agent with tools
base_agent= Agent(
    name= "CurrencyConverterAgent",
    instructions= "Yor are a helpful assistant that helps the user with their queries and also use the tools if needed. ",
    tools= [calculator, usd_to_pkr, get_current_weather],
)

# Clone the base agent to create a creative agent
creative_agent= base_agent.clone(
    name= "Creative Agent",
    instructions= "You are a creative agent that can think outside the box and use the tools creatively to help the user with their queries. ",

)

# Run the agent with a query
result= Runner.run_sync(
    creative_agent,
    # "If I have 100 USD, how much PKR will I get?",
    # "what is the current weather of New York?",
    "what is 15 multiplied by 3?",
    run_config= config
)
# Print the final output
print(result.final_output)
