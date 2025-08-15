import json
import os
from dotenv import load_dotenv

load_dotenv()  

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#step1: Define method that your tool will use

def get_weather(latitude, longitude):
    """
    Get the weather for a given latitude and longitude
    """
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true")
    data = response.json()
    print (data)
    return data["current_weather"]

# Step2: provide tool definition

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given latitude and longitude in celcius",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    }
]

# make call to OpenAI API

messages = [
    {"role": "system", "content": "You are a weather assistant."},
    {"role": "user", "content": "What's the weather in Morristown, NJ today?"},
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# see if model call function?

completion.model_dump_json()
#messages.append(completion.choices[0].message.model_dump())


# Step3: go through the response and see if the model called the function and if it did we execute the method defined earlier

def call_method(name, arguments):
    if name == "get_weather":
        return get_weather(arguments["latitude"], arguments["longitude"])

for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"Calling {name} with {arguments}")
    result = call_method(name, arguments)
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": tool_call.function.arguments
            }
        }]
    })
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

# Step4: make a second call to the model with the new messages

class WeatherResponse(BaseModel):
    temperature: float = Field(description="The current temperature in celcius for the given location")
    response: str = Field(description="A natural language response to the user's question")

completion_2 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)

#final step: check model response

final_response = completion_2.choices[0].message.content
print(final_response)