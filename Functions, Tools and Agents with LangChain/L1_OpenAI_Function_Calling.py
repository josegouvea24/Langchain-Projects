from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# define an OpenAI JSON function object
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

# Create a query message to send to the OpenAI API which should evoke the function use
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]

from gen_ai_hub.proxy.native.openai import chat

# Call the OpenAI API to get a response
response = chat.completions.create(deployment_id="d9ff9ca9eab7c12a", messages=messages, functions=functions)

# Observe the response 
# The API does not call the function, it simply returns the right function and arguments to use according to the message
print(response)
response_message = response.choices[0].message
print(response_message)
args = json.loads(response_message.function_call.arguments)
print(args)
print(get_current_weather(args))

# Message unrelated to the function
messages_1 = [
    {
        "role": "user",
        "content": "hi!",
    }
]

messages_2 = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]

# Call the OpenAI API to get a response
response = chat.completions.create(deployment_id="d9ff9ca9eab7c12a", messages=messages_1, functions=functions)
# Model does not return the weather function now
print(response)

# Call the OpenAI API to get a response with function_call="auto" (default setting, model decides when to call the function)
response = chat.completions.create(deployment_id="d9ff9ca9eab7c12a", messages=messages_1, functions=functions, function_call="auto")
# Model does not return the weather function
print(response)

# Call the OpenAI API to get a response with function_call="none" (model never calls the function)
response = chat.completions.create(deployment_id="d9ff9ca9eab7c12a", messages=messages_2, functions=functions, function_call="none")
# Model does not return the weather function now even though the message is related
print(response)

# Call the OpenAI API to get a response with function_call={"name": <function_name>} (model is forced to call the function)
response = chat.completions.create(deployment_id="d9ff9ca9eab7c12a", messages=messages_1, functions=functions, function_call={"name": "get_current_weather"})
# Model returns the weather function even though the message is unrelated and incompatible
print(response)

# Passing the function call results pack into the LLM

response = chat.completions.create(deployment_id="d9ff9ca9eab7c12a", messages=messages_2, functions=functions, function_call={"name": "get_current_weather"})
# Append the model's response to the messages list
messages_2.append(response.choices[0].message)

# Call the function with the arguments from the model's response
args = json.loads(response_message.function_call.arguments)
observation = get_current_weather(args)

# Appent the observation to the messages list
messages_2.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)

# Call the OpenAI API to get a response
response = chat.completions.create(deployment_id="d9ff9ca9eab7c12a", messages=messages_2)
print(response)