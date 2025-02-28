from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

#############
# 1 - Tools #
#############

# Import a tool decorator
from langchain.agents import tool

# Tool decorator converts a function into a LangChain tool
@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"

# Print the tool's name, description, and arguments
print(search.name)
print(search.description)
print(search.args)

from pydantic import BaseModel, Field

# Using Pydantic models to add argument descriptions
# Argument descriptions are important as they provide context for the LLM to chose the right inputs when calling the function
class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")
    
# Use the tool decorator with the Pydantic model, thus passing the argument descriptions to the function
@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"

# Arguments are now described
print(search.args)
print(search("hi"))

### A more complex example ###

import requests
from pydantic import BaseModel, Field
import datetime

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    # Parse the response into a JSON
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    # Get the forecast closest to the current time
    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'

print(get_current_temperature.name)
print(get_current_temperature.description)
print(get_current_temperature.args)

# Creating a tool from the function
from langchain.tools.render import format_tool_to_openai_function

print(format_tool_to_openai_function(get_current_temperature))

# Calling the tool
print(get_current_temperature({"latitude": 13, "longitude": 14}))

### Another example ###

import wikipedia

# This function searches Wikipedia for the given query and returns summaries of the top 3 pages.
# It uses the Wikipedia API to search for page titles and fetches the summaries of the top 3 results.
# If a page cannot be retrieved due to a PageError or DisambiguationError, it skips that page.
# If no summaries are found, it returns a message indicating that no good results were found.
# Otherwise, it returns the summaries concatenated with newline characters.
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

print(search_wikipedia.name)
print(search_wikipedia.description)

print(format_tool_to_openai_function(search_wikipedia))

print(search_wikipedia({"query": "langchain"}))

# Interacting with open APIs using OPenAPI specifications
from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain_community.tools import OpenAPISpec

# Generate OpenAPI specification data
text = """
{
  "openapi": "3.0.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "license": {
      "name": "MIT"
    }
  },
  "servers": [
    {
      "url": "http://petstore.swagger.io/v1"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "How many items to return at one time (max 100)",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 100,
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A paged array of pets",
            "headers": {
              "x-next": {
                "description": "A link to the next page of responses",
                "schema": {
                  "type": "string"
                }
              }
            },
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pets"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPets",
        "tags": [
          "pets"
        ],
        "responses": {
          "201": {
            "description": "Null response"
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Expected response to a valid request",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "required": [
          "id",
          "name"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "format": "int64"
          },
          "name": {
            "type": "string"
          },
          "tag": {
            "type": "string"
          }
        }
      },
      "Pets": {
        "type": "array",
        "maxItems": 100,
        "items": {
          "$ref": "#/components/schemas/Pet"
        }
      },
      "Error": {
        "type": "object",
        "required": [
          "code",
          "message"
        ],
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          }
        }
      }
    }
  }
}
"""

# Create the OpenAPISpec from the text
spec = OpenAPISpec.from_text(text)

# Extract the service's functions and callables
pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)

print(pet_openai_functions)

# Binding the service functions to a chat model
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')
llm_model = "gpt-35-turbo"

model= ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client, temperature=0.0).bind(functions=pet_openai_functions)

# Test the chat model
print(model.invoke("what are three pets names"))
print(model.invoke("what is the name of the pet with id 1"))

###############
# 2 - Routing #
###############

functions = [
    format_tool_to_openai_function(f) for f in [
        search_wikipedia, get_current_temperature
    ]
]

model= ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client, temperature=0.0).bind(functions=functions)

print(model.invoke("what is the weather in SF today?"))
print(model.invoke("what is langchain?"))

# Adding a prompt before the model call
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])

chain = prompt | model

print(chain.invoke({"input": "what is the weather in SF today?"}))
print(chain.invoke({"input": "what is langchain?"}))

# Convert into a more usable format
# The function either calls a tool or not
# When it does call a tool, we are interested in which tool it's calling and the input arguments
# When it's not calling a tool, we are interested in the message 'content'
# The OpenAIFunctionsAgentOutputParser takes care of this
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

# Test a query which should trigger a function call
result = chain.invoke({"input": "what is the weather in SF today?"})

# Parse returns an AgentActionMessageLog type
print(type(result))
print(result.tool)
print(result.tool_input)

# The tool input can be used directly in a function call
print(get_current_temperature(result.tool_input))

# Test a query which shouldn't trigger a function call
result = chain.invoke({"input": "hi!"})

# Parse returns an AgentFinish type
print(type(result))
print(result.return_values)

# Adding a route function to handle the chain output and call the appropriate tool directly,
# if the query ask for it, i.e. if the chain output is not AgentFinish type
from langchain.schema.agent import AgentFinish

# Route function to handle the chain output
# Returns the chain output or calls the right tool (with the right input) based on the chain output type
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)

# Define the complete chain
chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route

# Test the chain
result = chain.invoke({"input": "What is the weather in san francisco right now?"})
print(result)

result = chain.invoke({"input": "What is langchain?"})
print(result)

result = chain.invoke({"input": "hi!"})
print(result)