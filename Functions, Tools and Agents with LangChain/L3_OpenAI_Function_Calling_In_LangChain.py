from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from typing import List
from pydantic import BaseModel, Field

#######################
# 1 - Pydantic Syntax #
#######################

# Simple python class
class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email
        
foo = User(name="Joe",age=32, email="joe@gmail.com")
# Passing invalid types does not produce errors
foo = User(name="Joe",age="bar", email="joe@gmail.com")

#Pydantic class
class pUser(BaseModel):
    name: str
    age: int
    email: str
    
foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")
# Passing invalid types produces an error
# foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")

# Nested Pydantic classes
class Class(BaseModel):
    students: List[pUser]
    
obj = Class(
    students=[pUser(name="Jane", age=32, email="jane@gmail.com")]
)

###########################################
# 2 - Pydantic OpenAI function definition #
###########################################

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")
    
from langchain_core.utils.function_calling import convert_to_openai_function

weather_function = convert_to_openai_function(WeatherSearch)
print(weather_function)

# Define a Pydantic class with a description
class WeatherSearch1(BaseModel):
    airport_code: str = Field(description="airport code to get weather for")
    
# This will output an error due to the missing description
convert_to_openai_function(WeatherSearch)

# Define a Pydantic class without an argument description
class WeatherSearch2(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str

# This won't output an error since argument descriptions are optional in openai functions
convert_to_openai_function(WeatherSearch2)

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')
llm_model = "gpt-35-turbo"

model = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client)

# Call the model with the function
print(model.invoke("what is the weather in SF today?", functions=[weather_function]))

# Bind the function to the model
model_with_function = model.bind(functions=[weather_function])
print(model_with_function.invoke("what is the weather in SF today?"))

###########################
# 3 - Using it in a chain #
###########################

from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])

chain = prompt | model_with_function

print(chain.invoke({"input": "what is the weather in SF today?"}))

################################
# 4 - Using multiple functions #
################################

class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")
    
functions = [
    convert_to_openai_function(WeatherSearch),
    convert_to_openai_function(ArtistSearch)
]

model_with_functions = model.bind(functions=functions)

print(model_with_functions.invoke("what is the weather in SF today?"))
print(model_with_functions.invoke("what are the top 5 songs by the Beatles?"))
print(model_with_functions.invoke("hi!"))

