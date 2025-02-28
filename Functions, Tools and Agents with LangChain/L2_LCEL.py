from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

####################
# 1 - Simple Chain #
####################

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')
llm_model = "gpt-35-turbo"

prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client)
output_parser = StrOutputParser()

# Define chain in linux pipe syntax
chain = prompt | model | output_parser

print(chain.invoke({"topic": "bears"}))

##########################
# 2 - More complex chain #
##########################

from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from langchain_community.vectorstores import DocArrayInMemorySearch

# Vectorstore refresher
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=init_embedding_model('text-embedding-ada-002')
)

retriever = vectorstore.as_retriever()

print(retriever.get_relevant_documents("where did harrison work?"))
print(retriever.get_relevant_documents("what do bears like to eat?"))

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

# Define a chain using linux pipe syntax
# The RunnableMap will collect the relevant documents from the retriever and the new query so that they can be inserted into the prompt
inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

chain = inputs | prompt | model | output_parser

print(inputs.invoke({"question": "where did harrison work?"}))
print(chain.invoke({"question": "where did harrison work?"}))

#########################
# 3 - Binding functions #
#########################

functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)

# Bind the function to the model
model = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client).bind(functions=functions)

runnable = prompt | model

# Returns AIMessage type with correct function call
print(runnable.invoke({"input": "what is the weather in sf"}))

# Changing the models binded functions
functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]

model = model.bind(functions=functions)

runnable = prompt | model

# Returns AIMessage type with correct function call
print(runnable.invoke({"how did the patriots do yesterday?"}))

#################
# 3 - Fallbacks #
#################

from gen_ai_hub.proxy.langchain.openai import OpenAI
import json

# No available models in SAP AI Launchpad support the completion operation
simple_model = OpenAI(proxy_model_name='tiiuae--falcon-40b-instruct', proxy_client=proxy_client)

simple_chain = simple_model | json.loads

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"

# print(simple_model.invoke(challenge))

# Deliver an json decode error because the models output is not actually in json format
# print(simple_chain.invoke(challenge))

model = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client, temperature=0.0)
# Output parser converts model output into a string
chain = model | StrOutputParser() | json.loads

# The call now works because we are using a chat model
print(chain.invoke(challenge))

# Defining fallbacks
final_chain = simple_chain.with_fallbacks([chain])

print(final_chain.invoke(challenge))

#################
# 4 - Interface #
#################

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Simple invoke call
print(chain.invoke({"topic": "bears"}))
# Batch call (the queries are executed in parallel, as much as possible)
print(chain.batch([{"topic": "bears"}, {"topic": "cats"}]))
# Stream call
for i in chain.stream({"topic": "bears"}):
    print(i)

# Asynchronous invoke call
import asyncio

async def async_invoke():
    response = await chain.ainvoke({"topic": "bears"})
    print(response)

asyncio.run(async_invoke())