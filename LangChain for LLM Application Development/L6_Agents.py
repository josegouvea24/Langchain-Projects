from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.utilities.python import PythonREPL
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')

#########################
# 1 - Creating an Agent #
#########################

# Initialize the language model
llm_model = "gpt-35-turbo"
llm = ChatOpenAI(temperature=0.0, proxy_model_name=llm_model, proxy_client=proxy_client)

# Load some tools
# llm-match: Chain which uses and LLM in conjunction with a calculator to answer math questions
# wikipedia: API connected to wikipedia, allowing to run seacrh queries and get back results
tools = load_tools(["llm-math","wikipedia"], llm=llm)

# Initialize Agent
# agent= initialize_agent(
#     tools, 
#     llm, 
#     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # CHAT: Agent optimized to work with chat models; REACT: Prompting technique designed to get the best reasoning performance out of LLMs.
#     handle_parsing_errors=True, # When LLM outputs something that cannot be parsed into an action input (desired output), missformatted text is passed back to the LLM so that it corrects itself, handling the error gracefully.
#     verbose = True)

# agent("What is the 25% of 300?")

question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
# result = agent(question)
# print(result)

###############################
# 2 - Creating a Python Agent #
###############################

# Initialize the Python Agent
# agent = create_python_agent(
#     llm,
#     tool=PythonREPLTool(), # REPL is a tool to interact with Python code, allows to the agent to run Python code and get back results
#     verbose=True
# )

customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]

# agent.invoke(f"""Sort these customers by \
# last name and then first name \
# and print the output: {customer_list}""") 

# Observe the detailed chain outputs
import langchain
langchain.debug=True
# agent.invoke(f"""Sort these customers by \
# last name and then first name \
# and print the output: {customer_list}""") 
langchain.debug=False

############################
# 2 - Define a Costum Tool #
############################

from langchain.agents import tool #tool decorator: can be applyed to any function to make it a tool that langchain can use
from datetime import date

# Define a custom tool
# The doc string is used by the agent to determine when and how to use the tool
# Any aditional input or output requirements should be clearly defined in the doc string
@tool
def time(text: str) -> str: 
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")

