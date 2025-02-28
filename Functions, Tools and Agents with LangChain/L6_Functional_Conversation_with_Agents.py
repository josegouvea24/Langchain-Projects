from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

########################
# 1 - Define the Tools #
########################

from langchain.tools import tool
import requests
from pydantic import BaseModel, Field
import datetime
import wikipedia

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

# Define the weather tool
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
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'


# Define the wikipedia search tool
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

tools = [get_current_temperature, search_wikipedia]

########################
# 2 - Define the Chain #
########################

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

proxy_client = get_proxy_client('gen-ai-hub')
llm_model = "gpt-35-turbo"

# Create the model functions from tools 
functions = [convert_to_openai_function(f) for f in tools]

# Create the chat model and bind the functions
model = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client, temperature=0.0).bind(functions=functions)

# Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])

# Create the chain
chain = prompt | model | OpenAIFunctionsAgentOutputParser()

# Test the chain
result = chain.invoke({"input": "what is the weather in SF today?"})
print(result)
print(result.tool)
print(result.tool_input)

##############################################
# 2 - Feeding the result back into the chain #
##############################################

from langchain.prompts import MessagesPlaceholder

# Adding a MessagesPlaceholder to the prompt to pass in the action, observation pairs to the chat model after a function call
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Redeclare the chain with the new prompt
chain = prompt | model | OpenAIFunctionsAgentOutputParser()

result1 = chain.invoke({
    "input": "what is the weather is sf?",
    "agent_scratchpad": []
})

print(result1.tool)

observation = get_current_temperature.invoke(result1.tool_input)
print(observation)

type(result1)
print(result1.message_log)

# Feeding the result and observation back into the chain
from langchain.agents.format_scratchpad import format_to_openai_functions

# Format a list of result and observation to be passed into the agent scratchpad
format_to_openai_functions([(result1, observation), ])

# Call the chain with the new agent scratchpad
result2 = chain.invoke({
    "input": "what is the weather is sf?", 
    "agent_scratchpad": format_to_openai_functions([(result1, observation)])
})

# Result is now an AgentFinish type
print(result2)

###############################
# 3 - Creating the Agent Loop #
###############################

from langchain.schema.agent import AgentFinish
def run_agent(user_input):
    # Initialize the result, observation list to pass into the chain at each step
    intermediate_steps = []
    # Whilw the output is not an AgentFinish type
    while True:
        # Call the chain with the intermediate steps list
        result = chain.invoke({
            "input": user_input, 
            "agent_scratchpad": format_to_openai_functions(intermediate_steps) # format the intermediate steps list
        })
        # If the result is an AgentFinish type, return it
        if isinstance(result, AgentFinish):
            return result
        # Else call the tool and push the result, observation pair to the intermediate steps list
        tool = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))
        
# Transfering the scratchpad formating step into the chain with a RunnablePassthrough
from langchain.schema.runnable import RunnablePassthrough

# RunnablePassthrough extracts theintermediate_steps from the input, 
# formats them and assigns them to the agent_scratchpad key in the next chain element, i.e. the prompt
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | chain

# Redifine the loop function for the new agent_chain
def run_agent(user_input):
    intermediate_steps = []
    while True:
        result = agent_chain.invoke({
            "input": user_input, 
            "intermediate_steps": intermediate_steps
        })
        if isinstance(result, AgentFinish):
            return result
        tool = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))
        
print(run_agent("what is the weather in SF today?"))
print(run_agent("what is LangChain?"))
print(run_agent("hi"))

###########################
# 4 - Using AgentExecutor #
###########################

# AgentExecutor is a utility class that can be used to run the agent loop with extra regidity:
# Improved logging capabilities
# Error handling: for model outputs not in JSON format and tool errors (model is asked to correct the input and try again)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "what is Pycharm"}))

# Agent still has no chat memory capabilities
print(agent_executor.invoke({"input": "my name is bob"}))
print(agent_executor.invoke({"input": "what is my name"}))

#############################
# 4 - Adding Memory Support #
#############################

# Redefine the prompt to include a MessagesPlaceholder for the chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()

from langchain.memory import ConversationBufferMemory

# ConversationBufferMemory is a memory tool that stores the chat history in a list
# return_messages=True ensures the chat history is returned as a list of messages instead of a string
memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

# Reinitialize the AgentExecutor with the new agent_chain and memory
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)

# Test the agent's recall
print(agent_executor.invoke({"input": "my name is bob"}))
print(agent_executor.invoke({"input": "what is my name"}))

print(agent_executor.invoke({"input": "what is the weather in SF today?"}))
print(agent_executor.invoke({"input": "what is LangChain?"}))

########################
# 5 - Create a Chatbot #
########################

@tool
def get_most_popular_song(artist: str) -> str:
    """
    Given an artist's name, returns a song by that artist using the iTunes Search API.
    """
    url = "https://itunes.apple.com/search"
    params = {
        "term": artist,
        "entity": "song",
        "limit": 10  # fetch up to 10 songs
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error fetching data for artist '{artist}': {response.status_code}"
    
    data = response.json()
    if data.get("resultCount", 0) > 0:
        # Return the track name of the first result as a heuristic for the most popular song.
        return data["results"][0].get("trackName", "Unknown Song")
    else:
        return f"No songs found for artist '{artist}'."
    
tools = [get_current_temperature, search_wikipedia, get_most_popular_song]

# Adding a UI
import panel as pn  # GUI
pn.extension()
import panel as pn
import param

class cbfs(param.Parameterized):
    
    last_query = ''
    
    def __init__(self, tools, **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.functions = [convert_to_openai_function(f) for f in tools]
        self.model = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client, temperature=0.0).bind(functions=functions).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful but sassy assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.chain = RunnablePassthrough.assign(
            agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.model | OpenAIFunctionsAgentOutputParser()
        self.qa = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory)
    
    def convchain(self, query):
        if not query:
            return
        if self.last_query != query:
            result = self.qa.invoke({"input": query})
            self.answer = result['output'] 
            self.panels.extend([
                pn.Row('User:', pn.pane.Markdown(query, width=450)),
                pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=450, styles={'background-color': '#F6F6F6'}))
            ])
        inp.value = ''
        self.last_query = query
        return pn.WidgetBox(*self.panels, scroll=True)


    def clr_history(self,count=0):
        self.chat_history = []
        return 
    
    
cb = cbfs(tools)

inp = pn.widgets.TextInput( placeholder='Enter text here…')

conversation = pn.bind(cb.convchain, inp) 

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation,  loading_indicator=True, height=400),
    pn.layout.Divider(),
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# QnA_Bot')),
    pn.Tabs(('Conversation', tab1))
)
dashboard.show()