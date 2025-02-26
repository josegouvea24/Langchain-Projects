from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

llm_model = "gpt-4o"
proxy_client = get_proxy_client('gen-ai-hub')

################################
# 1 - ConversationBufferMemory #
################################

llm = ChatOpenAI(temperature=0.0, proxy_model_name=llm_model, proxy_client=proxy_client)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True # print the conversation logs upon response
)

conversation.predict(input="Hi, my name is Andrew")

conversation.predict(input="What is 1+1?")
# The model will recall the previous conversation and answer the question
conversation.predict(input="What is my name?")

# Obtain conversation log
print(memory.buffer)
memory.load_memory_variables({})

# Adding context to the memory
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})

print(memory.buffer)

######################################
# 2 - ConversationBufferWindowMemory #
######################################

from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1) # 1 is the window size, i.e. the number of previous conversational exchanges the model will keep in memory
           
           
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

# Only the last exchange is kept in memory
memory.load_memory_variables({})

memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
# The model will not recall the first exchange and therefore will not be able to answer the question
conversation.predict(input="What is my name?")

#####################################
# 3 - ConversationTokenBufferMemory #
#####################################

from langchain.memory import ConversationTokenBufferMemory

# max_token_limit is the maximum number of tokens the model will keep in memory
# tokens are defined diferently by different llms, they represent the smallest unit of text, e.g. words, punctuation, etc., 
# tokens are more closely related to computational resources than conversational exchanges
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50) # deprecated
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

memory.load_memory_variables({})

#######################################
# 3 - ConversationSummaryBufferMemory #
#######################################

from langchain.memory import ConversationSummaryBufferMemory

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100) # deprected
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

print(memory.load_memory_variables({}))

#######################################
# 4 - ConversationSummaryBufferMemory #
#######################################

from langchain.memory import ConversationSummaryBufferMemory

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

# The model will keep in memory the last 100 tokens of the conversation
# Anything beyond that will be summarised and stored in memory
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

memory.load_memory_variables({})

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="What would be a good demo to show?")

# The model will keep in memory the most recent messages exchanged until max_token_limit is reached
# Messages unable to be FULLY stored in memory will be summarised and stored in the "System" context of the memory
# If the last response is longer than max_token_limit, the model will summarize the response and store it in the "System" context
print(memory.load_memory_variables({}))