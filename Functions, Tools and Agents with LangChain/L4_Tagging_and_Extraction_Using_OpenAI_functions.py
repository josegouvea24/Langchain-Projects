from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

###############
# 1 - Tagging #
###############

from typing import List
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function

# Define a Pydantic class
class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")
    
print(convert_to_openai_function(Tagging))

from langchain.prompts import ChatPromptTemplate
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')
llm_model = "gpt-35-turbo"

model= ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client)

tagging_functions = [convert_to_openai_function(Tagging)]

# Create the adequate prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

# Bind the function to the model, forcing the function call to be "Tagging"
model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)

# Define the chain
tagging_chain = prompt | model_with_functions

print(tagging_chain.invoke({"input": "I love langchain"}))

print(tagging_chain.invoke({"input": "non mi piace questo cibo"}))

# Adding a parser to parse the output to JSON
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

print(tagging_chain.invoke({"input": "non mi piace questo cibo"}))

##################
# 2 - Extracting #
##################

from typing import Optional

class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")
    
class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")
    
print(convert_to_openai_function(Information))

extracting_functions = [convert_to_openai_function(Information)]
# Force bind the extraction function to the model
extraction_model = model.bind(functions=extracting_functions, function_call={"name": "Information"})

# Model returns two people but is because it is not able to extract the age of the second person it returns 0
# Note: It doesn't exhibit this behavior anymore
print(extraction_model.invoke("Joe is 30, his mom is Martha"))

# Ensuring the model doesn't guess the info using a prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

# Define the chain with the prompt and a JSON parser
extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()
print(extraction_chain.invoke("Joe is 30, his mom is Martha"))

# Extracting a particular attribute
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
print(extraction_chain.invoke("Joe is 30, his mom is Martha"))

##########################
# 2 - Real World Example #
##########################

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
documents = loader.load()
doc = documents[0]
page_content = doc.page_content[:10000]
print(page_content[:1000])

class Overview(BaseModel):
    """Overview of a section of text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")
    
overview_tagging_functions = [convert_to_openai_function(Overview)]
tagging_model = model.bind(functions=overview_tagging_functions, function_call={"name": "Overview"})
tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()

# Observe extracted content
print(tagging_chain.invoke({"input": page_content}))

# Another example: Extracting the paper mentioned in the loaded article
class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]

paper_extraction_functions = [convert_to_openai_function(Info)]
extraction_model = model.bind(functions=paper_extraction_functions, function_call={"name": "Info"})
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")

# Model is confused, it returns the author of the article and not the papers mentioned withing the articles
print(extraction_chain.invoke({"input": page_content}))

# Correct models confusion with a prompt
template = """A article will be passed to you. Extract from it all papers that are mentioned by this article follow by its author. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")

# Model now returns the papers mentioned in the article
print(extraction_chain.invoke({"input": page_content}))

# Model returns an empty list if no papers are mentioned as instructed in the prompt
print(extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"}))

# Extraction on the entire article using a recursive splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
splits = text_splitter.split_text(doc.page_content)
print(len(splits))

# Define a function to flaten lists of lists
def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

# Test the function
print(flatten([[1,2,3],[4,5,6]]))

# Splits are strings and the chain expects a dictionary with an "input" key
print(splits[0])

# Define a function to convert the splits into the expected format
from langchain.schema.runnable import RunnableLambda

# RunnableLambda is a simple LangChain wrapper which takes a function(lambda) and converts it o a runnable object
prep = RunnableLambda(
    lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
)

# Test the RunnableLambda
print(prep.invoke("hi there"))

# Define the chain
# The map() call will operate on each element of the previous input, in this case a list
# Wrapping the flaten function in a RunnableLambda is optional beacuse it is not the first object in the chain
extraction_chain = prep | prompt.map() | extraction_model.map() | JsonKeyOutputFunctionsParser(key_name="papers").map() | flatten

# Extract the papers mentioned in the article
papers = extraction_chain.invoke(doc.page_content)
print(papers)