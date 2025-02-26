from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

##############################
# 1 - Loading a PDF Document #
##############################

from langchain_community.document_loaders import PyPDFLoader

# Load the document
loader = PyPDFLoader("docs/pdf/SAPTEC_EN_Col23_Part_A4.pdf")
pages = loader.load()

print(len(pages))

# Visualize document metadata
page_0 = pages[0]
print(page_0.metadata)

##################################
# 2 - Loading a YouTube Document #
##################################

from langchain_community.document_loaders.generic import GenericLoader,  FileSystemBlobLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain_community.document_loaders import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"

# Create the generic loader by loading the youtube audio and parsing it
# Using the OpenAIWhisperParser requires an API key to be set in the .env file
# loader = GenericLoader(
#     #YoutubeAudioLoader([url],save_dir),  # fetch from youtube
#     FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally (once you have downloaded the file)
#     OpenAIWhisperParser()
# )

# Visualize the first
# docs = loader.load()
# print(docs[0].page_content[0:500])

######################################
# 3 - Loading a Document through URL #
######################################

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")

docs = loader.load()
print(docs[0].page_content[:500])

######################################
# 4 - Loading a Document from Notion #
######################################

from langchain_community.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()

print(docs[0].page_content[0:200])

docs[0].metadata