import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

############################
# 2 - Using Text Splitters #
############################

chunk_size =26
chunk_overlap=4

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# String is smaller than the chunk size so no split occurs
text1 = 'abcdefghijklmnopqrstuvwxyz'
print(r_splitter.split_text(text1))

# Observe the chunk overlap when the string is longer than the chunk size
text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
print(r_splitter.split_text(text2))

# Blank spaces count for the chunk zize but are not included in the split
text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
print(r_splitter.split_text(text3))

# Character splitter split on '\n' by default
print(c_splitter.split_text(text3))

# Character splitter with ' ' separator
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator=' '
)

print(c_splitter.split_text(text3))

###########################
# 2 - Recursive Splitters #
###########################

some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

print(len(some_text))

c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)

# Recursive splitter attempts to split the text on the listed separators, starting from left to right
# If the splitter does not find a way to split the text on a certain separator, it will look to split on the next separator in the list until it succeeds
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""] 
)

# Character splitter split
print(c_splitter.split_text(some_text))

# Recursive splitter split
print(r_splitter.split_text(some_text))

# Recursive splitter split with a smaller chunk size and a "\. " separator
# The period ends up at the start of the next chunk instead of at the end of the previous one
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
)
print(r_splitter.split_text(some_text))

# Specify different REGEX to correct the period issue
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
print(r_splitter.split_text(some_text))

#####################
# 3 - PDF Splitting #
#####################

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/pdf/SAPTEC_EN_Col23_Part_A4.pdf")
pages = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len # len is the default (defines the counting unit)
)

docs = text_splitter.split_documents(pages)
print(len(docs))
print(len(pages))

########################
# 4 - Notion Splitting #
########################

from langchain_community.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
notion_db = loader.load()

docs = text_splitter.split_documents(notion_db)
print(len(notion_db))
print(len(docs))

#######################
# 4 - Token Splitting #
#######################

from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

text1 = "foo bar bazzyfoo"

print(text_splitter.split_text(text1))

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

docs = text_splitter.split_documents(pages)

print(docs[0],'\n')

print(pages[0].metadata)

###############################
# 4 - Context Aware Splitting #
###############################

from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Initialize the markdown splitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = markdown_splitter.split_text(markdown_document)

# Observe the chunks' metadata
print(md_header_splits[0])
print(md_header_splits[2])

# Load the Notion cocuments and pack them into a single string
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
txt = ' '.join([d.page_content for d in docs])

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = markdown_splitter.split_text(txt)

print(md_header_splits[0])