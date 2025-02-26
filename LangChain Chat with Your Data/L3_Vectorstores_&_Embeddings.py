import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

##################
# 1 - Embeddings #
##################

from gen_ai_hub.proxy.langchain.init_models import init_embedding_model

embedding = init_embedding_model('text-embedding-ada-002')

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np


print(np.dot(embedding1, embedding2))
print(np.dot(embedding1, embedding3))
print(np.dot(embedding2, embedding3))

####################
# 2 - Vectorstores #
####################

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# Load PDF (This will not work since the files do not exist)
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
    
# Split the documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)
print(len(splits))

# Define a persistant directory path for the vectorstore
persist_directory = 'docs/chroma/'

# Delete the directory if it already exists
import shutil

if os.path.exists('./docs/chroma'):
    shutil.rmtree('./docs/chroma')

# Create the vectorstore
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# Check the number of documents in the vectorstore
print(vectordb._collection.count())

#########################
# 3 - Similarity Search #
#########################

# Define a query
question = "is there an email i can ask for help"

# Run similarity search returning the top k most similar documents
docs = vectordb.similarity_search(question,k=3)

len(docs)

print(docs[0].page_content)

#####################
# 3 - Failure modes #
#####################

question = "what did they say about matlab?"

docs = vectordb.similarity_search(question,k=5)

# FAILURE 1: Due to the duplicate documents, the two most similar chunks are the same (one from each loaded document)
print(docs[0].page_content == docs[1].page_content)

question = "what did they say about regression in the third lecture?"

docs = vectordb.similarity_search(question,k=5)

# FAILURE 2: The embeddings fail to capture the queries contextual information, therefore returning results not from the third lecture
for doc in docs:
    print(doc.metadata)
    
print(docs[0].page_content)