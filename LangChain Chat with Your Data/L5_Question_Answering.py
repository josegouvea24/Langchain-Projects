from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain_community.vectorstores import Chroma
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model

embedding = init_embedding_model('text-embedding-ada-002')

persist_directory = 'docs/chroma/'
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

# Test the vector store with a similarity search
question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')
llm_model = "gpt-35-turbo"

# Initialize the llm
llm = ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client, temperature=0.0)

#########################
# 1 - RetrievalQA Chain #
#########################

from langchain.chains import RetrievalQA

# Initialize the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

result = qa_chain.invoke({"query": question})
print(result["result"])

##############
# 2 - Prompt #
##############

from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
Context: {context}
Question: {question}
Helpful Answer:"""

# Initialize the prompt template
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Initialize the RetrievalQA chain with the prompt
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "what is the most powerful ML method taught in the lectures?"
result = qa_chain.invoke({"query": question})
print(result["result"])
print(result["source_documents"][0])

###############################
# 3 - RetrievalQA Chain Types #
###############################

# Map_reduce chain will retrive the k most relevant documents from the vector store
# Then it will pass each document to the LLM to generate a summary of any relevant information contained
# The summaries are then passed to the context of the final LLM along with the query to generate the final response
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)

result = qa_chain_mr.invoke({"query": question})
print(result["result"])

# Reduce chain selects the k most relevant documents from the vector store
# It passes the documents one at a time to the LLM to generate a summary of any relevant information
# The summary obtained at each step is passed down in the next LLM call along with another relevant document
# The response is build iteratively until all documents are processed
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr.invoke({"query": question})
print(result["result"])

###############################
# 3 - RetrievalQA Limitations #
###############################

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

question = "Is probability a class topic?"
result = qa_chain({"query": question})
print(result["result"])

question = "why are those prerequesites needed?"
result = qa_chain({"query": question})
print(result["result"])
