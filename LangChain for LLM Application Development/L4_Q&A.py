from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

llm_model = "gpt-35-turbo"

#####################
# 1 - Load Document #
#####################

file = 'OutdoorClothingCatalog_1000.csv'
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)

docs = loader.load()

print(docs[0])

###########################
# 2 - Create Vector Store #
###########################

from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

embeddings = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002')

# Example: Embeding a query 
embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed))

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

################################################
# 3 - Query the Model through Similarity Score #
################################################

query = "Please suggest a shirt with sunblocking"
response_docs = db.similarity_search(query)

len(response_docs)
print(response_docs[0])

##########################
# 4 - Q&A over Documents #
##########################

# Initialize a retriver (generic interface which takes in query and returns docs)
retriever = db.as_retriever()

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

# Initialize a language model
proxy_client = get_proxy_client('gen-ai-hub')
llm = ChatOpenAI(temperature=0.0, proxy_model_name='gpt-35-turbo', proxy_client=proxy_client)

# Combine documents into a single string
qdocs = "".join(response_docs[i].page_content for i in range(len(response_docs)))

# Query the model
response = llm.invoke(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

from IPython.display import display, Markdown

# Display the response in markdown format
# display(Markdown(response.content).data)

#####################################################
# 4 - Encapsulate all previous steps w/ RetrievalQA #
#####################################################

from langchain.chains import RetrievalQA

# Initialize the retrieval QA chain
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", #simplest method: stuffs all documents into context, makes one query to an llm
    retriever=retriever, 
    verbose=True
)

# Query the model
query =  "Please list all your items with rain protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)

display(Markdown(response).data)

#################################################################
# 4 - Encapsulate all previous steps w/ VectorstoreIndexCreator #
#################################################################

from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])

response = index.query(query, 
                       llm = llm)

display(Markdown(response).data)
