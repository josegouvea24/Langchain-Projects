from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain_community.vectorstores import Chroma
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
persist_directory = 'docs/chroma/'

embedding = init_embedding_model('text-embedding-ada-002')

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

#########################
# 1 - Similarity Search #
#########################

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Tell me about all-white mushrooms with large fruiting bodies"

# Simple similarity search
smalldb.similarity_search(question, k=2)

# MMR search
smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)

#################################
# 2 - Addressing Diversity: MMR #
#################################

question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question,k=3)

print(docs_ss[0].page_content[:100])
print(docs_ss[1].page_content[:100])

# Notice the diference in the results obtained using MMR
docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)

print(docs_mmr[0].page_content[:100])
print(docs_mmr[1].page_content[:100])

############################################
# 3 - Addressing Specificity with metadata #
############################################

question = "what did they say about regression in the third lecture?"

# Manual implementation of the filter
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
)

for d in docs:
    print(d.metadata)

# Using the metadata filter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from gen_ai_hub.proxy.langchain.openai import OpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
    
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

document_content_description = "Lecture notes"

proxy_client = get_proxy_client('gen-ai-hub')
model_name = 'gpt-3.5-turbo'

# Initialize the LLM
# llm = OpenAI(proxy_model_name=model_name, proxy_client=proxy_client, temperature=0.0)

# Create the SelfQuery retriever
# retriever = SelfQueryRetriever.from_llm(
#     llm, #LLM
#     vectordb, #Vectorstore
#     document_content_description, #Description of the document content
#     metadata_field_info, #Metadata field information
#     verbose=True 
# )

question = "what did they say about regression in the third lecture?"

# docs = retriever.get_relevant_documents(question)

# This outputs: 
# query='regression' filter=Comparison(comparator=<Comparator.EQ: 'eq'>, 
# attribute='source', value='docs/cs229_lectures/MachineLearning-Lecture03.pdf') limit=None

# Output metadata shows all documents originate from the third lecture
# for d in docs:
#     print(d.metadata)
    
##############################
# 4 - Contextual Compression #
##############################

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def pretty_print_docs(docs):
    """
    This function takes a list of documents and prints them in a formatted manner.
    Each document is printed with a header indicating its number, followed by its content.
    The documents are separated by a line of 100 hyphens for better readability.
    
    Args:
        docs (list): A list of document objects, where each document has a 'page_content' attribute.
    """
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# llm = OpenAI(proxy_model_name=model_name, proxy_client=proxy_client)

# The LLMChainExtractor extracts only the relevant parts of each document and pass them as the response  
# compressor = LLMChainExtractor.from_llm(llm)

# Create the compression retriever which will compress the documents before returning them
# This will still return some repeated content since it still employs similarity search
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=vectordb.as_retriever()
# )

question = "what did they say about matlab?"
# compressed_docs = compression_retriever.get_relevant_documents(question)
# pretty_print_docs(compressed_docs)

"""
This would return:
Document 1:

- "those homeworks will be done in either MATLA B or in Octave"
- "I know some people call it a free ve rsion of MATLAB"
- "MATLAB is I guess part of the programming language that makes it very easy to write codes using matrices, to write code for numerical routines, to move data around, to plot data."
- "there's also a software package called Octave that you can download for free off the Internet."
- "it has somewhat fewer features than MATLAB, but it's free, and for the purposes of this class, it will work for just about everything."
- "once a colleague of mine at a different university, not at Stanford, actually teaches another machine learning course."
----------------------------------------------------------------------------------------------------
Document 2:

- "those homeworks will be done in either MATLA B or in Octave"
- "I know some people call it a free ve rsion of MATLAB"
- "MATLAB is I guess part of the programming language that makes it very easy to write codes using matrices, to write code for numerical routines, to move data around, to plot data."
- "there's also a software package called Octave that you can download for free off the Internet."
- "it has somewhat fewer features than MATLAB, but it's free, and for the purposes of this class, it will work for just about everything."
- "once a colleague of mine at a different university, not at Stanford, actually teaches another machine learning course."
----------------------------------------------------------------------------------------------------
Document 3:

"Oh, it was the MATLAB."
----------------------------------------------------------------------------------------------------
Document 4:

"Oh, it was the MATLAB."
"""

################################
# 5 - Combining the techniques #
################################

# Compression Retrieval with MMR 
# Returns a more diverse set of documents
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=vectordb.as_retriever(search_type = "mmr")
# )

question = "what did they say about matlab?"
# compressed_docs = compression_retriever.get_relevant_documents(question)
# pretty_print_docs(compressed_docs)

"""
This would return:
Document 1:

- "those homeworks will be done in either MATLA B or in Octave"
- "I know some people call it a free ve rsion of MATLAB"
- "MATLAB is I guess part of the programming language that makes it very easy to write codes using matrices, to write code for numerical routines, to move data around, to plot data."
- "there's also a software package called Octave that you can download for free off the Internet."
- "it has somewhat fewer features than MATLAB, but it's free, and for the purposes of this class, it will work for just about everything."
- "once a colleague of mine at a different university, not at Stanford, actually teaches another machine learning course."
----------------------------------------------------------------------------------------------------
Document 2:

"Oh, it was the MATLAB."
----------------------------------------------------------------------------------------------------
Document 3:

- learning algorithms to teach a car how to drive at reasonably high speeds off roads avoiding obstacles.
- that's a robot program med by PhD student Eva Roshen to teach a sort of somewhat strangely configured robot how to get on top of an obstacle, how to get over an obstacle.
- So I think all of these are robots that I think are very difficult to hand-code a controller for by learning these sorts of learning algorithms.
- Just a couple more last things, but let me just check what questions you have right now.
- So if there are no questions, I'll just close with two reminders, which are after class today or as you start to talk with other people in this class, I just encourage you again to start to form project partners, to try to find project partners to do your project with.
- And also, this is a good time to start forming study groups, so either talk to your friends or post in the newsgroup, but we just encourage you to try to start to do both of those today, okay? Form study groups, and try to find two other project partners.
"""

################################
# 5 - Other types of retrieval #
################################

from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

question = "What are major topics for this class?"
docs_svm=svm_retriever.get_relevant_documents(question)
docs_svm[0]

question = "what did they say about matlab?"
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
docs_tfidf[0]