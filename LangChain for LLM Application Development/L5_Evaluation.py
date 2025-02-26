from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings

proxy_client = get_proxy_client('gen-ai-hub')

########################
# 1 - Initialize Model #
########################

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

# Initialize embedding model
embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)

# Initialize the vectorstore index
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding_model
).from_loaders([loader])

# Initialize the language model
llm_model = "gpt-35-turbo"
llm = ChatOpenAI(temperature=0.0, proxy_model_name=llm_model, proxy_client=proxy_client)

# Initialize the retriever QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

###################################
# 2 - Generate Evaluation Queries #
###################################

# Observe data points to create some evaluation queries
print(data[11])
print(data[10])

# Initialize evaluation queries based on the observed data
examples = [
    {
        "query": "What materials is the Lightweight Sleeping Bag made out of?",
        "answer": "Nylon and Polyester"
    },
    {
        "query": "Will the Waterproof Watch work at 150m depth?",
        "answer": "No"
    }
]

######################################################
# 3 - Generate Evaluation Queries w/ QAGenerateChain #
######################################################

from langchain.evaluation.qa import QAGenerateChain

# Initialize the QA chain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(proxy_model_name=llm_model, proxy_client=proxy_client))

# Parse some documents
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

new_examples = [example.get('qa_pairs') for example in new_examples]

# Observe the data
print(new_examples[0])
print(data[0])

examples += new_examples

#########################
# 4 - Manual Evaluation #
#########################

# Test an example
qa.invoke(examples[0]["query"])

# Enable debug for more data
import langchain
langchain.debug = True

# Test an example
qa.invoke(examples[0]["query"])

##############################################
# 5 - LLM Assisted Evaluation w/ QAEvalChain #
##############################################

langchain.debug = False

#Create predictions for all the examples
predictions = qa.apply(examples)

# Observe the predictions
print(predictions[0])

from langchain.evaluation.qa import QAEvalChain

# Initialize the evaluation chain
llm = ChatOpenAI(temperature=0.0, proxy_model_name=llm_model, proxy_client=proxy_client)
eval_chain = QAEvalChain.from_llm(llm)

# Evaluate the predictions
graded_outputs = eval_chain.evaluate(examples, predictions)

# Observe the evaluation output
print(graded_outputs[0])

# Print the evaluation results
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['results'])
    print()