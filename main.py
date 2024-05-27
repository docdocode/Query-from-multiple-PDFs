import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.service_context import ServiceContext

# from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# For manually setting the GPT model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=1000)
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000, chunk_overlap=20)

query = input("Ask your question: ")

document = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(document, service_context=service_context)

def top_response():
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    pprint_response(response)

def dynamic_retrieval(number_of_results):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=number_of_results)
    query_engine = RetrieverQueryEngine(retriever=retriever)
    response = query_engine.query(query)
    pprint_response(response, show_source=True)

# top_response()
dynamic_retrieval(3)


