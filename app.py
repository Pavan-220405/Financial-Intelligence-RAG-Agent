
from flask import Flask, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
from langchain.schema import Document
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langchain.tools import StructuredTool
from llama_index.retrievers.pathway import PathwayRetriever
from langchain_community.vectorstores import PathwayVectorClient

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize LangChain components
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

client = PathwayVectorClient(
    url="http://172.30.2.194:8788",
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embd = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
retriever = PathwayRetriever(url="http://172.30.2.194:8788", similarity_top_k=10)

# LangChain workflow setup
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system_router = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to SEC fillings or financial data of multiple companies.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_router),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# Define the workflow
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    count: int
    queries: List[str]
    company_name: str
    year: str
    table: str
    mode: str

def retrieve(state):
    # Retrieve documents logic
    pass

def generate(state):
    # Generate answer logic
    pass

def grade_documents(state):
    # Grade documents logic
    pass

def transform_query(state):
    # Transform query logic
    pass

def web_search(state):
    # Web search logic
    pass

def possible_queries(state):
    # Possible queries logic
    pass

def route_question(state):
    # Route question logic
    pass

def decide_to_generate(state):
    # Decide to generate logic
    pass

def decide_after_transform(state):
    # Decide after transform logic
    pass

def grade_generation_v_documents_and_question(state):
    # Grade generation vs documents and question logic
    pass

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("possible_queries", possible_queries)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "possible_queries",
    },
)

workflow.add_edge("possible_queries", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "transform_query",
    decide_after_transform,
    {
        "web_search": "web_search",
        "retrieve": "retrieve",
    },
)
workflow.add_edge("generate", END)

# Compile the workflow
app_workflow = workflow.compile()

class DataNode(BaseModel):
    query: str = Field(description="The Query to be processed for fetching data")

def data_node_function(query: str) -> str:
    inputs = {
        "question": query,
        "count": 0,
        "documents": [],
        "generation": "",
        "mode": ""
    }
    results = app_workflow.invoke(inputs)
    return results['generation']

data_node_tool = StructuredTool.from_function(
    data_node_function,
    name="data_node_tool",
    description="""data_node_tool(query: str) -> str:
    An LLM agent with access to a structured tool for fetching internal data or online source.
    Internal data includes financial documents, SEC filings, and other financial data of various companies.
    Use it whenever you need to fetch internal data or online source.
    It can satisfy all your queries related to data retrieval.
    SEARCH SPECIFIC RULES:
        Provide concise queries to this tool, DO NOT give vague queries for search like
        - 'What was the gdp of the US for last 5 years?'
        - 'What is the percentage increase in Indian income in the last few years?'
        Instead, provide specific queries like
        - 'GDP of the US for 2020'
        - 'Income percentage increase in India for 2019'
        ALWAYS mention units for searching specific data wherever applicable and use uniform units for an entity accross queries.
        Eg: Always use 'USD' for currency,'percentage' for percentage, etc.
    INTERNAL DATA SPECIFIC RULES:
        The tool can fetch internal data like financial documents, SEC filings, and other financial data of various companies.
        The retriever is very sensitive to the query, so if you are unable to infer from the data in 1-2 queries, keep on trying again with rephrased queries

    ALWAYS provide specific queries to get accurate results.
    DO NOT try to fetch multiple data points in a single query, instead, make multiple queries.
    """,
    args_schema=DataNode,
)

# Route to serve the HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'medai.html')

# Route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data['message']

    if question.lower() == 'quit':
        return jsonify({'answer': 'Goodbye!'})

    # Use the LangChain workflow to generate the answer
    answer = data_node_tool.invoke({"query": question})
    return jsonify({'answer': answer})

if __name__ == '__main__':
    print("Server is ready. Run the application and navigate to http://localhost:5000 in your web browser.")
    app.run(debug=True)


# from flask import Flask, request, jsonify, send_from_directory
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import os

# app = Flask(__name__)

# # Download necessary NLTK data
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)

# # Load multiple text files
# def load_texts(directory_path):
#     texts = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(directory_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 texts.append(file.read())
#     return texts

# # Preprocess the texts
# def preprocess_texts(texts):
#     all_sentences = []
#     all_processed_sentences = []
#     stop_words = set(stopwords.words('english'))

#     for text in texts:
#         sentences = sent_tokenize(text)
#         all_sentences.extend(sentences)

#         processed_sentences = []
#         for sentence in sentences:
#             words = word_tokenize(sentence.lower())
#             words = [word for word in words if word.isalnum() and word not in stop_words]
#             processed_sentences.append(' '.join(words))
#         all_processed_sentences.extend(processed_sentences)

#     return all_sentences, all_processed_sentences

# # Find the most relevant sentence
# def find_most_relevant_sentence(question, processed_sentences, vectorizer, sentence_vectors):
#     question_vector = vectorizer.transform([question])
#     similarities = cosine_similarity(question_vector, sentence_vectors)
#     most_similar_idx = similarities.argmax()
#     return most_similar_idx

# # Main question answering function
# def answer_question(question, original_sentences, processed_sentences, vectorizer, sentence_vectors):
#     processed_question = ' '.join([word.lower() for word in word_tokenize(question) if word.isalnum()])
#     most_relevant_idx = find_most_relevant_sentence(processed_question, processed_sentences, vectorizer, sentence_vectors)
#     return original_sentences[most_relevant_idx]

# # Global variables to store processed data
# original_sentences = []
# processed_sentences = []
# vectorizer = None
# sentence_vectors = None

# # Route to serve the HTML file
# @app.route('/')
# def index():
#     return send_from_directory('.', 'medai.html')

# # Route to handle chat messages
# @app.route('/chat', methods=['POST'])
# def chat():
#     global original_sentences, processed_sentences, vectorizer, sentence_vectors

#     data = request.json
#     question = data['message']

#     if question.lower() == 'quit':
#         return jsonify({'answer': 'Goodbye!'})

#     answer = answer_question(question, original_sentences, processed_sentences, vectorizer, sentence_vectors)
#     return jsonify({'answer': answer})

# if __name__ == '__main__':
#     print("Loading and processing the texts. This may take a moment...")
#     directory_path = 'all_text_file'  # Replace with the path to your directory containing text files
#     texts = load_texts(directory_path)
#     original_sentences, processed_sentences = preprocess_texts(texts)

#     vectorizer = TfidfVectorizer()
#     sentence_vectors = vectorizer.fit_transform(processed_sentences)

#     print("Server is ready. Run the application and navigate to http://localhost:5000 in your web browser.")
#     app.run(debug=True)


