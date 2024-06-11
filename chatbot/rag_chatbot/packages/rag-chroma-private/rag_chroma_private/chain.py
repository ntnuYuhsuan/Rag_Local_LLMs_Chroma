# Load
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# Split

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# Add to vectorDB
# vectorstore = Chroma.from_documents(
#     documents=all_splits,
#     collection_name="rag-private",
#     embedding=GPT4AllEmbeddings(),
# )
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb
import pandas as pd
import os

llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()
# llm = Ollama(model='taide-llama3')
# embeddings = OllamaEmbeddings()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

raw = pd.read_csv('/home/newuser/chroma/online_banking_qa.csv')
FAQ_answer = raw['A'].to_dict()

persistent_client = chromadb.HttpClient(host='127.0.0.1', port=8000)
vectorstore = Chroma(
    client=persistent_client,
    collection_name="rag-chroma",
    embedding_function=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# Prompt
# Optionally, pull from the Hub
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")
# Or, define your own:
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
prompt = ChatPromptTemplate.from_messages([
    # MessagesPlaceholder(variable_name="chat_history"),
    ("system","你是一個來自台灣的 AI 助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。這裡是參考FAQ的資料:{context}"),
    ("user", "{input}"),
])

# LLM
# Select the LLM that you downloaded
# ollama_llm = "llama2:7b-chat"
# model = ChatOllama(model=ollama_llm)

# RAG chain
# chain = (
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#     | prompt
#     | model
#     | StrOutputParser()
# )
def format_docs(docs):
    return "\n\n".join(doc.page_content+'\n'+FAQ_answer[doc.metadata['FAQ']] for doc in docs)
rag_chain = (
    RunnableParallel({"context": retriever | format_docs, "input": RunnablePassthrough()})
    # {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str


chain = rag_chain.with_types(input_type=Question)
