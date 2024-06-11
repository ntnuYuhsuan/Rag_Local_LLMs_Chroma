from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
import os
from langchain_community.vectorstores import Chroma
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb

llm = ChatOllama(model="taide-llama3", format="json", temperature=0)

# llm = ChatOpenAI(model="gpt-4")
# llm = Ollama(model="taide-llama3")

query = "數位帳戶的單日轉帳額度上限是多少？"

# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# raw = pd.read_csv('../chroma/online_banking_qa.csv')
# documents = raw['Q'].to_list()+raw['gen_Q1'].to_list()+raw['gen_Q2'].to_list()+raw['gen_Q3'].to_list()+raw['gen_Q4'].to_list()
# metadatas = [{"FAQ":n%len(raw)} for n in range(len(raw)*5)]
raw = pd.read_csv('../chroma/online_banking_qa.csv')
FAQ_answer = raw['A'].to_dict()

# loader = CSVLoader(file_path='../chroma/online_banking_qa.csv')
# data = loader.load()
# db = Chroma.from_texts(collection_name="online_banking",texts=documents, metadatas=metadatas, embedding=OpenAIEmbeddings(), persist_directory='./')
persistent_client = chromadb.HttpClient(host='127.0.0.1', port=8000)
db = Chroma(
    client=persistent_client,
    collection_name="online_banking",
    embedding_function=OpenAIEmbeddings()
)

# docs = db.similarity_search(query)
# print([doc.metadata['FAQ'] for doc in docs])
# print("👽: "+FAQ_answer[int(docs[0].metadata['FAQ'])])
# context = FAQ_answer[int(docs[0].metadata['FAQ'])]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一個來自台灣的 AI 助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。這裡是參考FAQ的資料:{context}"
        # "You're a helpful AI assistant. Given a user question and some QA pair references, answer the user question. If none of the reference answer the question, just say you don't know.\n\nHere are the QA pair reference:{context}",
        ),
        ("human", "{question}"),
    ]
)

retriever = db.as_retriever(search_type="similarity")

def format_docs(docs):
    return "\n\n".join(doc.page_content+'\n'+FAQ_answer[doc.metadata['FAQ']] for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)
print('\n')