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

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()
# llm = Ollama(model='taide-llama3')
# embeddings = OllamaEmbeddings()

raw = pd.read_csv('../chroma/online_banking_qa.csv')
FAQ_answer = raw['A'].to_dict()

persistent_client = chromadb.HttpClient(host='127.0.0.1', port=8000)
db = Chroma(
    client=persistent_client,
    collection_name="online_banking",
    embedding_function=OpenAIEmbeddings()
)
retriever = db.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content+'\n'+FAQ_answer[doc.metadata['FAQ']] for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    # MessagesPlaceholder(variable_name="chat_history"),
    ("system","你是一個來自台灣的 AI 助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。這裡是參考FAQ的資料:{context}"),
    ("user", "{input}"),
])
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

input_text = input('🌱 >>>')
while input_text.lower() != 'bye':
    if input_text:
        for chunk in rag_chain.stream(input_text):
            print(chunk, end="", flush=True)
        print(retriever.invoke(input_text),'\n')
        # chat_history.append(HumanMessage(content=input_text))
        # chat_history.append(AIMessage(content=response['answer']))
    input_text = input('🌱 >>>')