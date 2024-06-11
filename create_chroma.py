import chromadb
import pandas as pd
import os
import chromadb.utils.embedding_functions as embedding_functions
from langchain_openai import OpenAIEmbeddings

raw = pd.read_csv('/workspace/online_banking_qa.csv')
FAQ_answer = raw['A'].to_dict()
persistent_client = chromadb.PersistentClient(path="./chroma")
Q = raw['Q'].tolist()
A = raw['A'].tolist()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002" # text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
            )

# persistent_client.delete_collection(name="rag-chroma") # 刪除collection, 無法復原
collection = persistent_client.get_or_create_collection(name="rag-chroma", embedding_function=openai_ef)

n = len(Q)
Q_len = len(Q)  # 只計算一次 Q 的長度
for i, x in enumerate(['Q']): # , 'gen_Q1', 'gen_Q2', 'gen_Q3', 'gen_Q4'
    test = raw[x].tolist()
    collection.add(
        documents=test,
        metadatas=[{"FAQ":x} for x in range(len(test))],
        ids=[str(n+x) for x in range(len(Q))],
    )
    n += Q_len  # 使用之前存儲的 Q_len
