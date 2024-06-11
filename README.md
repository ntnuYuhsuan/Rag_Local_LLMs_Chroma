# Rag_Local_LLM_Chroma
RAG Chatbot with Taide-8B-llama3 / Breeze-7b-32K

### 預設環境
```
Anaconda
python 3.8  # langchain (chatbot) , Ollama
python 3.11  # ChromaDB
```
以我的環境重建(不建議)
```
conda create -f envs/chatbot.yml # chroma.yml, ollama.yml
```

---
### 架構

- **Chroma server (Vector Database)**
- **Ollama server (LLM framework)**
- **Langchain Serve Server (core LLM api)**

---
## Create Chroma Vector Database
[Chroma info](https://docs.trychroma.com/)
```
pip install chromadb langchain_openai pandas
```

## 路徑更新&建立向量資料庫
### 隨機建立網路銀行常見問題,國泰/富邦網銀 (客製化FAQ內容)

```
pd.read_csv('/workspace/online_banking_qa.csv') # changing route
-
python create_chroma.py
```

遇到sqlite3報錯
[解法](https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300)
```
pip install pysqlite3-binary
```
在檔案前面添加
```
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

## 啟動Chroma
```
cd chroma
chroma run --path ./ --host 127.0.0.1
```

---
## Install Ollama
```
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

## Pull LLM
可以直接載入已上傳的量化模型，包括tokenizer、預設prompt以及相關template

(https://ollama.com/jcai/llama3-taide-lx-8b-chat-alpha1:q6_k)

(https://ollama.com/jcai/breeze-7b-32k-instruct-v1_0:q4_k_m)
```
ollama pull jcai/breeze-7b-32k-instruct-v1_0:q4_k_m
ollama pull jcai/llama3-taide-lx-8b-chat-alpha1:q6_k
```

也可以手動將 .gguf (AKA量化模型下載)，透過Modelfile部署
### TAIDE

(https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit/tree/main)
### MS phi-3

(https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
定義Modelfile內的 system prompt template
```
cd Ollama
ollama create taide-llama3 -f taide-llama3.Modelfile
```

### Ollama list

現有模型可供調用
```
ollama list

NAME                                            ID              SIZE    MODIFIED          
jcai/breeze-7b-32k-instruct-v1_0:q4_0           8b84df66d3e5    4.3 GB  About an hour ago
jcai/breeze-7b-32k-instruct-v1_0:q4_k_m         96db840ae4ff    4.5 GB  2 hours ago      
jcai/llama3-taide-lx-8b-chat-alpha1:q6_k        befe3f5e406e    6.6 GB  3 hours ago      
jcai/llama3-taide-lx-8b-chat-alpha1:f16         76ba6fda2ac0    16 GB   3 hours ago      
taide-llama3:latest                             4e3baf18e69f    4.9 GB  3 hours ago
```

---
## Deploy Langchain Serve
[Langchain Serve Sample](https://github.com/langchain-ai/langchain/tree/master/templates/rag-chroma-private)
[langchain document](https://api.python.langchain.com/en/latest/langchain_api_reference.html)

### Installation & load framework
```
pip install -U langchain-cli langchain_community 
langchain app new rag_chatbot --package rag-chroma-private
```

### 增加API節點，修改/rag_chatbot/app/server.py
```
from rag_chroma_private import chain as rag_chroma_private_chain

add_routes(app, rag_chroma_private_chain, path="/rag-chroma-private")
```

### 編輯RAG對話鏈，以我的chain.py複寫
修改/rag_chatbot/packages/rag-chroma-private/rag_chroma_private/chain.py

38行，抽換模型跟Embedding，若要使用gpt要先將環境變數 $OPENAI_API_KEY 設定完
```
llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()
# llm = Ollama(model='jcai/breeze-7b-32k-instruct-v1_0:q4_k_m') # jcai/llama3-taide-lx-8b-chat-alpha1:q6_k
# llm = Ollama(model='taide-llama3') # 透過Modelfile自己部署的地端模型
# embeddings = OllamaEmbeddings()
```

44行，資料路徑
```
raw = pd.read_csv('/home/newuser/chroma/online_banking_qa.csv')
FAQ_answer = raw['A'].to_dict()
```

### 執行Langchain Serve

Launch LangServe，到專案目錄，可以自己定義要部署在哪個port
```
cd /rag_chatbot
langchain serve --port=8100
```

### Access API documentation at http://127.0.0.1:8100/docs
![image](https://github.com/ntnuYuhsuan/Rag_Taide_Chroma/assets/167750277/8dc13b01-762b-4df2-9072-808c2d4cac92)

### Access UI demo at http://127.0.0.1:8100/rag-chroma-private/playground/
![image](https://github.com/ntnuYuhsuan/Rag_Taide_Chroma/assets/167750277/e2047326-8adc-49b8-98a7-48f739c8077e)

RAG routing
![image](https://github.com/ntnuYuhsuan/Rag_Taide_Chroma/assets/167750277/02d73802-5bc4-48b5-ac9b-890db054cdd7)

Output
![image](https://github.com/ntnuYuhsuan/Rag_Taide_Chroma/assets/167750277/6df6eeca-e5e6-4702-a668-a4312c24bd06)
