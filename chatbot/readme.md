#### 可以參考chat.py是最直觀且容易理解langchain概念的sample code

若遇到版本問題可以直接試試看
```
python app/server.py
```

注意Chroma資料庫多筆複寫的問題

```
/rag_chroma_private/chain.py Guide
```

66行prompt可以調整，也是影響RAG回應品質的關鍵

84行format_docs()可能可以再想想怎麼串接RAG的內容，或是篩選較高關聯的上下文
