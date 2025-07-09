import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
DATA_DIR = Path("data")  

docs = []

for path in DATA_DIR.iterdir():
    if path.suffix.lower() in [".txt", ".md"]:
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text,
                             metadata={"source": str(path)}))

    elif path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        for row in df.itertuples():
            content = f"{row.description}\nResolution: {row.resolution}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                    "source": str(path),
                    "id": row.id,
                    "title": row.title,
                    "type": row.type_level,     # 1, 2 o 3
                    },
                )
            )


    elif path.suffix.lower() == ".pdf":
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(str(path))
        docs.extend(loader.load())

    elif path.suffix.lower() in [".docx", ".pptx"]:
        from langchain_community.document_loaders import UnstructuredWordLoader, UnstructuredPPTLoader
        loader = UnstructuredWordLoader(str(path)) if path.suffix.lower()==".docx" else UnstructuredPPTLoader(str(path))
        docs.extend(loader.load())

    elif path.suffix.lower()==".urls":
        from langchain_community.document_loaders import WebBaseLoader
        for url in path.read_text().splitlines():
            docs.extend(WebBaseLoader(url).load())

print(f"correct {len(docs)}")

# 2) Chunking
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200,
    chunk_overlap=40
)

chunks = splitter.split_documents(docs)
print(f"generate {len(chunks)} chunks text")

# 3) Embeddings + VectorStore
emb = OpenAIEmbeddings()
vs = Chroma.from_documents(
    chunks,
    embedding=emb,
    persist_directory="vector_store"
)

print("created vector store in ./vector_store")