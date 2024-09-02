import os
from langchain_community.vectorstores import (Chroma)
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

loader = DirectoryLoader('documents', glob="*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

text_split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
texts = text_split.split_documents(documents)
texts_len = len(texts)

dir = 'db'

embedding = OpenAIEmbeddings()
step = 100
for k in range(0, texts_len, step):
    Chroma.from_documents(documents=texts[k:k+step],
                          embedding=embedding,
                          persist_directory=dir)

#vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=dir)
