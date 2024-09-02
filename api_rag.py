import os

from fastapi import FastAPI, Query
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

app = FastAPI()


@app.get("/get-query/")
def get_query(n_responses: int = Query(lt=3), query: str = Query(description="Introduceti intrebarea:")):
    embedding = OpenAIEmbeddings()

    vectordb = Chroma(persist_directory='db', embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": n_responses})
    llm_response = retriever.invoke(query)

    # Set up the turbo LLM
    turbo_llm = ChatOpenAI(
        temperature=1,
        model_name='gpt-3.5-turbo'
    )
    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                      chain_type="stuff",
                                      retriever=retriever,
                                      return_source_documents=True,
    )

    return {"Answer": qa_chain(query)}
