import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]


def process_response(response):
    for doc in response:
        print("Documents:")
        print(doc.page_content)
        print('\nMetadata:')
        for key, val in doc.metadata.items():
            print(f"{key}: {val}")

        print("\n-----------------------\n")


embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory='db',
                  embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
#docs = retriever.invoke("Ce informații specifice trebuie incluse în anunțurile de achiziții publice?")
#print(docs)

#qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

#query = "Ce informații specifice trebuie incluse în anunțurile de achiziții publice?"
llm_response = retriever.invoke("Ce informații specifice trebuie incluse în anunțurile de achiziții publice?")
process_response(llm_response)
