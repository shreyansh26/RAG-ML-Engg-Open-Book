import os
import sys
import uuid
import chromadb
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PythonLoader
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.storage import InMemoryByteStore
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from flask import Flask, request
from api_key import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_APIKEY

app = Flask(__name__)

HOST = "0.0.0.0"
PORT = 9600

def produce_rag_chain():
    llm = ChatOpenAI(model="gpt-4-1106-preview")
    loader1 = DirectoryLoader('./ml-engineering', glob="**/*.md")
    loader2 = DirectoryLoader('./ml-engineering', glob="**/*.py", loader_cls=PythonLoader)
    docs = loader1.load() + loader2.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    persistent_client = chromadb.PersistentClient(path="./chroma_db_2")
    collection = persistent_client.get_or_create_collection("stas00_articles")

    vectorstore = Chroma(
        client=persistent_client,
        collection_name="stas00_articles",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", disallowed_special=()),
    )

    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    doc_ids = [str(uuid.uuid4()) for _ in docs]

    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

    sub_docs = []
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)

    retriever.vectorstore.add_documents(sub_docs)
    print("Count of sub-docs", len(sub_docs))
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

rag_chain = produce_rag_chain()

@app.route('/ask', methods=["GET"])
def get_historical_ids():
    query = request.args.get("query")
    answer = rag_chain.invoke(query)
    sources = [answer['context'][i].metadata['source'] for i in range(len(answer['context']))]
    return {'answer': answer['answer'], 'sources': sources}

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)