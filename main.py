from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.llms import OpenAI
import os
import pinecone  # to store vector data we use pinecone
from dotenv import load_dotenv

load_dotenv()  # this will give variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)

if __name__ == "__main__":
    print("Hello World!")

    read = "pdf"

    if read == "text":
        # ---------reading txt with pinecone------------
        loader = TextLoader(os.getenv("FILE_PATH"))
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents=document)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        docsearch = Pinecone.from_documents(
            texts, embedding=embeddings, index_name=os.getenv("PINECONE_INDEX_NAME")
        )

        qa = VectorDBQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            vectorstore=docsearch,
            return_source_documents=True,
        )

        query = "What is a vector DB? Give me a 15 word answer for a beginner"

        result = qa({"query": query})
        print(result)
        # ---------reading txt with pinecone------------

    elif read == "pdf":
        # ---------reading txt with FAISS------------

        pdf_path = os.getenv("PDF_FILE_PATH")

        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30, separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index_book")

        new_vectorstore = FAISS.load_local("faiss_index_book", embeddings)
        
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
        )
        res = qa.run("what is three mistakes he made?")
        print(res)

        # ---------reading txt with FAISS------------
        pass
