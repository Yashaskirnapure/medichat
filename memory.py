from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DATA_PATH='data/'
def load_pdf(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_embedding_model():
    model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return model

documents = load_pdf(data=DATA_PATH)
chunks = create_chunks(documents=documents)
embedding_model = get_embedding_model()


DB_FAISS_PATH = 'vectorstore/db_faiss'
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)