
# # Updated for langchain 1.2.10
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings



## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()


  # Correct import
import os

# Step 1: Load raw PDF(s)
DATA_PATH = r"E:\Medical-fast-api-chatbot\Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf"

def load_pdf_files(data):
    loader = PyPDFLoader(data)  # single PDF ke liye DirectoryLoader nahi
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print("Number of pages loaded: ", len(documents))

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print("Number of text chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings
# Set your Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # "cuda" if GPU available
)

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)
print("FAISS vector store saved at:", DB_FAISS_PATH)

