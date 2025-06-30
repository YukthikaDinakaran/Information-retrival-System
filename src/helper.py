import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import GooglePalm
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# ✅ Set Google API key from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")

# ✅ Initialize LLM
llm = GooglePalm(google_api_key=GOOGLE_API_KEY)
def get_llm():
    from langchain_community.llms import GooglePalm
    return GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"])


# ✅ Read PDF content
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# ✅ Split into text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

# ✅ Vectorize text
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# ✅ Conversational chain
def get_conversational_chain(vector_store):
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain