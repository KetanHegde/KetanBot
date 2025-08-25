import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import requests
import tempfile
from fastapi.middleware.cors import CORSMiddleware

# Load API keys from .env
load_dotenv()

# Public file ID
file_id = os.getenv("DRIVE_FILE_ID")
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Download the PDF into memory
response = requests.get(url)
response.raise_for_status()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(response.content)
    tmp_path = tmp_file.name

# Initialize FastAPI
app = FastAPI()

origins = [
    "http://127.0.0.1:5500",  
    "http://localhost:5500",
    "https://ketanhegde.github.io/" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],           
)

# Define request schema
class ChatRequest(BaseModel):
    query: str

# Load the document once (one-page PDF)
loader = PyPDFLoader(tmp_path)
documents = loader.load()

# Convert to plain text (assume single page)
document_text = "\n".join([doc.page_content for doc in documents])

# Setup LLM
llm = init_chat_model(os.getenv("MODEL_NAME"), model_provider=os.getenv("MODEL_PROVIDER"), temperature = 0.1)

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are Ketan, graduated recently, and this chatbot is part of your personal portfolio website. "
     "The following information describes you, your work, and your experiences. "
     "When answering, always speak as yourself (first-person), as if you are talking directly to a recruiter or a technical professional. "
     "Base your answers strictly on the information provided below. "
     "Do not mention that you are using a document or dataset—treat the information as your own memory. "
     "If a user asks something that is not covered, respond naturally in first person with something like "
     "'I’m not sure about that,' or 'I don’t have an answer for this right now, could you rephrase?' "
     "Keep your answers clear, concise, polite, and slightly elaborative when needed. "
     "Information about you:\n\n{document}\n"),
    ("user", "{query}")
])


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Format prompt
        messages = prompt_template.format_messages(
            document=document_text,
            query=request.query
        )

        # Get response from LLM
        response = llm.invoke(messages)

        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))