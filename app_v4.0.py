import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Import or paste your core initialization, faqs_kb, faq_embeddings, 
# and interact_with_bot functions from your script here.

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Replace with your exact IONOS domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    # Calls the interaction function from your codebase
    reply = interact_with_bot(request.message)
    return {"reply": reply}
