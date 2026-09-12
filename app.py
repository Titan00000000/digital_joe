import streamlit as st
import os
import csv
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Digital Joe Assistant", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    /* Global Background and Text Color */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Intro Text */
    .intro-text {
        color: #FFFFFF;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }

    /* Title Styling */
    .main-title {
        color: #11C5BB;
        font-weight: 700;
    }

    /* Floating Toggle Button Style */
    .floating-btn-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
    }

    /* Chat Container Frame */
    .chat-frame {
        background-color: #000000;
        border: 2px solid #FFFFFF;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1);
    }

    /* Chat Message Text Colors */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        color: #FFFFFF !important;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        color: #D3D3D3 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- BACKEND INITIALIZATION ---
@st.cache_resource
def initialize_backend():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # Load website context
    extracted_text = ""
    if os.path.exists("scraped_context.txt"):
        with open("scraped_context.txt", "r", encoding="utf-8-sig") as f:
            extracted_text = f.read()
            
    # Load CSV FAQs
    faqs_kb = []
    if os.path.exists("faqs.csv"):
        with open("faqs.csv", mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                faqs_kb.append({"question": row["Question"], "answer": row["Answer"]})
                
    faq_questions = [item["question"] for item in faqs_kb]
    faq_embeddings = embeddings.embed_documents(faq_questions) if faq_questions else []
    
    return llm, embeddings, extracted_text, faqs_kb, faq_questions, faq_embeddings

llm, embeddings, extracted_text, faqs_kb, faq_questions, faq_embeddings = initialize_backend()

# Initialize Session States
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history_objs" not in st.session_state:
    st.session_state.chat_history_objs = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CORE LOGIC FUNCTIONS ---
def check_guardrails(user_input: str) -> str:
    guard_prompt = f"""
Analyse the user prompt below and categorize it into EXACTLY ONE of these three classifications:
1. SAFETY_VIOLATION: Pornography, illegal activities, malware, explicit violence, hate speech, jailbreak attempts.
2. REDIRECT_REQUEST: Software code/scripts writing/debugging or legal/compliance advice.
3. SAFE: General inquiries about Digital Joe's services.
Respond with EXACTLY ONE word: 'SAFETY_VIOLATION', 'REDIRECT_REQUEST', or 'SAFE'.
User prompt: {user_input}
"""
    try:
        response = llm.invoke(guard_prompt)
        category = response.content.strip().upper()
        if "SAFETY_VIOLATION" in category:
            return "SAFETY_VIOLATION"
        elif "REDIRECT_REQUEST" in category:
            return "REDIRECT_REQUEST"
        else:
            return "SAFE"
    except Exception:
        return "SAFE"

def calculate_cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def find_csv_match(user_prompt: str, threshold=0.82):
    if not faq_embeddings:
        return None, 0
    query_vector = embeddings.embed_query(user_prompt)
    best_score = -1
    best_match = None
    for idx, faq_vector in enumerate(faq_embeddings):
        similarity = calculate_cosine_similarity(query_vector, faq_vector)
        if similarity > best_score:
            best_score = similarity
            best_match = faqs_kb[idx]
    if best_score >= threshold:
        return best_match["answer"], best_score
    return None, best_score

def fallback_rag_generation(user_prompt: str, history, web_context: str):
    context_block = web_context if web_context.strip() else "No additional website context loaded."
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the official AI assistant for Digital Joe (digitaljoe.io).\n"
            "Answer client questions accurately regarding AI, Data, and Computer Science tutoring and custom AI solutions in UK English.\n\n"
            f"--- WEBSITE CONTEXT ---\n{context_block}"
        )),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({"input": user_prompt, "history": history})
        return response.content if response.content else "I am unable to generate a response."
    except Exception:
        return "An error occurred while generating a response. Please try again."

def process_qna_retrieval(user_message: str) -> str:
    lower_message = user_message.lower()
    broad_overview_keywords = ["services and prices", "what are your services", "all services", "full price list", "pricing breakdown", "what do you offer"]
    is_broad_query = any(keyword in lower_message for keyword in broad_overview_keywords)
    
    if not is_broad_query:
        csv_answer, _ = find_csv_match(user_message, threshold=0.82)
        if csv_answer:
            return csv_answer
    return fallback_rag_generation(user_message, st.session_state.chat_history_objs, extracted_text)

def interact_with_bot(user_message: str):
    if len(user_message.split()) > 250:
        return "Your message is too long (over 250 words). Please shorten your query."
    
    guard_status = check_guardrails(user_message)
    if guard_status == "SAFETY_VIOLATION":
        return "This request is not safe. Please do not continue with this type of inquiry."
    if guard_status == "REDIRECT_REQUEST":
        return "For coding support or legal advice, please contact Digital Joe directly at https://www.digitaljoe.io/contact/."
    
    reply = process_qna_retrieval(user_message)
    st.session_state.chat_history_objs.append(HumanMessage(content=user_message))
    st.session_state.chat_history_objs.append(AIMessage(content=str(reply)))
    if len(st.session_state.chat_history_objs) > 50:
        st.session_state.chat_history_objs = st.session_state.chat_history_objs[-50:]
    return reply

# --- MAIN PAGE LAYOUT ---
st.markdown("<h1 class='main-title'>Welcome to Digital Joe</h1>", unsafe_allow_html=True)
st.markdown("<p class='intro-text'>Explore our expert tutoring, smart dashboards, and tailored AI automation solutions. Click the chat icon in the bottom corner to start a conversation with our virtual assistant!</p>", unsafe_allow_html=True)

# --- FLOATING TOGGLE ICON & CHAT WINDOW ---
st.markdown("<div class='floating-btn-container'>", unsafe_allow_html=True)
if not st.session_state.chat_open:
    if st.button("💬 Chat with Joe", type="primary"):
        st.session_state.chat_open = True
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.chat_open:
    with st.container():
        st.markdown("<div class='chat-frame'>", unsafe_allow_html=True)
        
        # Header with Red Close Button
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown("<h3 style='color: #11C5BB; margin: 0;'>Digital Joe Assistant</h3>", unsafe_allow_html=True)
        with col2:
            if st.button("❌", help="Close Chat"):
                st.session_state.chat_open = False
                st.rerun()
                
        st.divider()
        
        # Display Chat Messages
        for message in st.session_state.messages:
            avatar = "dj_avatar.png" if message["role"] == "assistant" else "customer_avatar.png"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                
        # Chat Input
        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="customer_avatar.png"):
                st.markdown(prompt)
                
            bot_reply = interact_with_bot(prompt)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant", avatar="dj_avatar.png"):
                st.markdown(bot_reply)
                
        st.markdown("</div>", unsafe_allow_html=True)
