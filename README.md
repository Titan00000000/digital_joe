# Digital Joe Assistant RAG-Based Chatbot

An intelligent, Retrieval-Augmented Generation (RAG) chatbot designed to act as a knowledgeable assistant for Digital Joe (digitaljoe.io). This system utilises a hybrid retrieval architecture built using Python, NumPy, LangChain, and the Google Gemini API, deployed via a Streamlit interface.

## Project Overview
This project automates the end-to-end process of:
* **Web Scraping:** Extracting and cleaning text data from 7 public pages of the website (https://www.digitaljoe.io) using Playwright and BeautifulSoup.
* **Synthetic Q/A Generation:** Leveraging gemini-2.5-flash with structured outputs to generate a high-quality dataset of 200 domain-specific question-and-answer pairs covering tutoring, smart dashboards, and tailored AI tools.
* **Single-Index Vector Store:** Vectorises the FAQ dataset using gemini-embedding-2-preview embeddings and performs semantic searches using NumPy-based cosine similarity in-memory.
* **Hybrid Routing Logic:** Employs a custom content moderation classifier to filter safety violations or coding requests, coupled with cosine-similarity matching (threshold $\ge 0.82$) and a Gemini RAG fallback system.

---

## Tech Stack & Frameworks
* **Language:** Python 
* **Frontend UI:** Streamlit
* **Vector Database:** NumPy for vector similarity calculations and in-memory storage
* **Orchestration & Frameworks:** LangChain
* **Embedding & LLM Generation:** Google Gemini API (gemini-2.5-flash & gemini-embedding-2-preview)
* **Web Scraping:** Playwright, Beautiful Soup 4, Requests

---

## Author
* **Mr Timur Rahman**
