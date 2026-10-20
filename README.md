\# 📄 IntelliAssist AI: Smart Document Assistant



IntelliAssist AI is a Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, FAISS, and Google Gemini. It allows users to upload documents (PDF, DOCX, TXT) and ask natural-language questions to extract accurate, context-grounded insights with transparent source citations.



\---



\## 🚀 Key Features



\- \*\*Multi-Format Ingestion:\*\* Extracts text from `.pdf`, `.docx`, and `.txt` files while tracking document source metadata.

\- \*\*Chunk Inspector:\*\* Displays split chunk metrics (chunk ID, source document, character count) via a Pandas DataFrame.

\- \*\*Dense Vector Embeddings:\*\* Encodes document chunks locally using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`.

\- \*\*Vector Search:\*\* Fast semantic indexing and retrieval powered by FAISS.

\- \*\*Scikit-Learn Re-ranking:\*\* Computes explicit cosine similarity between query and retrieved document embeddings, filtering out irrelevant chunks.

\- \*\*Grounded LLM Generation:\*\* Answers user queries using Google's `gemini-2.5-flash` model strictly from retrieved context to avoid hallucinations.

\- \*\*Source Citations:\*\* Visual breakdown showing relevance scores and exact source chunks used for every answer.



\---



\## 🛠️ Tech Stack



\- \*\*Frontend / UI:\*\* \[Streamlit](https://streamlit.io/)

\- \*\*Document Processing:\*\* PyPDF2, python-docx

\- \*\*Text Splitting \& Orchestration:\*\* LangChain (`langchain-text-splitters`, `langchain-community`)

\- \*\*Embeddings:\*\* Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) via `langchain-huggingface`

\- \*\*Vector Database:\*\* FAISS (Facebook AI Similarity Search)

\- \*\*Data Handling \& Analytics:\*\* Pandas, NumPy, Scikit-learn

\- \*\*Large Language Model:\*\* Google Gemini (`gemini-2.5-flash`) via `google-genai` SDK



\---



\## 📋 System Architecture



1\. \*\*Document Loading \& Metadata Tagging:\*\* Documents are parsed into discrete `Document` objects tagged with their filename.

2\. \*\*Chunking:\*\* Text is split using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200).

3\. \*\*Vector Store Creation:\*\* Embeddings are generated via `all-MiniLM-L6-v2` and indexed into FAISS locally (`faiss\_index/`).

4\. \*\*Candidate Retrieval:\*\* Top $k$ candidate chunks are retrieved from FAISS upon user query.

5\. \*\*Similarity Scoring \& Filtering:\*\* Cosine similarity is calculated between query and document vectors using Scikit-Learn.

6\. \*\*Synthesis:\*\* Confident chunks are formatted into a prompt and submitted to Google Gemini for final response synthesis.



\---



\## ⚙️ Installation \& Local Setup



\### 1. Clone the repository

```bash

git clone \[https://github.com/Dharshan369-gif/IntelliAssist-AI.git](https://github.com/Dharshan369-gif/IntelliAssist-AI.git)

cd IntelliAssist-AI

