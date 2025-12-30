# 🤖 AI Portfolio Chatbot

A recruiter-facing Generative AI chatbot integrated into my portfolio website to answer questions about my experience, projects, and skills conversationally.

The project demonstrates **end-to-end GenAI system design**, including retrieval-based experimentation, deployment constraints, and production-ready prompt engineering.

---

## ✨ What This Chatbot Does

* Answers recruiter-style questions about my:

  * Work experience
  * Technical skills
  * Projects
  * Certifications
* Uses **strict grounding rules** to avoid hallucinations
* Responds in **third person** (recruiter-safe, resume-aligned)
* Runs as a **live API** integrated into my portfolio website

👉 **Live Demo**: Available via my portfolio website

---

## 🏗️ Architecture Overview

This repository intentionally contains **two backends**:

### 1️⃣ `backend-deploy` (Production Version)

A lightweight, deployable GenAI service optimized for free-tier cloud environments.

* Uses prompt-engineered context injection
* Avoids vector databases to reduce memory usage
* Designed for low-latency and stability
* Deployed as a FastAPI service and connected to the portfolio UI

This is the **version running live**.

---

### 2️⃣ `backend-rag` (RAG Experimentation Version)

A full Retrieval-Augmented Generation (RAG) implementation preserved for learning and interview discussion.

* FAISS-based similarity search
* Sentence-transformer embeddings
* Chunked resume and project documents
* Reproducible via `ingest.py`

This version demonstrates **retrieval fundamentals**, even though it is not deployed due to cloud memory limits.

---

## 🧰 Tech Stack

### Backend

* FastAPI
* Python
* Groq API (LLaMA 3.1)
* FAISS (RAG prototype)
* Sentence Transformers

### Frontend

* React
* TypeScript
* Tailwind CSS

### Deployment

* Render (Free Tier)
* API-based integration into portfolio website

---

## ⚙️ Key Engineering Decisions

* Started with a **full RAG pipeline** to ensure grounded answers
* Identified **free-tier cloud memory constraints** during deployment
* Refactored into a **prompt-engineered production architecture**
* Preserved RAG implementation separately for transparency and interviews
* Enforced response rules to prevent:

  * Hallucinated experience
  * Incorrect timelines
  * First-person answers


