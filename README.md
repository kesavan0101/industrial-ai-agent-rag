# Autonomous Industrial Maintenance Agent & RAG Pipeline

An end-to-end Agentic AI system that continuously monitors multivariate industrial time-series data, detects equipment degradation using Machine Learning, and autonomously queries technical equipment manuals via Retrieval-Augmented Generation (RAG) to generate structured Maintenance Action Tickets.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B-purple)
![Vector DB](https://img.shields.io/badge/Vector_DB-ChromaDB-blueviolet)

---

## System Architecture

```text
                               ┌─► Tool 1: ML Anomaly Detector (Isolation Forest on 160k+ NASA Rows)
[User / System Input] ──► [LangGraph ReAct Agent]
 (e.g. Unit #260)              └─► Tool 2: Semantic RAG System (ChromaDB + Technical PDF Manuals)
                                       │
                                       ▼
                       [Structured Maintenance Action Ticket]
Key Features
1.Multi-Dataset Time-Series Analytics: Processes over 160,000 time-series sensor records from the NASA CMAPSS dataset across 260 distinct turbofan engines.
2.Unsupervised Anomaly Detection: Applies IsolationForest across multivariate sensor streams ($S_2, S_3, S_4, S_{11}$) to identify equipment degradation.
3.Semantic Document Retrieval: Vectorizes 1,300+ pages of technical PDF manuals into a local ChromaDB store using HuggingFace sentence embeddings (all-MiniLM-L6-v2).
4.Autonomous Tool Reasoning: Employs LangGraph (ReAct pattern) powered by Llama-3.3-70B (via Groq API) to dynamically select when to calculate sensor statistics versus querying technical documentation.
5.Interactive Diagnostic CLI: Enables engineers to enter any Engine Unit ID (1 to 260) for real-time diagnostic reporting.
