# Project-Exhibition-I

# 📄 Smart PDF / Document Knowledge Extractor

> An NLP-powered application that analyzes multiple PDF documents to extract important keywords, measure document similarity, and visualize relationships between documents.

---

## 🚀 Overview

**Smart PDF / Document Knowledge Extractor** is a Python-based NLP application designed to analyze multiple PDF documents simultaneously.

Instead of manually reading several documents to find common or related information, the system processes the uploaded PDFs and provides useful insights such as:

- 🔑 Important keywords from each document
- 📊 Document similarity scores
- 🔗 Relationships between documents
- 📈 Similarity matrix
- 🕸️ Visual document relationship graph

The application provides an interactive **Streamlit web interface** where users can upload multiple PDF files and analyze them.

---

## ✨ Current Features

- 📤 Upload multiple PDF documents
- 📖 Extract text from PDF files
- 🧹 Clean and preprocess extracted text
- 🔑 Extract important keywords using **TF-IDF**
- 🔢 Convert documents into numerical vectors
- 📐 Calculate **Cosine Similarity** between documents
- 📊 Generate a document similarity matrix
- 🕸️ Visualize document relationships using **NetworkX**
- 🖥️ Interactive web interface using **Streamlit**
- 🥇 Identify the most similar pair of documents

---

## 🧠 How It Works

The system follows the pipeline below:

```text
        📄 PDF Documents
              │
              ▼
       📤 PDF Upload
              │
              ▼
      📖 Text Extraction
              │
              ▼
       🧹 Text Cleaning
              │
              ▼
          🔑 TF-IDF
              │
              ▼
     📝 Keyword Extraction
              │
              ▼
     🔢 Document Vectorization
              │
              ▼
      📐 Cosine Similarity
              │
              ▼
       📊 Similarity Matrix
              │
              ▼
       🕸️ Relationship Graph
