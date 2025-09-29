# Summarization Engine

# 📄 PDF Summarization and RAG Q&A with LangChain and Gemini

This project provides a robust solution for loading data from a PDF, segmenting the content, and using the Gemini model to answer questions or generate a summary based *only* on the information contained in the document (Retrieval Augmented Generation - RAG).

---

## ✨ Features

* **PDF Document Loading:** Efficiently loads content from a local PDF file using `PyPDFLoader`.
* **Intelligent Text Splitting:** Uses a Recursive Character Text Splitter to preserve document structure and create optimized chunks for the model.
* **Vectorization & Embedding:** Converts text chunks into numerical vectors using Google's `GoogleGenerativeAIEmbeddings` for semantic search.
* **Vector Store Creation:** Stores and indexes the text vectors using a vector database (e.g., ChromaDB or in-memory) for fast retrieval.
* **RAG Chain:** Implements a complete RAG workflow using LangChain to retrieve relevant document chunks before generating a final answer with the **Gemini 2.5 Flash** model.

---

## ⚙️ How to Run

### 1. Prerequisites

Before running the script, ensure you have:

* **Python 3.10+** installed.
* A **PDF file** that you want to analyze (e.g., `report.pdf`).
* A **Gemini API Key**.

### 2. Setup

#### A. Install Dependencies

Install the necessary Python libraries.

```bash
pip install -U \
  langchain-google-genai \
  langchain-community \
  langchain-core \
  pypdf \
  chromadb
``` 
### 3.Update the script

pdf_path = r"C:\Users\bhatt\Downloads\report.pdf" # <-- Use your verified, correct path
loader = PyPDFLoader(pdf_path)

### 4. Execute

python summary.py

graph TD
    A[Start: summary.py Execution] --> B(1. Load PDF File);

    B --> C{2. Text Splitting};
    C --> D(3. Generate Embeddings);
    D --> E(4. Create/Load Vector Store);

    E --> F[5. User Question];
    F --> G(6. Embed User Question);
    G --> H(7. Vector Store Retrieval (Similarity Search));

    H --> I(8. Context Chunks);
    I --> J(9. Construct Prompt: "Answer based on Context Chunks");
    J --> K(10. Gemini 2.5 Flash (LLM) Invocation);
    K --> L[11. Final Summary/Answer];
    L --> M[End];
