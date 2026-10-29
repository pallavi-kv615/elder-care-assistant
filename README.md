# Elder-Care Assistant: Senior Citizen Rights & Maintenance Act Bot

An AI-powered assistant designed to help users understand senior citizens' rights and maintenance provisions under the Maintenance and Welfare of Parents and Senior Citizens Act, 2007.

## Project Overview

The Elder-Care Assistant uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from a legal document and generate answers to users' questions. It aims to make important information more accessible through a simple, senior-friendly interface.

## Features

- Ask questions about the Senior Citizens Act, 2007.
- Retrieve relevant passages from the source document.
- Generate answers using Google's Gemini AI.
- Display source information to help users verify answers.
- Provide a simple interface designed for accessibility.

## Technology Stack

- **Frontend:** Streamlit
- **AI Model:** Google Gemini
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** FAISS
- **Document Processing:** PyPDF
- **Programming Language:** Python

## Project Structure

```text
elder-care-assistant/
├── documents/
│   └── Senior_Citizen_Act_2007.pdf
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── src/
│   ├── app.py
│   ├── process_documents.py
│   ├── read_pdf.py
│   └── search_documents.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/pallavi-kv615/elder-care-assistant.git
   cd elder-care-assistant
   ```

2. Create and activate a Python virtual environment.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your Gemini API key using an environment variable named `GEMINI_API_KEY`. Never commit API keys or `.env` files to GitHub.

5. Run the application from the repository root:

   ```bash
   streamlit run src/app.py
   ```

## Scope and Limitations

This project is an educational information assistant. It does not replace a lawyer, official legal advice, or confirmation from the relevant government authority. Users should verify important legal information against official sources.

## Future Improvements

- Add verified NPS guidelines and related resources.
- Improve voice-based accessibility.
- Enhance retrieval accuracy and source citations.
- Deploy the application on Streamlit Community Cloud.

## Author

**Pallavi K. V.**

BCA Student
