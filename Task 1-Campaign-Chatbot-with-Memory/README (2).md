# Campaign Chatbot with Memory (Gemini + LangChain + Streamlit)

A minimal chatbot that answers client campaign queries and retains context across turns using LangChain's `ConversationBufferMemory` and Google's Gemini (via Google AI Studio).

## Features
- Gemini Pro via LangChain
- Conversation memory across turns
- Streamlit UI for quick testing
- Sample synthetic prompts for financial-planning style content

## Prerequisites
- Python 3.9+
- Gemini API Key from Google AI Studio — get one at https://aistudio.google.com/apikey

## Setup

1) Create a virtual environment and activate it:
```bash
python -m venv .venv
. .venv\\Scripts\\Activate.ps1
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Configure API key:
- Copy `.env.example` to `.env` and set your key.
- The app reads `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

```
GEMINI_API_KEY=your_key_here
```

4) Run the app:
```bash
streamlit run app/main.py
```

## Example Flow
- "Suggest a campaign idea for a luxury watch."
- "Now adapt it for Instagram Reels."

## References
- LangChain Docs: https://python.langchain.com/docs/introduction/
- Gemini API Keys: https://aistudio.google.com/apikey
