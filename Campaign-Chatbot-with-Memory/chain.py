from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as import_error:  # pragma: no cover
    raise import_error


SYSTEM_CONTEXT = (
    "You are a marketing strategist AI. You create concise, creative, and on-brand "
    "campaign concepts, channels, and content ideas. Prioritize clarity, distinctiveness, "
    "and actionable next steps. Keep answers skimmable with short paragraphs and bullets."
)


def _get_api_key() -> Optional[str]:
    """Return Gemini API key from env using common names."""
    load_dotenv()
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def _get_default_model(model_name: Optional[str]) -> str:
    # Prefer explicit arg, else env override, else fast default
    return (
        model_name
        or os.getenv("GEMINI_MODEL")
        or "gemini-1.5-flash"
    )


def build_conversation_chain(model_name: Optional[str] = None) -> RunnableWithMessageHistory:
    """Create a LangChain chain with Gemini and message history."""
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment."
        )

    model_to_use = _get_default_model(model_name)

    llm = ChatGoogleGenerativeAI(
        model=model_to_use,
        api_key=SecretStr(api_key),
        temperature=0.6,
    )

    # Create the prompt template with system message and message history
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_CONTEXT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Create the chain
    chain = prompt | llm

    # Create a simple in-memory store for chat histories
    store: Dict[str, BaseChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Wrap the chain with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history
