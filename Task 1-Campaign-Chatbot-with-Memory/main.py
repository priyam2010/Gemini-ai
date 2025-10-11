import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Support running via `streamlit run app/main.py` (cwd becomes app/) and as a package
try:  # local import when cwd is app/
    from chain import build_conversation_chain
except Exception:  # fallback when running with project root on sys.path
    from app.chain import build_conversation_chain


def get_chain(model_name: str | None = None):
    if "chain" not in st.session_state or st.session_state.get("model_name") != model_name:
        st.session_state.model_name = model_name
        st.session_state.chain = build_conversation_chain(model_name=model_name)
    return st.session_state.chain


def render_header():
    st.title("Campaign Chatbot with Memory")
    st.caption("Gemini via LangChain + ConversationBufferMemory")


def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "Model",
            options=["gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
            help="Flash is faster/cheaper and helps avoid free-tier quota limits",
        )
        st.session_state.selected_model = model
        st.markdown("[Get an API key](https://aistudio.google.com/apikey)")
        return model


def main():
    load_dotenv()
    render_header()
    selected_model = render_sidebar()

    chain = get_chain(model_name=selected_model)

    # Display history on first render
    if "history_rendered" not in st.session_state:
        st.session_state.history_rendered = True
        # Get the chat history from the session
        session_history = chain.get_session_history("default")
        for message in session_history.messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

    user_input = st.chat_input("Ask for a campaign idea or iterate on one...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use the new chain format with session history
                    response = chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "default"}}
                    )
                    st.markdown(response.content)
                except Exception as e:
                    # Surface quota or API errors without crashing the app
                    st.error(
                        "The model could not process your request. If you're on the free tier, you may have hit rate/usage limits. Try switching to 'gemini-1.5-flash', simplifying the prompt, or waiting a minute.\n\nDetails: "
                        + str(e)
                    )


if __name__ == "__main__":
    main()
