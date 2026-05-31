import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


load_dotenv()


def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY")


def demo_chatbot():
    api_key = get_api_key()

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Add it in Streamlit Secrets.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key,
        max_output_tokens=1024,
        convert_system_message_to_human=True
    )

    return llm


def demo_memory():
    llm = demo_chatbot()

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=False,
        memory_key="history",
        input_key="input"
    )

    return memory


CHAT_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a helpful, friendly AI assistant.
You remember the ongoing conversation and answer clearly.

Conversation so far:
{history}

User: {input}
Assistant:"""
)


def demo_conversation(input_text, memory):
    llm = demo_chatbot()

    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=CHAT_PROMPT,
        verbose=False
    )

    try:
        chat_reply = conversation_chain.predict(input=input_text)
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory
