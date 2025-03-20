from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

import os
import streamlit as st

from llm import load_model, set_custom_prompt

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def main():
    st.title("Ask Chatbot!!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message["content"])

    prompt=st.chat_input("Pass your query here --->")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({ 'role': 'user', 'content': prompt })

        try:
            store = get_vectorstore()
            if store is None:
                st.error("Failed to load vector store!")
                
            llm = load_model(HUGGINGFACE_REPO_ID)
            qa_chain = RetrievalQA.from_chain_type(
                llm= llm,
                chain_type="stuff",
                retriever=store.as_retriever(search_kwargs={ "k":3 }),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": set_custom_prompt()
                },
            )

            response = qa_chain.invoke({ 'query': prompt })
            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({ 'role': 'assistant', 'content': result })

        except Exception as e:
            st.error(f"Error raised {str(e)}")

if __name__ == "__main__":
    main()