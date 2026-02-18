# import streamlit as st
# from rag_chain_llm import rag_chain


# st.set_page_config(page_title="RAG PDF Chatbot", page_icon="🤖")
# st.title("RAG PDF Chatbot with Google Gemini + FAISS")

# # Initialize RAG chain
# chain = rag_chain()

# # User input
# query = st.text_input("Ask something:", placeholder="Type your question here...")

# if st.button("Submit") and query:
#     with st.spinner("Generating answer..."):
#         try:
#             output = chain.invoke(query)
#             st.success("Answer:")
#             st.write(output.content)
#         except Exception as e:
#             st.error(f"Error: {str(e)}")

import streamlit as st
import requests

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="🤖")
st.title("RAG PDF Chatbot with Google Gemini + FAISS")


# User input
query = st.text_input("Ask something:", placeholder="Type your question here...")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a question first.")
    else:
        # Call FastAPI backend
        try:
            response = requests.post("http://localhost:8000/ask", json={"question": query})
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.success(answer)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")

