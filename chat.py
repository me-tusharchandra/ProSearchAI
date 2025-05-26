import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Google API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
# model = genai.GenerativeModel(model_name="gemini-1.0-pro")

# Load embeddings
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Load the vector store
loaded_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Create a conversation memory
memory = ConversationBufferMemory(k=5)  # Keeps last 5 conversations

# Create a prompt template for query rephrasing
rephrase_template = """Given the following conversation history, please rephrase the current query to include essential keywords and context to help find the most relevant information about products:
Conversation history:
{history}
Current query: {query}
Rephrased query with context:"""
rephrase_prompt = PromptTemplate(
    input_variables=["history", "query"],
    template=rephrase_template
)

# Create a prompt template for the RAG response
rag_template = """You are an AI assistant for a product recommendation system. Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Human: {query}
Assistant:"""
rag_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=rag_template
)

# def rephrase_query(query, history):
#     # If history is empty, skip rephrasing based on history
#     if not history:
#         return query  # Return the original query if there's no history

#     prompt = rephrase_prompt.format(history=history, query=query)
#     response = model.generate_content(prompt)
#     return response.text

def rephrase_query(query, history):
    # Only rephrase if the query is ambiguous or too short
    if len(query.split()) < 5 or history:
        prompt = rephrase_prompt.format(history=history, query=query)
        response = model.generate_content(prompt)
        return response.text
    else:
        return query  # If query is clear enough, return as-is

def generate_rag_response(query, context):
    prompt = rag_prompt.format(context=context, query=query)
    response = model.generate_content(prompt)
    return response.text

def process_user_query(user_input, vectorstore, model, memory):
    # Get conversation history, default to an empty string if no history exists
    history = memory.load_memory_variables({}).get("history", "")

    # Rephrase the query, handling the case where history is empty
    rephrased_query = rephrase_query(user_input, history)

    # Perform a similarity search in the vector store based on the rephrased query
    search_results = vectorstore.similarity_search(rephrased_query, k=3)  # Retrieve top 3 similar products
    
    if len(search_results) == 0:
        # If no results are found, suggest refining the query
        return "I couldn't find any products matching your query. Please try again with different or more specific details."

    # Prepare context from search results
    context = "\n\n".join([f"Product: {doc.page_content}\nURL: {doc.metadata.get('url', 'No URL available')}" for doc in search_results])
    
    # Add both the rephrased query and the context to the model for better query response
    response = generate_rag_response(rephrased_query, context)
    
    # Check if the response indicates a lack of relevant information
    if "I don't know" in response or "no relevant context" in response:
        return "I couldn't find specific information related to your query. Please try again with more details or different keywords."

    # Update memory
    memory.save_context({"input": user_input}, {"output": response})

    return response

# Streamlit UI
st.set_page_config(page_title="ProSearchAI", page_icon="🔍", layout="wide")

st.title("ProSearchAI: AI-powered smart search tool")
st.markdown("This tool uses advanced AI to understand the semantic meaning of your search, going beyond traditional keyword-based algorithms.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to search for?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process the user query
    response = process_user_query(prompt, loaded_vectorstore, model, memory)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a sidebar with additional information
st.sidebar.title("About ProSearchAI")
st.sidebar.info(
    "ProSearchAI is an advanced product search tool that uses AI to understand "
    "the context and intent behind your queries. It provides smart recommendations "
    "based on a comprehensive product database."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Team Spambots <3")