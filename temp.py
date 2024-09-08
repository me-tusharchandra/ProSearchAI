
import io
import os
import markdown
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configure Google API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize models
model_text = genai.GenerativeModel(model_name="gemini-1.5-flash")
model_vision = genai.GenerativeModel(model_name="gemini-pro-vision")

# Use a more powerful embedding model for better semantic search
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
loaded_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Chat history to maintain context for follow-up queries
chat_history = []

def generate_response_from_model(user_query, documents, model):
    context = "\n\n".join([
        f"Product {i+1}: {doc.page_content}\n"
        f"Description: {doc.metadata.get('description', 'N/A')}\n"
        f"Brand: {doc.metadata.get('brand', 'N/A')}\n"
        f"Image: {doc.metadata.get('imgUrl', 'N/A')}\n"
        f"URL: {doc.metadata.get('productURL', 'N/A')}\n"
        f"Original Price: {doc.metadata.get('retailPrice', 'N/A')}\n"
        f"Sale Price: {doc.metadata.get('discountedPrice', 'N/A')}\n"
        # f"Rating: {doc.metadata.get('productRating', 'N/A')}\n"
        f"Overall Rating: {doc.metadata.get('overallRating', 'N/A')}"
        for i, doc in enumerate(documents)
    ])

    # Join chat history into a single string
    chat_history_str = "\n".join(chat_history)

    prompt = f"""
    **Prompt:**

    You are an advanced search assistant designed to enhance user experience by providing semantic search results rather than traditional keyword-based responses. Your main objectives are as follows:

    1. **Chat History Consideration**: 
    - This might be a follow-up query. Based on the chat history:
    {chat_history_str}

    2. **User Query**: 
    - Here's what the customer asked: "{user_query}"

    3. **Search Results**:
    - Here are the search results based on the user query:
    {context}

    4. **Initial Search Response**: 
    - When a user initiates a search, analyze the semantic meaning of their query to identify exactly three relevant products unless otherwise specified. For each product, provide:
        - A brief description.
        - The product link (mandatory).
        - Relevant meta information without overwhelming the user.
        - Use proper numbering for the list of products.

    5. **Follow-up Interaction**: 
    - After presenting the initial search results, determine if the user has a follow-up question or if they wish to conduct a new search. If the user engages with a follow-up question, respond appropriately based on their request.

    6. **Product Comparison**: 
    - If the user expresses a desire to compare two products, facilitate a side-by-side comparison by extracting key features and differences between the selected products in a clear and concise format.

    7. **Contextual Meta Information**: 
    - Provide additional information (such as images, specifications, and reviews) only when specifically requested by the user to keep the initial interaction streamlined and focused.

    8. **Goal Orientation**: 
    - Constantly aim to expedite the search process and save time for the user, ensuring that the interaction feels efficient and user-friendly.

    9. **User Engagement**: 
    - Keep track of user preferences and behaviors to refine your responses and improve future interactions. Always prioritize clarity, relevance, and user satisfaction in your communications.

    Please ensure that your responses align with these objectives and maintain a professional and informative tone throughout your interactions.
    """

    response = model.generate_content(prompt)
    return response.text

def process_user_query(user_input, image, vectorstore, model):
    if image:
        # If an image is provided, use the vision model
        prompt = f"Describe this image and suggest similar products. User query: {user_input}"
        response = model_vision.generate_content([prompt, image])
        text_query = response.text
    else:
        text_query = user_input

    search_results = vectorstore.similarity_search(text_query, k=10)
    
    if not search_results:
        return "I couldn't find any matching products. Could you please try rephrasing your request?"

    response_text = generate_response_from_model(user_input, search_results, model)
    
    # Add the user query and response to chat history
    chat_history.append(f"User: {user_input}\nAssistant: {response_text}\n")
    
    return markdown.markdown(response_text)

# Streamlit UI
st.title("ProSearchAI: Multimodal AI-powered Search Tool")

st.write("""
This tool uses advanced AI to understand the semantic meaning of your search,
going beyond traditional keyword-based algorithms. You can input text in any language,
and even upload an image to find similar products!
""")

# Text input
user_input = st.text_input("Enter your search query in any language:")

# Image upload
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    image = None

if st.button("Search"):
    if user_input or image:
        with st.spinner("Searching for products..."):
            response = process_user_query(user_input, image, loaded_vectorstore, model_text)
        # st.markdown(response, unsafe_allow_html=True)
    else:
        st.warning("Please enter a search query or upload an image.")

# Display chat history
st.write("### Chat History")
for i, entry in enumerate(chat_history):
    st.markdown(entry)
