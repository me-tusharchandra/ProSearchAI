# ProSearchAI - AI-Powered Product Search Tool

**ProSearchAI** is an advanced, semantic search tool that leverages AI to understand user intent and context, providing intelligent product recommendations beyond simple keyword-based matching. Powered by a combination of Google’s Generative AI, vector embeddings, and conversational memory, ProSearchAI makes product discovery intuitive and efficient.

## Use Case (Important)

**Before ProSearchAI**: Sam searches for a monitor with a blue light filter, height-adjustable frame, and VESA mount support, but Amazon's results are filled with irrelevant products, leaving him frustrated.

**After ProSearchAI**: Sam enters his requirements into ProSearchAI and instantly receives a curated list of monitors that match his exact needs, saving time and effort.

## Features

- **Semantic Search**: Understands the meaning of user queries, retrieving relevant products even with non-exact keyword matches.
- **Contextual Query Rephrasing**: Rephrases queries based on the conversation history to add contextual relevance.
- **Retrieval-Augmented Generation (RAG)**: Generates responses based on relevant context from previous interactions and retrieved product data.
- **Conversational Memory**: Tracks conversation history to build on previous queries for a cohesive and personalized search experience.
- **Streamlit Interface**: Interactive chat-based UI to facilitate intuitive searches and responses.

## Tech Stack

- **Google Generative AI (Gemini)**: For generating rephrased queries and contextual responses.
- **LangChain**: To handle embedding generation, vector storage, and conversation memory.
- **Chroma**: Persistent vector storage solution for efficient similarity search.
- **Streamlit**: For creating an interactive chat-based UI.

## Requirements

- Python 3.7+
- Required libraries: `streamlit`, `pandas`, `google-generativeai`, `langchain`, `chromadb`, `dotenv`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ProSearchAI.git
   cd ProSearchAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Launch** the app by running `streamlit run app.py`.
2. **Input Queries** in the chat window. ProSearchAI will process the input, consider the conversation context, and display product recommendations.
3. **Review Responses** displayed by ProSearchAI, which provide context-aware product suggestions.

## How it Works

1. **Query Rephrasing**: When a user submits a query, ProSearchAI rephrases it based on the conversation history using Google Generative AI.
2. **Vector Search**: The rephrased query is used to search for semantically similar product descriptions within Chroma's vector store.
3. **RAG Response Generation**: ProSearchAI formulates a response based on the retrieved products and the user’s specific requirements.
4. **Conversational Memory**: The conversation history is stored to allow for continuous, context-aware interactions.

## About the Team

**ProSearchAI** is crafted with passion by **Team Spambots**.