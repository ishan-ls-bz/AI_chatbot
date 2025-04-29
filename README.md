# QUIC.cloud & LiteSpeed Assistant

An AI-powered chatbot built with Streamlit that provides expert support and information for QUIC.cloud and LiteSpeed Technologies.

## Features

- Interactive chat interface
- Authentication system
- Retrieval-augmented generation using Pinecone vector store
- Groq LLM integration (using LLaMA-4 Scout 17B model)
- Real-time streaming responses

## Code Structure

The application is built around several key components:

- **Authentication System**: Secures the application with username/password login
- **Vector Database**: Uses Pinecone to store and retrieve relevant documentation
- **LLM Integration**: Leverages Groq's LLaMA-4 model for fast, accurate responses
- **Conversational Memory**: Maintains context through a window-based memory system
- **Custom Prompting**: Uses a tailored prompt template to ensure consistent, helpful responses

### Key Functions

- `authenticate()`: Handles user login and session management
- `create_retrieval_chain()`: Sets up the RAG pipeline with LangChain
- `format_response()`: Ensures consistent formatting for chat messages
- `generate_response_chunks()`: Creates streaming response chunks for a natural chat experience
- `main()`: The core application runner that sets up the Streamlit interface

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/ishan-ls-bz/AI_chatbot.git
   cd AI_chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `secrets.toml` file in the `.streamlit` directory with your API keys:
   ```toml
   [api_keys]
   openai = "your-openai-api-key"
   pinecone = "your-pinecone-api-key"
   groq = "your-groq-api-key"

   [credentials]
   username = "your-username"
   password = "your-password"
   ```

4. Run the application:
   ```
   streamlit run streamlit_app.py
   ```

## Making Changes and Pushing to GitHub

After making changes to your code:

```bash
# Check modified files
git status

# Add changes
git add .

# Commit changes with a descriptive message
git commit -m "Description of changes"

# Push to GitHub
git push origin main
```

## Technology Stack

- **Streamlit**: Web interface for building the chat application
- **LangChain**: Framework for LLM application development
- **Pinecone**: Vector database for efficient document retrieval
- **Groq**: LLM provider for fast inference with LLaMA-4 models
- **OpenAI**: Embeddings generation for document vectorization

## Security Considerations

- All API keys and credentials are stored in `.streamlit/secrets.toml` (not committed to Git)
- Authentication is required before accessing the chatbot
- Session management to maintain user context securely

## Future Enhancements

- Multi-user support with different permission levels
- Additional knowledge sources for the retrieval system
- Integration with other LLM providers
- Advanced analytics on usage patterns
