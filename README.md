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
