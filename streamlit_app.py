import streamlit as st
import os
import time
import uuid
import re
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate

# Page title and configuration
st.set_page_config(
    page_title="QUIC.cloud & LiteSpeed Assistant",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for button colors and improving loading display
st.markdown("""
<style>
    .stButton button {
        background-color: #FF4B4B;
        color: white;
    }
    .stButton button:hover {
        background-color: #D44141;
        color: white;
    }
    .stButton button:focus {
        background-color: #D44141;
        color: white;
    }
    /* Hide empty streamlit elements */
    .element-container:has(.stSpinner) + div:has(div:empty) {
        display: none;
    }
    /* Improved styling for chat messages */
    .stChatMessage div {
        max-width: 95% !important; 
    }
    /* Better spacing for paragraphs in chat */
    .stChatMessage p {
        margin-bottom: 0.8em !important;
    }
    /* Improve code block readability */
    .stChatMessage code {
        white-space: pre-wrap !important;
    }
</style>
""", unsafe_allow_html=True)

# Authentication function
def authenticate():
    """Authenticates the user before proceeding."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Center the login form with custom CSS
        st.markdown("""
            <style>
            .centered {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                width: 300px;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Login container
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.title("🔒 Login")
                username = st.text_input("Username", key="username_input")
                password = st.text_input("Password", type="password", key="password_input")
                login_button = st.button("Login")
                
                if login_button:
                    # Check credentials against stored values in secrets.toml
                    if (username == st.secrets["credentials"]["username"] and 
                        password == st.secrets["credentials"]["password"]):
                        # Create a unique session ID if one doesn't exist
                        if "session_id" not in st.session_state:
                            st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.authenticated = True
                        # Initialize memory on login with window size of 3
                        st.session_state.memory = ConversationBufferWindowMemory(
                            return_messages=True,
                            memory_key="chat_history",
                            output_key="answer",
                            k=3  # Only keep the 3 most recent message exchanges
                        )
                        # Reset messages on new login
                        st.session_state.messages = []
                        # Ensure all state variables are reset
                        st.session_state.processing_query = False
                        st.session_state.current_response = None
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        # Return False to prevent the rest of the app from loading
        return False
    
    return True

# Set API keys from secrets
def set_api_keys():
    os.environ['OPENAI_API_KEY'] = st.secrets["api_keys"]["openai"]
    os.environ['PINECONE_API_KEY'] = st.secrets["api_keys"]["pinecone"]
    os.environ['GROQ_API_KEY'] = st.secrets["api_keys"]["groq"]

# Custom prompt template for expert responses
CUSTOM_PROMPT = """
# Role and Identity
You are an expert technical support specialist EXCLUSIVELY for QUIC.cloud and LiteSpeed Technologies.
You have deep knowledge of:
- WordPress optimization and caching
- QUIC.cloud CDN, Image Optimization, and other services
- LiteSpeed Web Server and LiteSpeed Cache plugin
- DNS configuration, domains, and subdomains
- Server administration and PHP configuration
- Website performance optimization

# Information Context
Analyze the following information from the QUIC.cloud and LiteSpeed documentation:

<context>
{context}
</context>

User Question: {question}

# ⚠️ STRICT TOPIC ENFORCEMENT - HIGHEST PRIORITY ⚠️
- You are FORBIDDEN from answering ANY questions not DIRECTLY related to QUIC.cloud or LiteSpeed documentation.
- For ANY off-topic questions, respond ONLY with: "I'm sorry, but I can only provide information about QUIC.cloud and LiteSpeed products and services. Please ask a question related to these topics."
- FORBIDDEN TOPICS (always refuse these):
  * General Linux/Unix commands or administration not specific to LiteSpeed
  * Generic programming help or coding questions
  * General server management unrelated to LiteSpeed
  * Web development topics not specific to QUIC.cloud/LiteSpeed
  * Competitor products or services
  * Non-technical or personal questions
  * Instructions for tasks unrelated to QUIC.cloud/LiteSpeed
- EXAMPLES of forbidden questions (NEVER answer these):
  * "How do I delete files older than 7 days in Linux?"
  * "Give me a bash script to automate backups"
  * "How do I set up Nginx with WordPress?"
  * "What's the best way to learn JavaScript?"
- NEVER try to relate off-topic questions back to LiteSpeed/QUIC.cloud
- DO NOT provide workarounds or partial answers to off-topic questions
- If uncertain if a question is on-topic, err on the side of refusing to answer

# Response Guidelines (ONLY for on-topic questions)
1. ANSWER FIRST, REFER SECOND:
   - Provide a complete, detailed answer to the user's question first
   - Only after fully answering the question, mention relevant documentation links
   - Never just tell the user to check documentation without first providing a substantive answer
   - Include all necessary details in your answer rather than relying on references

2. PERSONALIZED RESPONSES (HIGHEST PRIORITY):
   - ALWAYS use any personal information the user provides to personalize your response
   - If they mention their domain name (e.g., example.com), USE IT in your examples and solutions
   - If they mention specific server configurations, plugin versions, or other personal details, INCORPORATE these into your answer
   - Show how solutions apply specifically to their situation, not just generic advice
   - Address their exact scenario rather than providing general information
   - Adapt technical explanations to match their apparent level of expertise

3. KNOWLEDGE BOUNDARIES:
   - Answer based ONLY on the information provided in the context
   - If the question is partially covered, address what you can and acknowledge limitations
   - If the question is completely unrelated to QUIC.cloud or LiteSpeed, use the EXACT refusal message specified above
   - Do not make up information if it's not in the context

4. DETAIL AND DEPTH:
   - Provide comprehensive, well-explained responses that demonstrate expertise
   - Include technical details when appropriate for the user's question
   - Use examples to illustrate concepts where helpful
   - Explain technical reasoning behind recommendations
   - Don't oversimplify complex topics - provide sufficient depth for understanding

5. CLARITY AND STRUCTURE:
   - Begin with a direct answer to the primary question
   - Organize longer responses with a logical flow of information
   - For technical procedures, include clear numbered steps
   - Use paragraphs to separate distinct ideas or aspects of the answer
   - End with a summary or conclusion that reinforces key points

6. CITATIONS:
   - Only after providing a complete answer, reference sources using the permanent_link from the context
   - Mention specific documentation sections when relevant
   - Present references as supplementary material, not primary answers

7. TONE AND STYLE:
   - Be professional, helpful, and technically precise
   - Use clear language that matches the user's technical level
   - For formatting, use ONLY bold (**text**), italic (*text*), or numbered lists
   - DO NOT use large headings or excessive formatting
   - Keep your formatting style consistent throughout the response
   - Use plain text for most content with selective bold or italic for emphasis

FINAL REMINDER: You MUST ONLY answer questions about QUIC.cloud and LiteSpeed products/services based on the provided documentation. For ANY off-topic question, respond ONLY with the exact refusal message. No exceptions.
"""

# Define custom prompt template
prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT)

# Format response to ensure consistent styling
def format_response(text):
    """Ensure response has consistent formatting without large headings."""
    # Remove any markdown headings (# and ##) but keep the text
    text = re.sub(r'^#\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
    # Keep ### as bold
    text = re.sub(r'^###\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
    # Keep #### and beyond as is, they're small enough
    
    # Ensure consistent spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

# Function to generate response chunks
def generate_response_chunks(result):
    """Generate response chunks from the response dictionary."""
    answer = result['answer']
    # Process the answer to ensure consistent formatting
    answer = format_response(answer)
    words = answer.split(' ')
    
    # Yield words one by one
    current_text = ""
    for word in words:
        current_text += word + " "
        yield word + " "  # Yield just the current word
        time.sleep(0.03)  # Add a slight delay between words for natural typing effect

# Create retrieval chain
def create_retrieval_chain(vectorstore):
    """Create a retrieval chain with a new memory instance to avoid session overlaps"""
    # Get memory from session state to ensure it's session-specific
    memory = st.session_state.memory
    
    # Initialize Groq LLM with moderate max_tokens
    llm = ChatGroq(
        temperature=0.3,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens=4096  # Moderate token limit to balance detail and efficiency
    )
    
    # Create the retrieval chain with custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve 5 documents for more comprehensive context
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )
    
    return qa_chain

# Main application function
def main():
    # Header
    st.title("🚀 QUIC.cloud & LiteSpeed Assistant")
    st.markdown("""
    This assistant provides detailed answers about QUIC.cloud and LiteSpeed products and services.
    Ask questions about CDN configuration, WordPress optimization, DNS settings, and more!
    """)

    # Initialize session state for conversation history display
    # Limited to most recent 4 exchanges for UI display
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize flag to track if we're processing a new query
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
        
    # Initialize current response buffer
    if "current_response" not in st.session_state:
        st.session_state.current_response = None
    
    # Keep UI chat history limited to 4 exchanges (8 messages)
    if len(st.session_state.messages) > 8:
        st.session_state.messages = st.session_state.messages[-8:]
        
    # Ensure memory is initialized with window size of 3
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            k=3  # Only keep the 3 most recent message exchanges
        )

    # Initialize embeddings and vectorstore only once
    @st.cache_resource
    def initialize_vectorstore():
        """Initialize embeddings and vectorstore"""
        try:
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            
            # Initialize Pinecone vector store
            index_name = "docschatbot"
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
            
            return vectorstore, True
        except Exception as e:
            st.error(f"Error initializing the vector store: {str(e)}")
            return None, False
    
    # Initialize vector store
    vectorstore, vectorstore_success = initialize_vectorstore()
    
    # Display previous chat messages (limited to most recent)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Additional Reference Material"):
                    st.markdown("_The following resources contain more information on this topic:_")
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{idx}. {source['title']}**")
                        st.markdown(f"[Read more in documentation]({source['permanent_link']})")

    # Get user input
    user_query = st.chat_input("Ask a question about QUIC.cloud or LiteSpeed...")

    # Process user query
    if user_query and vectorstore_success and not st.session_state.processing_query:
        # Set processing flag to prevent duplicate processing
        st.session_state.processing_query = True
        
        # Add user message to chat history UI
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Keep UI chat history limited to 4 exchanges (8 messages)
        if len(st.session_state.messages) > 8:
            st.session_state.messages = st.session_state.messages[-8:]
        
        # Display user message in a new container to avoid conflicting with history display
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Create a dedicated assistant container for the new response
        assistant_placeholder = st.empty()
        with assistant_placeholder.chat_message("assistant"):
            # Create placeholder for content
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            # Create retrieval chain with fresh memory instance
            qa_chain = create_retrieval_chain(vectorstore)
            
            with st.spinner("Searching documentation..."):
                start_time = time.time()
                
                try:
                    # Process the query
                    result = qa_chain.invoke({"question": user_query})
                    
                    # Extract sources
                    source_docs = result.get('source_documents', [])
                    
                    # Format source documents
                    sources = []
                    for doc in source_docs:
                        if doc.metadata:
                            sources.append({
                                'title': doc.metadata.get('title', 'Document'),
                                'url': doc.metadata.get('url', ''),
                                'permanent_link': doc.metadata.get('permanent_link', '')
                            })
                    
                    # Get the complete response first to avoid streaming issues
                    complete_response = format_response(result['answer'])
                    
                    # Use the message placeholder to display the response properly
                    message_placeholder.markdown(complete_response)
                    
                    # Display sources in an expander
                    if sources:
                        with sources_placeholder.expander("Additional Reference Material"):
                            st.markdown("_The following resources contain more information on this topic:_")
                            for idx, source in enumerate(sources, 1):
                                st.markdown(f"**{idx}. {source['title']}**")
                                if source['permanent_link']:
                                    st.markdown(f"[Read more in documentation]({source['permanent_link']})")
                    
                    # Store the response in session state for UI display
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": complete_response,
                        "sources": sources
                    })
                    
                    # Keep UI chat history limited to 4 exchanges (8 messages)
                    if len(st.session_state.messages) > 8:
                        st.session_state.messages = st.session_state.messages[-8:]
                    
                    # Show response time
                    st.caption(f"Response time: {time.time() - start_time:.2f} seconds")
                    
                except Exception as e:
                    error_msg = f"Error processing your query: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    # Keep UI chat history limited to 4 exchanges (8 messages)
                    if len(st.session_state.messages) > 8:
                        st.session_state.messages = st.session_state.messages[-8:]
                
                # Reset processing flag when done
                st.session_state.processing_query = False
                st.session_state.current_response = None

    # Sidebar with information
    with st.sidebar:
        st.title("About this Assistant")
        st.markdown("""
        ### QUIC.cloud & LiteSpeed Assistant
        
        This AI assistant provides comprehensive answers about:
        - QUIC.cloud services and features
        - LiteSpeed Web Server configuration
        - LiteSpeed Cache for WordPress
        - CDN setup and optimization
        - Domain and DNS management
        - Caching strategies and best practices
        - Performance optimization techniques
        - WordPress plugin configuration
        
        ### Need More Help?
        
        If you can't find what you need, open a support ticket:
        [QUIC.cloud Support](https://account.quic.cloud/plugin/support_manager/client_tickets/departments/)
        
        Or email: support@quic.cloud
        """)
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            # Reset memory on conversation clear
            st.session_state.memory = ConversationBufferWindowMemory(
                return_messages=True,
                memory_key="chat_history",
                output_key="answer",
                k=3  # Only keep the 3 most recent message exchanges
            )
            # Reset processing flag
            st.session_state.processing_query = False
            st.session_state.current_response = None
            st.rerun()
            
        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False
            # Clear all session state on logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.authenticated = False
            st.rerun()

# Main execution
if __name__ == "__main__":
    # Set API keys from secrets
    set_api_keys()
    
    # If authentication succeeds, run the main app
    if authenticate():
        main() 
