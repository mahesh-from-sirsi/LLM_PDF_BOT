# Importing necessary libraries
import streamlit as st                                        # Library for Streamlit App
from PyPDF2 import PdfReader                                  # Library for reading the PDF files
from langchain_openai import ChatOpenAI                       # Library for creating the ChatAPI model
from langchain.text_splitter import CharacterTextSplitter     # Library for creating the chunks of data read from a PDF file
from langchain_community.vectorstores import FAISS            # Vector Store from FaceBook
from langchain_openai import OpenAIEmbeddings                 # Library for creating the Embedding Models 
from langchain.memory import ConversationBufferMemory         # Library for adding the memory to the LLM to retail last n contextx (query + response) 
from langchain.chains import ConversationalRetrievalChain     # Library for simplifyling the user query to response (which otherwise needs query embedding, search vector store, split and parse etc)
from langchain.prompts import PromptTemplate                  # Library for the Prompt template where same template can be used for different use cases
import os                                                     # Library for OS Specific Operations (env variables/file paths etc)


# Set OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page configuration
st.set_page_config(page_title="Chat with PDF - AI Specific", page_icon="📚")
st.title("Chat with your PDF 📚")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

# Method to extract the PDF file content and return the text
# NOTE: THIS IS GOOD FOR TEXT ONLY and NOT FOR IMAGES/TABLES/CHARTS etc.
# One Has to use the specialised PDF extractors
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Method to chunk the text for embeddings for RAG use case
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Get the conversation chain for getting the response to our queries from the LLM
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o')
    
    template = """You are a helpful AI assistant that helps users understand their PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, or if the question is outside the scope of their PDF documents, 
    just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return conversation_chain

def process_docs(pdf_docs):
    try:
        # Get PDF text
        raw_text = get_pdf_text(pdf_docs)
        
        # Get text chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store using FAISS
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
        
        st.session_state.processComplete = True
        
        return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            success = process_docs(pdf_docs)
            if success:
                st.success("Processing complete!")
            else:
                st.success("Failed to process the PDF document!")

# Main chat interface
if st.session_state.processComplete:
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({
                    "question": user_question
                })
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

# Display initial instructions
else:
    st.write("👈 Upload your PDFs in the sidebar to get started!")

