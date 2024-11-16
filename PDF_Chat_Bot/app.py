import os
import logging
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader


# Load environment variables from the .env file
load_dotenv()


# Set up logging to both console and a file
def setup_logging():
    """
    Sets up logging to both console and file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler('app_logs.log')
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the file handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Optionally, add a stream handler to output logs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

# Set up logging when the script starts
logger = setup_logging()


# Check for Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Styling for the Streamlit page
def style_page():
    """
    Customize the layout and appearance of the Streamlit page. 
    Hides elements like the menu, footer, and header for a clean UI.
    """
    st.set_page_config(
        page_title="PDF QueryBot",
        page_icon="",
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    hide_st_style = """
            <style>
            [data-testid="stAppViewContainer"] {
                background-image: url('https://digitalsynopsis.com/wp-content/uploads/2017/07/beautiful-color-ui-gradients-backgrounds-royal.png');
                background-size: cover;
                margin-top: 0px; /* Set margin-top to 0px to remove space */
            }

            #MainMenu {visibility: hidden;}
            footer{visibility: hidden;}
            header {visibility: hidden;}
            .st-emotion-cache-18ni7ap {visibility: hidden;}
            .st-emotion-cache-1avcm0n.ezrtsby2{visibility: hidden;}
            .st-emotion-cache-z5fcl4 {width: 100%; padding: 0rem 1rem 1rem;} 
            .st-emotion-cache-10trblm{text-align: center;}
            .st-emotion-cache-qdbtli {
                width: 100%;
                padding: 5px 5px 5px 5px;
                n-width: auto;
                max-width: initial;
                }
            .st-emotion-cache-kgpedg{
                padding: 0rem 0rem 0rem;
            }
            .st-emotion-cache-1gwvy71 {
                padding: 5px;
            }

            .st-emotion-cache-6qob1r{
                background-color: rgba(8, 22, 40, 0.8);
            }
            .st-emotion-cache-7tauuy{padding: 3rem 1rem 1rem;}
            
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)



# Extract text from PDF
def get_pdf_text(pdf_docs):
    """
    Extracts text content from uploaded PDF files.
    
    Args:
        pdf_docs (list): List of uploaded PDF files.
        
    Returns:
        str: Concatenated string containing text from all pages of all PDFs.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks for processing
def get_text_chunks(text):
    """
    Split the extracted text into smaller chunks for embedding processing.
    
    Args:
        text (str): Extracted text from PDF files.
        
    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# Generate vector store from chunks of text
def get_vector_store(chunks):
    """
    Generate a FAISS vector store from text chunks, and save it locally.
    
    Args:
        chunks (list): List of text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logger.info("Vector store saved locally.")

# Create a conversational QA chain
def get_conversational_chain():
    """
    Creates a question-answering chain using Langchain and Google Generative AI.
    
    Returns:
        Langchain chain: Configured QA chain for handling user questions.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context and try to find out that question. please answer the question from the context only. if you cannot answer the question from the context then please reply, "Question can not be answered from the provided context"
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Clear chat history in the session state
def clear_chat_history():
    """
    Clears the chat history stored in the session state.
    """
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]
    logger.info("Chat history cleared.")

# Handle user input and fetch response from vector store
def user_input(user_question):
    """
    Handles the user's query by searching the vector store and getting an answer.
    
    Args:
        user_question (str): The question provided by the user.
        
    Returns:
        dict: The response generated by the model based on the vector store search.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    logger.info(f"User question: {user_question}")
    logger.info(f"Response generated: {response['output_text']}")
    
    return response

def main():
    """
    Main function to drive the application. Handles the user interface, file uploads, and interactions.
    """
    # Sidebar for uploading PDFs
    st.sidebar.title('PDF QueryBot')
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload PDF Files then Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing done. Now you can ask questions.")

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Main content area
    st.title("PDF QueryBot")
    st.write("Effortless Conversations with Your Documents!")

    # Only allow querying if PDFs are uploaded and processed
    if "faiss_index" in os.listdir() and pdf_docs:  
        # Initialize chat history if not present
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

        # Display the chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Handling user input and displaying response
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Get the bot's response to the user input
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = user_input(prompt)
                        placeholder = st.empty()
                        full_response = ''
                        for item in response['output_text']:
                            full_response += item
                            placeholder.markdown(full_response)
                        placeholder.markdown(full_response)

                # Append the assistant's response to the session state
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)
                logger.info(f"Assistant response: {full_response}")
    else:
        st.write("Please upload some PDFs and process them before asking questions.")

if __name__ == "__main__":
    style_page()
    main()
