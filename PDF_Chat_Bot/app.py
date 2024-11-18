import os
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
        layout="wide")

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

            .st-emotion-cache-1jicfl2 {
                width: 100%;
                padding: 5px 5px 5px 5px;
                min-width: auto;
                max-width: initial;
            }
            
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
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

# Create a conversational QA chain
def get_conversational_chain():
    """
    Creates a question-answering chain using Langchain and Google Generative AI.
    
    Returns:
        Langchain chain: Configured QA chain for handling user questions.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context from single of multiple pdfs"
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Clear chat history in the session state
def clear_chat_history():
    """
    Clears the chat history stored in the session state.
    """
    st.session_state.messages = [{"role": "assistant", "content": "Ask me any question."}]

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
    
    return response

def main():
    """
    Main function to drive the application. Handles the user interface, file uploads, and interactions.
    """
    
    # Main content area
    st.title("PDF QueryBot")
    st.write("Effortless Conversations with Your Documents!")
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Processing done. Now you can ask questions.")

    # Only allow querying if PDFs are uploaded and processed
    if "faiss_index" in os.listdir() and pdf_docs:  
        # Initialize chat history if not present
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "Ask me any question."}]

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
                    with st.spinner("Generating Answer..."):
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

            st.button('Clear Chat History', on_click=clear_chat_history)
    else:
        st.write("Please upload some PDFs and process them before asking questions.")
    
if __name__ == "__main__":
    style_page()
    main()
