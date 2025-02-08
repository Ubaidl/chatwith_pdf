import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv()
mykey = os.getenv("API_KEY")
# loader=PyPDFLoader("SUSPENDED_PARTICULATE_MATTERS[1][1][1]updated[1] updatednew.pdf")
# document=loader.load()
# print(document)

# #Function to extract text from the uploaded PDF
import tempfile

# Function to extract text from the uploaded PDF
def get_text_from_pdf(pdf_doc):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_doc.read())
        temp_file_path = temp_file.name

    # Load the document using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    document = loader.load()

    return document
  

# Function to split text into manageable chunks
def text_into_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = chunks = text_splitter.split_documents(document)
    return chunks

# Function to create a FAISS vector from chunks
def create_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embedding)
    return vectorstore

# Function to set up the conversational chat model and prompt
def conversational_chat():
    prompt_temp = """ 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details and 
    if the answer is not available, don't provide the wrong answer.
    context: {context}
    question: {question}
    """ 

    model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=mykey,
        temperature=0.7
    )  
    prompt = PromptTemplate(template=prompt_temp, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process the user input and return the chat response
def user_input(user_question, vectorstore):
    docs = vectorstore.similarity_search(user_question)
    chain = conversational_chat()
    response = chain({"input_documents": docs, "question": f"{user_question}"}, return_only_outputs=True)
    return response.get("output_text", "No answer found.")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat With PDF", layout="wide")
    st.header("Chat with PDF using Llama Model")

    # Initialize session state for vectorstore
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar menu for PDF upload
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload a PDF file", type="pdf")
        if st.button("Submit and Process"):
            if pdf_docs is not None:
                with st.spinner("Processing..."):
                    raw_text = get_text_from_pdf(pdf_docs)
                    text_chunks = text_into_chunks(raw_text)
                    st.session_state.vectorstore = create_vectorstore(text_chunks)
                    st.success("Processing complete! Vectorstore is ready.")
            else:
                st.error("Please upload a PDF file.")

    # Text input for user questions
    user_question = st.text_input("Ask a question:")
    if user_question and st.session_state.vectorstore is not None:
        with st.spinner("Fetching response..."):
            response = user_input(user_question, st.session_state.vectorstore)
            st.write(response)
    elif user_question and st.session_state.vectorstore is None:
        st.error("Please upload and process a PDF file first.")

if __name__ == "__main__":
    main()
