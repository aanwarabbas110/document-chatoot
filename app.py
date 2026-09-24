import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Anwar Ali's AI Document Chatbot", page_icon="📚", layout="centered")

st.title("📚 Anwar Ali's AI Document Q&A Chatbot")
st.write("Document upload karein, khulasa (summary) dekhein, aur apni pasandeeda zubaan mein sawal poochein!")

# Automatically fetch API key from .env file
# Fetch API key from Streamlit Secrets or Fallback to .env for local
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except:
    api_key = os.getenv("OPENROUTER_API_KEY")

# Sidebar Configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("**Developer:** Anwar Ali")
if api_key:
    st.sidebar.success("API Key loaded from `.env`!")
else:
    st.sidebar.warning("`.env` mein key nahi mili. Yahan enter karein:")
    api_key = st.sidebar.text_input("Enter OpenRouter API Key", type="password")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "openai/gpt-3.5-turbo",
        "meta-llama/llama-3-8b-instruct:free"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Language Settings")

# Summary Language Selection
summary_lang = st.sidebar.selectbox(
    "Summary Language",
    ["Roman Urdu", "English"]
)

# Question Response Language Selection
qa_lang = st.sidebar.selectbox(
    "Question Response Language",
    ["Roman Urdu", "English"]
)

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF or Word file", type=["pdf", "docx"])

if uploaded_file is not None and api_key:
    # Save uploaded file temporarily
    file_extension = uploaded_file.name.split(".")[-1]
    temp_file_path = f"temp.{file_extension}"
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load Document based on file type
    with st.spinner("Document process ho raha hai... bara-e-karam intezar karein..."):
        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == "docx":
            loader = Docx2txtLoader(temp_file_path)
        
        documents = loader.load()

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Embeddings & Vector Store (FAISS)
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # LLM Setup
        llm = ChatOpenAI(
            model_name=model_choice,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3
        )

    st.success("Document successfully analyze ho gaya hai!")

    # --- AUTO-SUMMARY BUTTON WITH LANGUAGE SUPPORT ---
    st.subheader("📌 Document Summary")
    if st.button("📄 Summarize Document"):
        with st.spinner("Document ka khulasa tayaar ho raha hai..."):
            summary_context = "\n\n".join([doc.page_content for doc in chunks[:3]])
            
            if summary_lang == "English":
                lang_instruction = "Provide a comprehensive and clear summary of this document in professional English (5-6 lines)."
            else:
                lang_instruction = "Is document ke text ko parh kar 5 se 6 lines ka aik behtareen aur comprehensive khulasa (summary) Roman Urdu mein likhein."

            summary_prompt = f"""
            Aap ek expert summarizer hain. {lang_instruction}
            
            Text:
            {summary_context}
            
            Summary:
            """
            summary_response = llm.invoke(summary_prompt)
            st.info(summary_response.content)

    st.markdown("---")
    st.subheader("💬 Ask Questions")

    # Initialize Chat History in Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box
    if user_query := st.chat_input("Apne document ke mutaliq sawal poochein..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve relevant chunks from FAISS (with source metadata)
        with st.spinner("Jawab tayaar ho raha hai..."):
            relevant_docs = vector_store.similarity_search(user_query, k=3)
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Dynamic instruction for QA language
            if qa_lang == "English":
                qa_instruction = "Provide a detailed and accurate answer to the user's question based on the given context in professional English. If the answer is not present in the context, clearly state: 'I could not find the answer in the document.'"
            else:
                qa_instruction = "Aap ek madadgar AI assistant hain. Diye gaye context ki buniyad par user ke sawal ka mukammal aur durust jawab Roman Urdu mein dein. Agar jawab context mein mojood na ho, toh saaf keh dein ke 'Mujhe document mein iska jawab nahi mila.'"

            # Prompt template
            prompt = f"""
            {qa_instruction}
            
            Context:
            {context}

            Sawal / Question: {user_query}
            Jawab / Answer:
            """

            # Generate response
            response = llm.invoke(prompt)
            answer = response.content

            # --- SOURCE CITATIONS / PAGE NUMBERS ---
            sources_set = set()
            for doc in relevant_docs:
                page_num = doc.metadata.get("page", None)
                if page_num is not None:
                    sources_set.add(str(page_num + 1))
                else:
                    sources_set.add("Document")
            
            sources_str = ", ".join(sources_set)
            final_answer = f"{answer}\n\n📌 **Source / Page(s):** {sources_str}"

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        with st.chat_message("assistant"):
            st.markdown(final_answer)

elif uploaded_file is not None and not api_key:
    st.warning("Bara-e-karam apni OpenRouter API Key `.env` file mein ya sidebar mein enter karein.")
else:
    st.info("Shuru karne ke liye PDF ya Word file upload karein.")