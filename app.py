import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader, 
    CSVLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Anwar Ali's AI Document Chatbot", page_icon="📚", layout="centered")

st.title("📚 Anwar Ali's AI Document Q&A Chatbot")
st.write("Aap chahay file upload karein ya apna text/notes paste karein — khulasa, Glossary, Q&A, MCQs aur Chat sab tayar hai!")

# Fetch API key from Streamlit Secrets or Fallback to .env for local
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except:
    api_key = os.getenv("OPENROUTER_API_KEY")

# Sidebar Configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("**Developer:** Anwar Ali")
if api_key:
    st.sidebar.success("API Key loaded successfully!")
else:
    st.sidebar.warning("API key nahi mili. Yahan enter karein:")
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

# Content Language Selection
content_lang = st.sidebar.selectbox(
    "Summary, Q&A, MCQs & Study Tools Language",
    ["Roman Urdu", "English"]
)

# Chat Response Language Selection
qa_lang = st.sidebar.selectbox(
    "Chat Response Language",
    ["Roman Urdu", "English"]
)

st.markdown("---")

# --- MULTI-INPUT SECTION (File Upload + Text Paste Simultaneously) ---
st.subheader("📂 Document Upload ya Text Paste Karein")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Option 1: File Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF, Word, TXT, MD, CSV", 
        type=["pdf", "docx", "txt", "md", "csv"]
    )

with col2:
    st.markdown("### Option 2: Paste Text")
    pasted_text = st.text_area(
        "Yahan apna text paste karein", 
        height=150, 
        placeholder="Apne notes yahan paste karein..."
    )

documents = []

# Process based on what user provided
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    temp_file_path = f"temp.{file_extension}"
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Uploaded document process ho raha hai..."):
        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == "docx":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == "txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")
        elif file_extension == "md":
            loader = UnstructuredMarkdownLoader(temp_file_path)
        elif file_extension == "csv":
            loader = CSVLoader(temp_file_path, encoding="utf-8")
        else:
            st.error("Unsupported file format!")
            st.stop()
        
        documents = loader.load()
        st.success("File kamyabi ke sath load ho gayi hai!")

elif pasted_text and pasted_text.strip() != "":
    documents = [Document(page_content=pasted_text, metadata={"page": 0})]
    st.success("Pasted text kamyabi ke sath load ho gaya hai!")

# Check if documents are loaded and API key is present
if documents and api_key:
    with st.spinner("AI model tayaar ho raha hai..."):
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

    st.success("Data successfully analyze ho gaya hai!")

    # --- 1. AUTO-SUMMARY SECTION ---
    st.subheader("📌 Document / Text Summary")
    if st.button("📄 Summarize Content"):
        with st.spinner("Khulasa tayaar ho raha hai..."):
            summary_context = "\n\n".join([doc.page_content for doc in chunks[:3]])
            
            if content_lang == "English":
                lang_instruction = "Provide a comprehensive and clear summary of this text in professional English (5-6 lines)."
            else:
                lang_instruction = "Is text ko parh kar 5 se 6 lines ka aik behtareen aur comprehensive khulasa (summary) Roman Urdu mein likhein."

            summary_prompt = f"""
            Aap ek expert summarizer hain. {lang_instruction}
            
            Text:
            {summary_context}
            
            Summary:
            """
            summary_response = llm.invoke(summary_prompt)
            st.info(summary_response.content)

    st.markdown("---")

    # --- 2. KEY TERMS & GLOSSARY SECTION ---
    st.subheader("📖 Key Terms & Glossary")
    if st.button("🔍 Generate Glossary"):
        with st.spinner("Key terms aur definitions tayaar ho rahe hain..."):
            glossary_context = "\n\n".join([doc.page_content for doc in chunks[:4]])
            
            if content_lang == "English":
                glossary_instruction = "Extract 5 to 6 key technical terms or important concepts from this text and provide clear definitions in professional English."
            else:
                glossary_instruction = "Is text mein se 5 se 6 ahem technical alfaz (Key Terms) dhoond kar unke mukammal ta'ruf aur definitions Roman Urdu mein likhein."

            glossary_prompt = f"""
            Aap ek expert professor hain. {glossary_instruction}
            
            Text:
            {glossary_context}
            
            Format:
            - **Term 1**: Definition...
            - **Term 2**: Definition...
            """
            glossary_response = llm.invoke(glossary_prompt)
            st.success("Yeh rahe Key Terms & Glossary:")
            st.write(glossary_response.content)

    st.markdown("---")

    # --- 3. Q&A GENERATOR SECTION ---
    st.subheader("❓ Generate Descriptive Q&A")
    if st.button("⚡ Generate Descriptive Q&A Pairs"):
        with st.spinner("Q&A tayaar ho rahe hain..."):
            qa_context = "\n\n".join([doc.page_content for doc in chunks[:4]])
            
            if content_lang == "English":
                qa_gen_instruction = "Generate 4 to 5 important descriptive questions and their accurate answers based on this text in professional English."
            else:
                qa_gen_instruction = "Is text mein se 4 ya 5 ahem descriptive sawal aur unke mukammal jawabaat Roman Urdu mein generate karein."

            qa_gen_prompt = f"""
            Aap ek expert tutor hain. {qa_gen_instruction}
            
            Text:
            {qa_context}
            
            Format:
            Q1: ...
            Answer: ...
            """
            qa_gen_response = llm.invoke(qa_gen_prompt)
            st.success("Yeh rahe descriptive Q&A:")
            st.write(qa_gen_response.content)

    st.markdown("---")

    # --- 4. MCQ GENERATOR SECTION ---
    st.subheader("📝 Generate Multiple Choice Questions (MCQs)")
    if st.button("🎯 Generate MCQs"):
        with st.spinner("MCQs tayaar ho rahe hain..."):
            mcq_context = "\n\n".join([doc.page_content for doc in chunks[:4]])
            
            if content_lang == "English":
                mcq_instruction = "Generate 5 Multiple Choice Questions (MCQs) with 4 options (A, B, C, D) and specify the Correct Answer based on this text in professional English."
            else:
                mcq_instruction = "Is text mein se 5 Multiple Choice Questions (MCQs) banayein. Har MCQ ke 4 options (A, B, C, D) aur neechay Correct Answer lazmi Roman Urdu mein likhein."

            mcq_prompt = f"""
            Aap ek expert quiz master hain. {mcq_instruction}
            
            Text:
            {mcq_context}
            
            Format:
            MCQ 1: [Sawal]
            A) ...
            B) ...
            C) ...
            D) ...
            Correct Answer: ...
            """
            mcq_response = llm.invoke(mcq_prompt)
            st.success("Yeh rahe MCQs:")
            st.write(mcq_response.content)

    st.markdown("---")

    # --- 5. FLASHCARDS GENERATOR SECTION ---
    st.subheader("⚡ Generate Study Flashcards")
    if st.button("🗂️ Generate Flashcards"):
        with st.spinner("Study flashcards tayaar ho rahe hain..."):
            fc_context = "\n\n".join([doc.page_content for doc in chunks[:4]])
            
            if content_lang == "English":
                fc_instruction = "Create 5 study flashcards (Front: Concept/Question, Back: Explanation/Answer) based on this text in professional English."
            else:
                fc_instruction = "Is text mein se 5 study flashcards banayein (Front: Concept ya Sawal, Back: Mukhtasir Jawab ya Explanation) Roman Urdu mein."

            fc_prompt = f"""
            Aap ek expert study coach hain. {fc_instruction}
            
            Text:
            {fc_context}
            
            Format:
            Flashcard 1:
            - **Front:** ...
            - **Back:** ...
            """
            fc_response = llm.invoke(fc_prompt)
            st.success("Yeh rahe aapke Flashcards:")
            st.write(fc_response.content)

    st.markdown("---")

    # --- 6. CHAT Q&A SECTION ---
    st.subheader("💬 Interactive Chat & Ask Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Diye gaye text ke mutaliq koi bhi sawal poochein..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Jawab tayaar ho raha hai..."):
            relevant_docs = vector_store.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            if qa_lang == "English":
                qa_instruction = "Provide a detailed and accurate answer to the user's question based on the given context in professional English. If the answer is not present in the context, clearly state: 'I could not find the answer in the text.'"
            else:
                qa_instruction = "Aap ek madadgar AI assistant hain. Diye gaye context ki buniyad par user ke sawal ka mukammal aur durust jawab Roman Urdu mein dein. Agar jawab context mein mojood na ho, toh saaf keh dein ke 'Mujhe text mein iska jawab nahi mila.'"

            prompt = f"""
            {qa_instruction}
            
            Context:
            {context}

            Sawal / Question: {user_query}
            Jawab / Answer:
            """

            response = llm.invoke(prompt)
            answer = response.content

            sources_set = set()
            for doc in relevant_docs:
                page_num = doc.metadata.get("page", None)
                if page_num is not None and page_num != 0:
                    sources_set.add(str(page_num + 1))
                else:
                    sources_set.add("Direct Text")
            
            sources_str = ", ".join(sources_set)
            final_answer = f"{answer}\n\n📌 **Source:** {sources_str}"

        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        with st.chat_message("assistant"):
            st.markdown(final_answer)

elif not api_key:
    st.warning("Bara-e-karam apni OpenRouter API Key enter karein.")
else:
    st.info("Shuru karne ke liye ya toh upar **File Upload** karein ya **Paste Text** wale box mein text paste karein.")