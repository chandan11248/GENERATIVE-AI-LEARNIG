import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv
import re

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Video Chatbot",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_loaded' not in st.session_state:
    st.session_state.video_loaded = False
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'detected_language' not in st.session_state:
    st.session_state.detected_language = None

# Initialize components
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

embedding = get_embedding_model()
model = get_llm()
parser = StrOutputParser()

# Extract video ID from YouTube URL
def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Function to detect language and translate to English
def detect_and_translate(text):
    """Detect language and translate to English if needed"""
    
    detect_prompt = PromptTemplate(
        template="""Detect the language of the following text and respond with ONLY the language name (e.g., "English", "Hindi", "Spanish", etc.):

Text: {text}

Language:""",
        input_variables=["text"]
    )
    
    # Detect language using first 500 characters
    detect_chain = detect_prompt | model | parser
    detected_lang = detect_chain.invoke({"text": text[:500]}).strip()
    
    # If not English, translate
    if detected_lang.lower() != "english":
        translate_prompt = PromptTemplate(
            template="""Translate the following {language} text to English. Provide ONLY the translation, no explanations:

Text: {text}

English Translation:""",
            input_variables=["language", "text"]
        )
        
        translate_chain = translate_prompt | model | parser
        
        # Translate in chunks to handle long texts
        chunk_size = 3000
        translated_parts = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            translated_chunk = translate_chain.invoke({
                "language": detected_lang,
                "text": chunk
            })
            translated_parts.append(translated_chunk)
        
        translated_text = " ".join(translated_parts)
        return translated_text, detected_lang
    
    return text, "English"

# Function to load YouTube video transcript
def load_youtube_video(url):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        video_id = extract_video_id(url)
        
        if not video_id:
            st.error("Could not extract video ID from URL.")
            return None, None
        
        st.info(f"📹 Video ID: {video_id}")
        
        transcript_text = ""
        original_language = "Unknown"
        
        try:
            # New API: use fetch() method
            ytt_api = YouTubeTranscriptApi()
            transcript_data = ytt_api.fetch(video_id)
            
            # Get language info
            original_language = getattr(transcript_data, 'language', 'Auto-detected')
            
            # Extract text from transcript snippets
            if hasattr(transcript_data, 'snippets'):
                transcript_text = " ".join([snippet.text for snippet in transcript_data.snippets])
            elif isinstance(transcript_data, list):
                transcript_text = " ".join([entry.get('text', '') if isinstance(entry, dict) else str(entry) for entry in transcript_data])
            else:
                # Try to iterate directly
                transcript_text = " ".join([entry.text if hasattr(entry, 'text') else str(entry) for entry in transcript_data])
            
        except Exception as e1:
            st.warning(f"Primary method failed: {str(e1)}")
            
            # Fallback: Try listing transcripts first
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.list(video_id)
                
                # Get first available transcript
                for transcript_info in transcript_list:
                    transcript_data = ytt_api.fetch(video_id, languages=[transcript_info.language_code])
                    original_language = transcript_info.language
                    
                    if hasattr(transcript_data, 'snippets'):
                        transcript_text = " ".join([snippet.text for snippet in transcript_data.snippets])
                    else:
                        transcript_text = " ".join([entry.text if hasattr(entry, 'text') else str(entry) for entry in transcript_data])
                    break
                    
            except Exception as e2:
                st.error(f"Could not fetch transcript: {str(e2)}")
                return None, None
        
        if transcript_text:
            document = Document(
                page_content=transcript_text,
                metadata={
                    "source": url,
                    "video_id": video_id,
                    "original_language": original_language
                }
            )
            return [document], original_language
        
        st.error("Transcript is empty.")
        return None, None
            
    except ImportError:
        st.error("Please install youtube-transcript-api: pip install youtube-transcript-api")
        return None, None
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
        return None, None

# Function to create vector store
def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="youtube_transcript"
    )
    return vector_store, chunks

# Function to generate summary
def generate_summary(transcript):
    summary_prompt = PromptTemplate(
        template="""You are an expert content summarizer. Summarize the following YouTube video transcript in a clear and structured way.

Transcript:
{transcript}

Provide:
1. **Main Topic**: What is the video about?
2. **Key Points**: List 5-7 main points covered
3. **Important Takeaways**: What should viewers remember?

Summary:""",
        input_variables=["transcript"]
    )
    
    chain = summary_prompt | model | parser
    return chain.invoke({"transcript": transcript[:8000]})

# Function to answer questions
def answer_question(question, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    qa_prompt = PromptTemplate(
        template="""You are a helpful assistant answering questions about a YouTube video.
Use the following context from the video transcript to answer the question in detail.

Context:
{context}

Question: {question}

Provide a comprehensive and detailed answer. If the question asks about a specific topic, explain it thoroughly.

Answer:""",
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | model
        | parser
    )
    
    return chain.invoke(question)

# Main UI
st.markdown('<h1 class="main-header">🎬 YouTube Video Chatbot</h1>', unsafe_allow_html=True)
st.markdown("##### Summarize & Ask Questions About YouTube Videos")

# Sidebar
with st.sidebar:
    st.header("🔗 Video Input")
    youtube_url = st.text_input(
        "Paste YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    load_button = st.button("📥 Load Video", type="primary", use_container_width=True)
    
    if load_button and youtube_url:
        with st.spinner("Loading video transcript..."):
            documents, orig_lang = load_youtube_video(youtube_url)
            
            if documents:
                raw_transcript = documents[0].page_content
                st.success(f"✅ Transcript loaded!")
                st.info(f"🌐 Original language: {orig_lang}")
                
                # Detect and translate if needed
                with st.spinner("Detecting language & translating if needed..."):
                    translated_text, detected_lang = detect_and_translate(raw_transcript)
                    st.session_state.detected_language = detected_lang
                    
                    if detected_lang.lower() != "english":
                        st.success(f"✅ Translated from {detected_lang} to English")
                    else:
                        st.success("✅ Transcript is in English")
                
                st.session_state.transcript = translated_text
                
                # Create document with translated text
                translated_doc = Document(
                    page_content=translated_text,
                    metadata=documents[0].metadata
                )
                
                # Create vector store
                with st.spinner("Creating vector store..."):
                    st.session_state.vector_store, chunks = create_vector_store([translated_doc])
                    st.success(f"✅ Created {len(chunks)} chunks")
                
                st.session_state.video_loaded = True
    
    st.divider()
    
    # Manual transcript input
    st.header("📝 Or Paste Transcript")
    manual_transcript = st.text_area(
        "Paste transcript manually:",
        height=100,
        placeholder="Paste transcript here..."
    )
    
    if st.button("📥 Load Transcript", use_container_width=True):
        if manual_transcript:
            with st.spinner("Processing transcript..."):
                # Detect and translate
                translated_text, detected_lang = detect_and_translate(manual_transcript)
                st.session_state.detected_language = detected_lang
                
                if detected_lang.lower() != "english":
                    st.success(f"✅ Translated from {detected_lang} to English")
                else:
                    st.success("✅ Transcript is in English")
                
                st.session_state.transcript = translated_text
                
                document = Document(
                    page_content=translated_text,
                    metadata={"source": "manual_input"}
                )
                
                st.session_state.vector_store, chunks = create_vector_store([document])
                st.success(f"✅ Created {len(chunks)} chunks")
                st.session_state.video_loaded = True
        else:
            st.warning("Please paste a transcript first")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🔄 Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content area
if st.session_state.video_loaded:
    
    # Language info
    if st.session_state.detected_language:
        st.info(f"🌐 Detected Language: **{st.session_state.detected_language}** → Processed in **English**")
    
    # Summary Section
    st.header("📝 Video Summary")
    
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            st.session_state.summary = generate_summary(st.session_state.transcript)
    
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    
    st.divider()
    
    # Chat Section
    st.header("💬 Ask Questions")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
    
    # Question input
    question = st.chat_input("Ask a question about the video...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(question, st.session_state.vector_store)
                st.write(answer)
        
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })
    
    # Transcript preview
    with st.expander("📜 View Transcript"):
        st.text(st.session_state.transcript[:3000] + "..." if len(st.session_state.transcript) > 3000 else st.session_state.transcript)

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome!
    
    1. **Paste a YouTube URL** in the sidebar
    2. **Click "Load Video"** to fetch transcript
    3. **Get summary** and **ask questions**
    
    ---
    
    #### ✨ Features:
    - 🌐 **Auto Language Detection** - Detects transcript language
    - 🔄 **Auto Translation** - Converts non-English to English
    - 📝 **AI Summary** - Get key points instantly
    - 💬 **Q&A Chat** - Ask anything about the video
    
    #### 🎥 Try these videos:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.code("https://www.youtube.com/watch?v=aircAruvnKk")
        st.caption("Neural Networks (English)")
    with col2:
        st.code("https://www.youtube.com/watch?v=VMj-3S1tku0")
        st.caption("Git Tutorial (English)")