import streamlit as st
import requests
import time

# Configure the page
st.set_page_config(
    page_title="IntelliDoc",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Change this to your FastAPI server URL

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "arxiv_papers" not in st.session_state:
    st.session_state.arxiv_papers = []

def upload_documents(files):
    """Upload documents to the FastAPI backend"""
    try:
        # Prepare files for upload
        file_data = []
        for file in files:
            file_data.append(("files", (file.name, file.getvalue(), file.type)))
        
        with st.spinner("Uploading documents..."):
            response = requests.post(f"{API_BASE_URL}/upload", files=file_data)
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            error_msg = response.json().get("error", "Upload failed")
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the API server. Please make sure your FastAPI server is running."
    except Exception as e:
        return False, f"Upload error: {str(e)}"

def ask_question(query: str, model: str = "groq"):
    """Ask a question to the API"""
    try:
        data = {
            "query": query,
            "model": model,
            "top_k": 10
        }
        
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_BASE_URL}/ask", data=data)
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            error_msg = response.json().get("error", "Question failed")
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the API server. Please make sure your FastAPI server is running."
    except Exception as e:
        return False, f"Error: {str(e)}"

def arxiv_search(query: str, model: str = "groq", action: str = "list", max_papers: int = 3):
    """Search ArXiv papers"""
    try:
        data = {
            "query": query,
            "model": model,
            "max_papers": max_papers,
            "action": action,
            "top_k": 10
        }
        
        with st.spinner(f"ArXiv {action}ing..."):
            response = requests.post(f"{API_BASE_URL}/arxiv_search", data=data)
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            error_msg = response.json().get("error", "ArXiv search failed")
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the API server. Please make sure your FastAPI server is running."
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def display_arxiv_papers(papers):
    """Display ArXiv papers in a nice format"""
    for i, paper in enumerate(papers, 1):
        with st.expander(f"📄 Paper {i}: {paper['title'][:100]}..."):
            st.write(f"**Authors:** {', '.join(paper['authors'])}")
            st.write(f"**Summary:** {paper['summary'][:500]}...")
            st.write(f"**PDF URL:** {paper['pdf_url']}")

# Main app
def main():
    # Header
    st.title("🤖 IntelliDoc")
    st.markdown("Ask anything. Know everything!")
    
    # Check API health
    if not check_api_health():
        st.error(f"⚠️ Cannot connect to the API server at {API_BASE_URL}")
        st.info("Please make sure your FastAPI server is running and the URL is correct.")
        st.code("uvicorn main:app --reload", language="bash")
        return
    
    st.success("✅ Connected to API server")
    
    # Sidebar for configuration and file management
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model = st.selectbox(
            "Select Model",
            options=["GROQ", "GEMINI"],
            index=0,
            help="Choose the AI model for answering questions"
        )
        
        st.divider()
        
        # ArXiv Search Section
        st.header("🔍 ArXiv Search")
        
        with st.form("arxiv_form"):
            arxiv_query = st.text_input(
                "ArXiv Search Query",
                placeholder="e.g., robotics"
            )
            
            max_papers = st.slider(
                "Max Papers",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum number of papers to find"
            )
            
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                list_papers = st.form_submit_button("📋 LIST", type="secondary")
            with action_col2:
                download_papers = st.form_submit_button("⬇️ DOWNLOAD", type="secondary")
            with action_col3:
                ingest_papers = st.form_submit_button("📚 INGEST", type="primary")
        
        # Handle ArXiv actions
        if list_papers and arxiv_query:
            success, result = arxiv_search(arxiv_query, model, "list", max_papers)
            if success:
                st.session_state.arxiv_papers = result.get("papers", [])
                st.success(f"Found {len(st.session_state.arxiv_papers)} papers")
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"🔍 Found {len(st.session_state.arxiv_papers)} ArXiv papers for: '{arxiv_query}'",
                    "timestamp": time.time()
                })
                st.rerun()
            else:
                st.error(f"ArXiv search failed: {result}")
        
        if download_papers and arxiv_query:
            success, result = arxiv_search(arxiv_query, model, "download", max_papers)
            if success:
                downloaded_files = result.get("downloaded_files", [])
                st.success(f"Downloaded {len(downloaded_files)} papers")
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"⬇️ Downloaded {len(downloaded_files)} ArXiv papers to local storage",
                    "timestamp": time.time()
                })
                st.rerun()
            else:
                st.error(f"Download failed: {result}")
        
        if ingest_papers and arxiv_query:
            success, result = arxiv_search(arxiv_query, model, "ingest", max_papers)
            if success:
                chunks_added = result.get("chunks_added", 0)
                st.success(f"Ingested papers - added {chunks_added} chunks")
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"📚 Ingested ArXiv papers and added {chunks_added} chunks to knowledge base",
                    "timestamp": time.time()
                })
                st.rerun()
            else:
                st.error(f"Ingestion failed: {result}")
        
        # Display found ArXiv papers
        if st.session_state.arxiv_papers:
            st.subheader("📄 Found ArXiv Papers")
            display_arxiv_papers(st.session_state.arxiv_papers)
        
        st.divider()
        
        # File upload section
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=['pdf', 'doc', 'docx', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload PDF, Word documents, or text files"
        )
        
        if uploaded_files:
            if st.button("Upload Documents", type="primary"):
                success, result = upload_documents(uploaded_files)
                
                if success:
                    st.success(f"✅ Successfully uploaded {len(uploaded_files)} file(s)")
                    st.info(f"Added {result.get('chunks_added', 0)} chunks to knowledge base")
                    
                    # Store uploaded files in session state
                    for file in uploaded_files:
                        if file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                            st.session_state.uploaded_files.append({
                                'name': file.name,
                                'size': file.size,
                                'type': file.type
                            })
                    
                    # Add success message to chat
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"📁 Successfully uploaded {len(uploaded_files)} document(s) and added {result.get('chunks_added', 0)} chunks to knowledge base.",
                        "timestamp": time.time()
                    })
                    
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {result}")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📄 Uploaded Documents")
            for i, file in enumerate(st.session_state.uploaded_files):
                with st.expander(f"{file['name']}", expanded=False):
                    st.write(f"**Size:** {file['size']:,} bytes")
                    st.write(f"**Type:** {file['type']}")
                    
                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state.uploaded_files.pop(i)
                        st.rerun()
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Clear ArXiv papers button
        if st.session_state.arxiv_papers and st.button("🗑️ Clear ArXiv Papers"):
            st.session_state.arxiv_papers = []
            st.rerun()
        
        # Status info
        st.subheader("📊 Status")
        st.write(f"**Documents:** {len(st.session_state.uploaded_files)}")
        st.write(f"**ArXiv Papers:** {len(st.session_state.arxiv_papers)}")
        st.write(f"**Model:** {model.upper()}")
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat Interface")
    
    with col2:
        total_sources = len(st.session_state.uploaded_files) + len(st.session_state.arxiv_papers)
        if total_sources > 0:
            st.info(f"📚 {total_sources} knowledge sources loaded")
        else:
            st.warning("📚 No knowledge sources loaded")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
                    
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
            elif message["role"] == "system":
                with st.chat_message("assistant", avatar="📁"):
                    st.success(message["content"])
            
            elif message["role"] == "error":
                with st.chat_message("assistant", avatar="❌"):
                    st.error(message["content"])
    
    # Chat input
    query = st.chat_input("Ask a question about your documents or ingested ArXiv papers...")
    
    # Handle user input
    if query:
        # Check if we have any knowledge sources
        has_sources = st.session_state.uploaded_files or st.session_state.arxiv_papers
        
        if not has_sources:
            st.error("❌ Please upload documents or search/ingest ArXiv papers first to ask questions")
            return
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "timestamp": time.time()
        })
        
        # Get AI response
        success, result = ask_question(query, model)
        
        if success:
            # Add AI response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", "No answer available"),
                "timestamp": time.time()
            })
        else:
            # Add error message to chat
            st.session_state.messages.append({
                "role": "error",
                "content": f"Error: {result}",
                "timestamp": time.time()
            })
        
        # Refresh the page to show new messages
        st.rerun()
    
    # Instructions
    if not st.session_state.messages:
        st.markdown("""
        ### 🚀 Getting Started:
        **Upload documents** in the sidebar or **search ArXiv papers** to build your knowledge base, then **ask questions** in the chat below to get intelligent AI-powered answers!.
        """)  
        

if __name__ == "__main__":
    main()