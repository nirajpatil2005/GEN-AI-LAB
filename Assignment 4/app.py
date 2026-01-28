import os
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import traceback

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Page Configuration
st.set_page_config(
    page_title="VitalBot - Your AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Custom CSS for "Professional & Visually Appealing" UI
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient */
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* Chat Bubbles */
    .user-message {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 0 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .ai-message {
        background-color: #ffffff;
        color: #333333;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 15px 0;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Input Box */
    .stChatInput {
        border-radius: 20px !important;
    }

    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid #eee;
    }

</style>
""", unsafe_allow_html=True)

# 4. Header Section
st.markdown("""
<div class="header-container">
    <div class="header-title">🩺 VitalBot</div>
    <div class="header-subtitle">Intelligent Health & Wellness Assistant powered by Groq LLaMA-3</div>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
    st.title("Settings")
    
    model_name = st.selectbox(
        "Choose Model",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
        help="Select the AI model architecture to use for generation."
    )
    
    st.markdown("---")
    st.info("💡 **Tip**: Be specific with your symptoms for better advice.")
    st.warning("⚠️ **Disclaimer**: This AI is not a doctor. In emergencies, call standard emergency services immediately.")
    
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# 6. Graph & Logic Definitions
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "your_groq_api_key" in api_key:
        st.error("🚨 **Error**: GROQ_API_KEY is missing in your `.env` file.")
        st.stop()
    
    return ChatGroq(
        model_name=model_name,
        temperature=0.3, # Lower temperature for more factual responses
        groq_api_key=api_key
    )

def chatbot_node(state: State):
    llm = get_llm()
    
    system_prompt = SystemMessage(content="""
    You are VitalBot, a highly advanced and empathetic health assistant.
    
    CORE RESPONSIBILITIES:
    1. Analyze the user's health related queries with precision.
    2. Provide practical, wellness-focused advice.
    3. Use formatting (bullet points, **bold** text) to make answers easy to read.
    
    SAFETY PROTOCOLS:
    - If symptoms sound severe (chest pain, difficulty breathing, high fever), IMMEDIATELY advise seeing a doctor.
    - Do NOT prescribe drugs.
    - Do NOT diagnose specific diseases definitively.
    
    TONE:
    - Professional, warm, and reassuring.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Graph Construction
try:
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot_node)
    
    # Updated Entry Point Logic for compatibility
    workflow.set_entry_point("chatbot")
    
    workflow.add_edge("chatbot", END)
    app = workflow.compile()
except Exception as e:
    st.error(f"Failed to compile graph: {e}")
    st.code(traceback.format_exc())
    st.stop()

# 7. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Message History
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(message.content)

# Input
user_input = st.chat_input("Type your health question here...", key="chat_input")

if user_input:
    # Render user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Add to state
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Process with Spinner
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Analyzing your query..."):
            try:
                # Invoke Graph
                inputs = {"messages": st.session_state.messages}
                result = app.invoke(inputs)
                
                ai_msg = result["messages"][-1]
                st.markdown(ai_msg.content)
                
                # Append to state
                st.session_state.messages.append(ai_msg)
                
            except KeyError as e:
                # Specific handling for the 'start' error if it persists
                if 'start' in str(e):
                    st.error("Internal Graph Error (KeyError: 'start'). This is likely a version mismatch. Please ensure `langgraph` is updated.")
                else:
                    st.error(f"KeyError: {e}")
                st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.code(traceback.format_exc())
