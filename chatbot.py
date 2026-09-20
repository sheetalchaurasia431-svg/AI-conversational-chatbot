import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================
   MAIN APP
========================= */

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #1e293b 100%
    );

    color: #ffffff;
}


/* =========================
   GENERAL TEXT
========================= */

p {
    color: #f8fafc !important;
}

span {
    color: #f8fafc !important;
}

label {
    color: #f8fafc !important;
}


/* =========================
   HEADINGS
========================= */

h1,
h2,
h3,
h4,
h5,
h6 {
    color: #ffffff !important;
}


/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background: #0b1120;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {
    color: #ffffff !important;
}


/* =========================
   HERO HEADER
========================= */

.hero {
    background: linear-gradient(
        135deg,
        rgba(59, 130, 246, 0.20),
        rgba(139, 92, 246, 0.20)
    );

    border: 1px solid rgba(255, 255, 255, 0.12);

    border-radius: 24px;

    padding: 30px;

    margin-bottom: 25px;

    box-shadow:
        0 10px 40px rgba(0, 0, 0, 0.25);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #cbd5e1 !important;
    font-size: 16px;
}

.status {
    display: inline-flex;

    align-items: center;

    background: rgba(34, 197, 94, 0.15);

    border: 1px solid rgba(34, 197, 94, 0.30);

    color: #86efac !important;

    padding: 6px 12px;

    border-radius: 20px;

    font-size: 13px;

    margin-top: 12px;
}

.status-dot {
    width: 8px;
    height: 8px;

    background: #22c55e;

    border-radius: 50%;

    margin-right: 7px;

    box-shadow: 0 0 10px #22c55e;
}


/* =========================
   CHAT MESSAGES
========================= */

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.06);

    border: 1px solid rgba(255, 255, 255, 0.10);

    border-radius: 18px;

    padding: 14px 18px;

    margin-bottom: 12px;
}

[data-testid="stChatMessage"] p {
    color: #ffffff !important;
}


/* =========================
   CHAT INPUT
========================= */

[data-testid="stChatInput"] {
    background: #1e293b !important;

    border-radius: 18px;
}

[data-testid="stChatInput"] textarea {
    color: #ffffff !important;

    background: #1e293b !important;

    border: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
}


/* =========================
   TEXT INPUT
========================= */

.stTextInput input {
    background-color: #1e293b !important;

    color: #ffffff !important;

    border: 1px solid #475569 !important;

    border-radius: 10px;
}

.stTextInput input::placeholder {
    color: #94a3b8 !important;
}


/* =========================
   SELECT BOX
========================= */

div[data-baseweb="select"] > div {
    background-color: #1e293b !important;

    color: #ffffff !important;

    border: 1px solid #475569 !important;
}

div[data-baseweb="select"] span {
    color: #ffffff !important;
}


/* Dropdown menu */

ul[role="listbox"] {
    background-color: #1e293b !important;
}

li[role="option"] {
    color: #ffffff !important;
    background-color: #1e293b !important;
}

li[role="option"]:hover {
    background-color: #334155 !important;
}


/* =========================
   BUTTONS
========================= */

.stButton > button {
    background: #1e293b !important;

    color: #ffffff !important;

    border: 1px solid #475569 !important;

    border-radius: 12px;

    transition: 0.2s;
}

.stButton > button:hover {
    background: #334155 !important;

    color: #ffffff !important;

    border-color: #818cf8 !important;
}


/* =========================
   INFO CARDS
========================= */

.info-card {
    background: rgba(255, 255, 255, 0.06);

    border: 1px solid rgba(255, 255, 255, 0.10);

    border-radius: 15px;

    padding: 15px;

    margin-top: 15px;
}

.info-title {
    color: #ffffff !important;

    font-weight: 600;

    margin-bottom: 5px;
}

.info-text {
    color: #cbd5e1 !important;

    font-size: 13px;
}


/* =========================
   METRIC
========================= */

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}


/* =========================
   DIVIDER
========================= */

hr {
    border-color: rgba(255, 255, 255, 0.10);
}


/* =========================
   STREAMLIT CLEANUP
========================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* =========================
   MAIN CONTAINER
========================= */

.block-container {
    max-width: 1100px;

    padding-top: 2rem;

    padding-bottom: 6rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=300
)

model = ChatHuggingFace(llm=llm)


# ============================================================
# CHAT PROMPT TEMPLATE
# ============================================================

chat_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI research assistant.

Your job is to provide:

- Clear explanations
- Accurate information
- Beginner-friendly language
- Useful examples when appropriate

Use previous conversation context when answering.

Do not invent information.
"""
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    (
        "human",
        """
Domain: {domain}

Topic: {topic}

Question:
{user_input}
"""
    )
])


# ============================================================
# LOAD CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

    try:

        with open(
            "chat_history.txt",
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if line.startswith("Human:"):

                    message = line.replace(
                        "Human:",
                        "",
                        1
                    ).strip()

                    st.session_state.chat_history.append(
                        HumanMessage(
                            content=message
                        )
                    )

                elif line.startswith("AI:"):

                    message = line.replace(
                        "AI:",
                        "",
                        1
                    ).strip()

                    st.session_state.chat_history.append(
                        AIMessage(
                            content=message
                        )
                    )

    except FileNotFoundError:

        pass


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            ⚙️ Assistant Settings
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-description">
            Customize your AI research assistant.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 Research Domain")

    domain = st.selectbox(
        "Select Domain",
        [
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Data Science",
            "Computer Science",
            "Natural Language Processing"
        ],
        label_visibility="collapsed"
    )

    st.markdown("### 📚 Topic")

    topic = st.text_input(
        "Enter Topic",
        value="LangChain",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📊 Conversation")

    message_count = len(
        st.session_state.chat_history
    )

    st.metric(
        "Messages",
        message_count
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        with open(
            "chat_history.txt",
            "w",
            encoding="utf-8"
        ):
            pass

        st.rerun()

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                🤖 Model
            </div>

            <div class="info-text">
                Llama 3.1 8B Instruct
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                🔗 Technology
            </div>

            <div class="info-text">
                LangChain + Hugging Face + Streamlit
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🤖 AI Research Assistant
        </div>

        <div class="hero-subtitle">
            Ask questions, explore concepts, and continue
            your conversation with your AI research assistant.
        </div>

        <div class="status">
            <span class="status-dot"></span>
            AI Assistant Online
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.chat_history) == 0:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                👋 Welcome!
            </div>

            <div class="info-text">
                Ask me anything about AI, Machine Learning,
                LangChain, Data Science, or Computer Science.
                Your conversation will be saved automatically.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    if isinstance(
        message,
        HumanMessage
    ):

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.write(
                message.content
            )

    elif isinstance(
        message,
        AIMessage
    ):

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            st.write(
                message.content
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask your research question..."
)


# ============================================================
# GENERATE RESPONSE
# ============================================================

if user_input:

    # Display user message
    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.write(
            user_input
        )

    # Create prompt
    prompt = chat_template.invoke({

        "domain": domain,

        "topic": topic,

        "chat_history": st.session_state.chat_history,

        "user_input": user_input

    })

    # Generate response
    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "🧠 Thinking..."
        ):

            result = model.invoke(
                prompt
            )

        st.write(
            result.content
        )

    # Save to session history
    st.session_state.chat_history.append(
        HumanMessage(
            content=user_input
        )
    )

    st.session_state.chat_history.append(
        AIMessage(
            content=result.content
        )
    )

    # Save to text file
    with open(
        "chat_history.txt",
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Human: {user_input}\n"
        )

        file.write(
            f"AI: {result.content}\n"
        )