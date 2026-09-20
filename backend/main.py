from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Research Assistant",
    description="AI chatbot using FastAPI, LangChain and Hugging Face",
    version="1.0"
)


# ============================================================
# SERVE FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=300
)

model = ChatHuggingFace(
    llm=llm
)


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
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str

    domain: str = "Artificial Intelligence"

    topic: str = "LangChain"


# ============================================================
# READ CHAT HISTORY
# ============================================================

def load_chat_history():

    chat_history = []

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

                    chat_history.append(
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

                    chat_history.append(
                        AIMessage(
                            content=message
                        )
                    )

    except FileNotFoundError:

        pass

    return chat_history


# ============================================================
# SAVE CHAT HISTORY
# ============================================================

def save_chat_message(
    user_message,
    ai_message
):

    with open(
        "chat_history.txt",
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Human: {user_message}\n"
        )

        file.write(
            f"AI: {ai_message}\n"
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        "frontend/index.html"
    )


# ============================================================
# CHAT API
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    # Load previous conversation
    chat_history = load_chat_history()

    # Create prompt
    prompt = chat_template.invoke({

        "domain": request.domain,

        "topic": request.topic,

        "chat_history": chat_history,

        "user_input": request.message

    })

    # Generate AI response
    result = model.invoke(prompt)

    # Save conversation
    save_chat_message(
        request.message,
        result.content
    )

    # Send response to frontend
    return {
        "response": result.content
    }


# ============================================================
# CLEAR CHAT HISTORY
# ============================================================

@app.delete("/clear-history")
def clear_history():

    with open(
        "chat_history.txt",
        "w",
        encoding="utf-8"
    ):
        pass

    return {
        "message": "Chat history cleared successfully"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "online",
        "message": "AI Research Assistant is running"
    }