from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


chat_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert AI research assistant.
Provide clear, concise, accurate, and beginner-friendly explanations."""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        """Domain: {domain}
Topic: {topic}

Question:
{user_input}"""
    )
])


# Read chat history from file
chat_history = []

try:
    with open("chat_history.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith("Human:"):
            chat_history.append(
                HumanMessage(content=line.replace("Human:", "").strip())
            )

        elif line.startswith("AI:"):
            chat_history.append(
                AIMessage(content=line.replace("AI:", "").strip())
            )

except FileNotFoundError:
    pass


prompt = chat_template.invoke({
    "domain": "Artificial Intelligence",
    "topic": "LangChain",
    "chat_history": chat_history,
    "user_input": "What is LangChain?"
})

print(prompt)