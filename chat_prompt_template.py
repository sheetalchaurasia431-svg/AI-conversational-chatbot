from langchain_core.prompts import ChatPromptTemplate


chat_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert AI research assistant.
        Provide clear, concise, accurate, and beginner-friendly explanations.
        Use examples when they help explain the concept."""
    ),
    (
        "human",
        """Domain: {domain}
Topic: {topic}

Question:
{user_input}

Please explain the answer clearly and use a practical example if appropriate."""
    )
])


prompt = chat_template.invoke({
    "domain": "Artificial Intelligence",
    "topic": "LangChain",
    "user_input": "What is LangChain and why is it used?"
})


print(prompt)