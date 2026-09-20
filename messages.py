from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a helpful assistant that provides concise and accurate information."),
    HumanMessage(content="Tell me about the latest advancements in AI research."),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)