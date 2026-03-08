from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
llm_client = ChatOpenAI(
openai_api_key=os.getenv("OPENAI_API_KEY"),
model="gpt-4.1-mini",
base_url="https://models.github.ai/inference")
print(llm_client.invoke("Hello, how are you?"))