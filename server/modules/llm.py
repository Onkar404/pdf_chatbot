from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an intelligent and helpful assistant designed to answer user questions based solely on the provided document content.

Please ensure your answers are accurate, concise, and strictly grounded in the context below helpful responses based **only on the provided context**.

---

🔍 **Context**:
{context}

🙋‍♂️ **User Question**:
{question}

---

💬 **Answer**:
- Use clear, professional language.
- If the answer is found in the context, answer confidently and clearly.
- If the answer is not available in the provided content, say:
  "I'm sorry, but I couldn't find relevant information in the provided document."
- Do NOT guess or fabricate information not present in the context.
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
