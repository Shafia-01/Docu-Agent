# src/qa.py
"""
QA Engine: LLM-agnostic question answering module.
Supports Gemini and Groq via llm_client.py
Compatible with existing ingestion.py and vectorstore.py.
"""

from typing import List, Dict
from src.llm_client import get_gemini_client, get_groq_client


# ----------------------
# LLM Interface
# ----------------------
class LLMInterface:
    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError


# ----------------------
# Gemini Wrapper
# ----------------------
class GeminiModel(LLMInterface):
    def __init__(self, embedding_model="models/text-embedding-004", llm_model="gemini-1.5-flash"):
        self.genai = get_gemini_client()
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        embeddings = []
        for text in text_list:
            resp = self.genai.embed_content(model=self.embedding_model, content=text)
            embeddings.append(resp["embedding"])
        return embeddings

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        model = self.genai.GenerativeModel(self.llm_model)
        chat = model.start_chat(history=[])
        resp = chat.send_message(
            f"{system_prompt or 'You are a helpful assistant.'}\n\n{prompt}"
        )
        return resp.text


# ----------------------
# Groq Wrapper
# ----------------------
class GroqModel(LLMInterface):
    def __init__(self, embedding_model="groq-embed-001", llm_model="mixtral-8x7b-32768"):
        self.client = get_groq_client()
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.embedding_model, input=text_list)
        return [d.embedding for d in resp.data]

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()


# ----------------------
# QA Engine
# ----------------------
class QAEngine:
    def __init__(self, vectorstore, llm: LLMInterface):
        self.vs = vectorstore
        self.llm = llm

    def add_documents(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        embeddings = self.llm.get_embeddings(texts)
        for c, vec in zip(chunks, embeddings):
            self.vs.add(vec, c, id=c["id"])
        print(f"[QA] Added {len(chunks)} chunks to vectorstore.")

    def ask(self, query: str, top_k: int = 5) -> str:
        print(f"[QA] User query: {query}")

        query_vec = self.llm.get_embeddings([query])[0]
        retrieved = self.vs.similarity_search(query_vec, top_k=top_k)
        contexts = [r["metadata"]["text"] for r in retrieved]

        context_str = "\n\n".join(contexts)
        prompt = f"""Answer the following question using ONLY the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context_str}

Question: {query}
Answer:
"""
        return self.llm.generate_text(prompt)


# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    from src.vectorstore import InMemoryVectorStore

    # Choose provider
    # llm = GeminiModel()
    # llm = GroqModel()

    vs = InMemoryVectorStore()
    # qa = QAEngine(vectorstore=vs, llm=llm)
    # qa.add_documents(chunks)
    # print(qa.ask("Summarize the document"))
