# src/qa.py
"""
QA Engine: LLM-agnostic question answering module.
Supports Gemini and Groq via llm_client.py
Compatible with existing ingestion.py and vectorstore.py.
Uses sentence-transformers for embeddings.
"""

from typing import List, Dict
from llm_client import get_gemini_client, get_groq_client

# ----------------------
# For local embeddings
# ----------------------
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-mpnet-base-v2")


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
    def __init__(self, llm_model="groq/compound"):
        self.client = get_groq_client()
        self.llm_model = llm_model

    # Use sentence-transformers for embeddings
    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        return [embed_model.encode(text).tolist() for text in text_list]

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
# QA Engine (Enhanced)
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

    def ask(
        self, 
        query: str, 
        top_k: int = 10, 
        per_document: bool = False
    ) -> str:
        """
        Ask a question over multiple documents.
        
        per_document=True -> answers are returned grouped by document.
        per_document=False -> answers are combined across all docs.
        """
        print(f"[QA] User query: {query}")

        query_vec = self.llm.get_embeddings([query])[0]
        retrieved = self.vs.similarity_search(query_vec, top_k=top_k)

        # Group chunks by document
        docs = {}
        for r in retrieved:
            meta = r["metadata"]
            doc_title = meta.get("title") or meta.get("source_name") or "Unknown Document"
            if doc_title not in docs:
                docs[doc_title] = []
            docs[doc_title].append(r["metadata"]["text"])

        if per_document:
            # Per-document answers
            answers = []
            for doc_title, texts in docs.items():
                context_str = "\n\n".join(texts)
                prompt = f"""Answer the following question using ONLY the context from this document.

Context:
{context_str}

Question: {query}
Answer:
"""
                ans = self.llm.generate_text(prompt)
                answers.append(f"{doc_title}:\n{ans}")
            return "\n\n".join(answers)
        else:
            # Combined answer
            all_texts = [text for texts in docs.values() for text in texts]
            context_str = "\n\n".join(all_texts)
            prompt = f"""Answer the following question using the combined context from all relevant documents.

Context:
{context_str}

Question: {query}
Answer:
"""
            return f"=== Combined Answer ===\n{self.llm.generate_text(prompt)}"


# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    from vectorstore import InMemoryVectorStore
    from ingest import extract_documents

    files = ["data/2509.08876v1.pdf",
             "data/2509.09661v1.docx"]
    chunks = extract_documents(files)
    vs = InMemoryVectorStore()

    # For Gemini (Google)
    # llm = GeminiModel()
    # qa = QAEngine(vectorstore=vs, llm=llm)
    # For Groq (local embeddings + Groq LLM)
    # llm=GroqModel() 
    llm = GroqModel()
    qa = QAEngine(vectorstore=vs, llm=llm)
    qa.add_documents(chunks)
    
    # Combined answer across all docs
    print(qa.ask("What is the main contribution of these papers?", per_document=False))

    # Per-document answers
    print(qa.ask("Summarize each paper", per_document=True))
    