from src.ingest import extract_documents
from src.vectorstore import InMemoryVectorStore
from src.qa import QAEngine, GeminiModel, GroqModel

# 1. Pick your test docs
files = ["data/2509.08876v1.pdf"]   # replace with your own PDF/DOCX

# 2. Ingest docs -> get chunks
print("[TEST] Ingesting documents...")
chunks = extract_documents(files)
print(f"[TEST] Extracted {len(chunks)} chunks")

# 3. Init vectorstore + LLM (choose one)
vs = InMemoryVectorStore()
llm = GeminiModel()   # or GroqModel()

# 4. Init QA engine
qa = QAEngine(vectorstore=vs, llm=llm)

# 5. Add chunks to vectorstore
print("[TEST] Adding documents to vectorstore...")
qa.add_documents(chunks)

# 6. Ask a sample question
question = "What is the main contribution of this paper?"
print(f"[TEST] Asking: {question}")
answer = qa.ask(question)

# 7. Print answer
print("\n=== Answer ===")
print(answer)
