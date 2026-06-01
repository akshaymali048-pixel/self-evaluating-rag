RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided context. "
    "If the context does not contain enough information, say you do not know."
)

RAG_ANSWER_PROMPT = """Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""
