from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough

class QAChain:
    def __init__(self, vector_db):
        """Initialize QA chain with vector database"""
        self.vector_db = vector_db
        self.retriever = vector_db.get_retriever()
        self.last_retrieved_docs = []
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.3
        )
        
        # Define prompt template
        template = """You are an expert at answering questions based on provided documents.
        
        Context: {context}
        
        Question: {question}
        
        Answer the question truthfully and concisely. If you don't know, say you don't know."""
        
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
    
    def ask_question(self, question):
        """Ask a question and return the answer"""
        self.last_retrieved_docs = self.retriever.invoke(question)
        return self.chain.invoke(question)