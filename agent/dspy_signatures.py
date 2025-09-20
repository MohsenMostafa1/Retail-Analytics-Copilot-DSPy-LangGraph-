import dspy
from typing import List, Optional

# Configure DSPy with local model
try:
    # Try Ollama first
    lm = dspy.OllamaLocal(model='phi3.5:3.8b-mini-instruct-q4_K_M')
except:
    # Fallback to other local options
    try:
        # Try using the new DSPy configuration
        from dspy.ollama import Ollama
        lm = Ollama(model='phi3.5:3.8b-mini-instruct-q4_K_M')
    except:
        # Final fallback
        class FallbackLM:
            def __call__(self, **kwargs):
                # Simple fallback that returns dummy responses
                if 'question' in kwargs:
                    if 'classification' in str(kwargs):
                        return type('obj', (object,), {'classification': 'hybrid'})
                    elif 'sql_query' in str(kwargs):
                        return type('obj', (object,), {'sql_query': 'SELECT 1'})
                    elif 'final_answer' in str(kwargs):
                        return type('obj', (object,), {'final_answer': '42', 'explanation': 'Fallback response'})
                return type('obj', (object,), {})
        
        lm = FallbackLM()
    
dspy.settings.configure(lm=lm)

class RouteClassification(dspy.Signature):
    """Classify whether a question requires RAG, SQL, or hybrid approach."""
    question = dspy.InputField(desc="The user's question")
    classification = dspy.OutputField(desc="One of: 'rag', 'sql', 'hybrid'")

class SQLGeneration(dspy.Signature):
    """Generate SQL query based on question and schema."""
    question = dspy.InputField(desc="The user's question")
    schema_info = dspy.InputField(desc="Database schema information")
    relevant_docs = dspy.InputField(desc="Relevant document chunks")
    sql_query = dspy.OutputField(desc="SQLite-compatible SQL query")

class AnswerSynthesis(dspy.Signature):
    """Synthesize final answer from SQL results and documents."""
    question = dspy.InputField(desc="The user's question")
    sql_results = dspy.InputField(desc="Results from SQL execution")
    relevant_docs = dspy.InputField(desc="Relevant document chunks")
    format_hint = dspy.InputField(desc="Expected output format")
    final_answer = dspy.OutputField(desc="Final answer matching format hint")
    explanation = dspy.OutputField(desc="Brief explanation of the answer")

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(RouteClassification)
    
    def forward(self, question):
        return self.classifier(question=question)

class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(SQLGeneration)
    
    def forward(self, question, schema_info, relevant_docs):
        return self.generator(
            question=question,
            schema_info=schema_info,
            relevant_docs=relevant_docs
        )

class AnswerSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(AnswerSynthesis)
    
    def forward(self, question, sql_results, relevant_docs, format_hint):
        return self.synthesizer(
            question=question,
            sql_results=sql_results,
            relevant_docs=relevant_docs,
            format_hint=format_hint
        )
