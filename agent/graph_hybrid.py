from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
from .dspy_signatures import Router, SQLGenerator, AnswerSynthesizer
from .rag.retrieval import SimpleRetriever
from .tools.sqlite_tool import SQLiteTool

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    question: str
    format_hint: str
    question_id: str
    classification: Optional[str]
    relevant_docs: List[Dict[str, Any]]
    sql_query: Optional[str]
    sql_results: Optional[Dict[str, Any]]
    final_answer: Optional[Any]
    explanation: Optional[str]
    citations: List[str]
    confidence: float
    repair_count: int

class HybridAgent:
    def __init__(self):
        self.retriever = SimpleRetriever()
        self.db_tool = SQLiteTool()
        self.router = Router()
        self.sql_generator = SQLGenerator()
        self.answer_synthesizer = AnswerSynthesizer()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.route_question)
        workflow.add_node("retriever", self.retrieve_docs)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("synthesize_answer", self.synthesize_answer)
        workflow.add_node("validate_output", self.validate_output)
        workflow.add_node("repair", self.repair)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Define edges
        workflow.add_edge("router", "retriever")
        workflow.add_conditional_edges(
            "retriever",
            self.decide_after_retrieval,
            {
                "sql_only": "generate_sql",
                "rag_only": "synthesize_answer",
                "hybrid": "generate_sql"
            }
        )
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_conditional_edges(
            "execute_sql",
            self.check_sql_execution,
            {
                "success": "synthesize_answer",
                "retry": "repair",
                "fail": "synthesize_answer"
            }
        )
        workflow.add_edge("synthesize_answer", "validate_output")
        workflow.add_conditional_edges(
            "validate_output",
            self.check_output_validation,
            {
                "valid": END,
                "invalid": "repair"
            }
        )
        workflow.add_edge("repair", "router")
        
        return workflow.compile()
    
    def route_question(self, state: AgentState) -> AgentState:
        """Route question to appropriate processing"""
        classification = self.router(question=state["question"])
        return {"classification": classification.classification}
    
    def retrieve_docs(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents"""
        relevant_docs = self.retriever.retrieve(state["question"])
        return {"relevant_docs": relevant_docs}
    
    def decide_after_retrieval(self, state: AgentState) -> str:
        """Decide next step after retrieval"""
        if state["classification"] == "sql":
            return "sql_only"
        elif state["classification"] == "rag":
            return "rag_only"
        else:
            return "hybrid"
    
    def generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL query"""
        schema_info = self.db_tool.get_schema()
        sql_result = self.sql_generator(
            question=state["question"],
            schema_info=schema_info,
            relevant_docs=state["relevant_docs"]
        )
        return {"sql_query": sql_result.sql_query}
    
    def execute_sql(self, state: AgentState) -> AgentState:
        """Execute SQL query"""
        if state["sql_query"]:
            results = self.db_tool.execute_query(state["sql_query"])
            return {"sql_results": results}
        return {"sql_results": None}
    
    def check_sql_execution(self, state: AgentState) -> str:
        """Check if SQL execution was successful"""
        if state["sql_results"] and state["sql_results"]["success"]:
            return "success"
        elif state.get("repair_count", 0) < 2:
            return "retry"
        else:
            return "fail"
    
    def synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesize final answer"""
        answer_result = self.answer_synthesizer(
            question=state["question"],
            sql_results=state.get("sql_results"),
            relevant_docs=state["relevant_docs"],
            format_hint=state["format_hint"]
        )
        
        # Extract citations
        citations = set()
        if state.get("sql_results") and state["sql_results"]["success"]:
            # Add table citations from SQL query
            tables_used = self._extract_tables_from_sql(state.get("sql_query", ""))
            citations.update(tables_used)
        
        # Add document citations
        for doc in state["relevant_docs"]:
            citations.add(doc['id'])
        
        return {
            "final_answer": answer_result.final_answer,
            "explanation": answer_result.explanation,
            "citations": list(citations)
        }
    
    def validate_output(self, state: AgentState) -> AgentState:
        """Validate output format and content"""
        is_valid = self._validate_answer_format(
            state["final_answer"], 
            state["format_hint"]
        )
        return {"valid": is_valid}
    
    def check_output_validation(self, state: AgentState) -> str:
        """Check output validation result"""
        if state.get("valid", False) or state.get("repair_count", 0) >= 2:
            return "valid"
        else:
            return "invalid"
    
    def repair(self, state: AgentState) -> AgentState:
        """Repair mechanism"""
        repair_count = state.get("repair_count", 0) + 1
        return {"repair_count": repair_count}
    
    def _extract_tables_from_sql(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = []
        if not sql_query:
            return tables
        
        # Simple extraction - can be improved
        sql_lower = sql_query.lower()
        for table in self.db_tool.get_table_names():
            if table.lower() in sql_lower:
                tables.append(table)
        
        return tables
    
    def _validate_answer_format(self, answer: Any, format_hint: str) -> bool:
        """Validate answer format matches the hint"""
        try:
            if format_hint == "int":
                return isinstance(answer, int)
            elif format_hint == "float":
                return isinstance(answer, (int, float))
            elif format_hint.startswith("list[{"):
                return isinstance(answer, list) and all(isinstance(item, dict) for item in answer)
            elif format_hint.startswith("{"):
                return isinstance(answer, dict)
            return True
        except:
            return False
    
    def run(self, question: str, format_hint: str, question_id: str) -> Dict[str, Any]:
        """Run the agent for a single question"""
        initial_state = {
            "messages": [],
            "question": question,
            "format_hint": format_hint,
            "question_id": question_id,
            "classification": None,
            "relevant_docs": [],
            "sql_query": None,
            "sql_results": None,
            "final_answer": None,
            "explanation": None,
            "citations": [],
            "confidence": 0.0,
            "repair_count": 0
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Calculate confidence
        confidence = self._calculate_confidence(final_state)
        
        return {
            "id": question_id,
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql_query", ""),
            "confidence": confidence,
            "explanation": final_state["explanation"],
            "citations": final_state["citations"]
        }
    
    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate confidence score"""
        confidence = 1.0
        
        # Penalize for repairs
        if state.get("repair_count", 0) > 0:
            confidence -= 0.2 * state["repair_count"]
        
        # Penalize for SQL failures
        if state.get("sql_results") and not state["sql_results"]["success"]:
            confidence -= 0.3
        
        # Boost for good document coverage
        if state["relevant_docs"] and any(doc['score'] > 0.3 for doc in state["relevant_docs"]):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))