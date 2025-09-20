#!/usr/bin/env python3
import sys
sys.path.append('.')
from agent.graph_hybrid import HybridAgent

def test_agent():
    try:
        agent = HybridAgent()
        result = agent.run(
            question="According to the product policy, what is the return window (days) for unopened Beverages?",
            format_hint="int",
            question_id="test_1"
        )
        
        print("Agent test result:")
        print(f"  Final answer: {result['final_answer']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Explanation: {result['explanation']}")
        print(f"  Citations: {result['citations']}")
        
        print("Agent test: SUCCESS")
        
    except Exception as e:
        print(f"Agent test: FAILED - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent()
