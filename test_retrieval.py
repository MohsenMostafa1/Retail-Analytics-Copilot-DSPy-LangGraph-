#!/usr/bin/env python3
import sys
sys.path.append('.')
from agent.rag.retrieval import SimpleRetriever

def test_retrieval():
    try:
        retriever = SimpleRetriever()
        results = retriever.retrieve("beverages return policy", top_k=2)
        
        print("Retrieval test results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['id']} (score: {result['score']:.3f})")
            print(f"     Content: {result['content'][:100]}...")
        
        print("Retrieval test: SUCCESS")
        
    except Exception as e:
        print(f"Retrieval test: FAILED - {e}")

if __name__ == "__main__":
    test_retrieval()
