#!/usr/bin/env python3
import sys
sys.path.append('.')
from agent.dspy_signatures import Router

def test_dspy():
    try:
        router = Router()
        result = router.forward("What is the return policy for beverages?")
        print(f"DSPy test - Classification: {result.classification}")
        print("DSPy test: SUCCESS")
        
    except Exception as e:
        print(f"DSPy test: FAILED - {e}")
        print("This might be expected if Ollama is not running")

if __name__ == "__main__":
    test_dspy()
