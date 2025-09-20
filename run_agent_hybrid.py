#!/usr/bin/env python3
import json
import click
from typing import List, Dict, Any
from agent.graph_hybrid import HybridAgent
import jsonlines

@click.command()
@click.option('--batch', required=True, help='Input JSONL file with questions')
@click.option('--out', required=True, help='Output JSONL file for results')
def main(batch: str, out: str):
    """Main CLI entrypoint"""
    agent = HybridAgent()
    
    # Read input questions
    questions = []
    with jsonlines.open(batch) as reader:
        for obj in reader:
            questions.append(obj)
    
    # Process each question
    results = []
    for q in questions:
        result = agent.run(
            question=q['question'],
            format_hint=q['format_hint'],
            question_id=q['id']
        )
        results.append(result)
    
    # Write results
    with jsonlines.open(out, mode='w') as writer:
        for result in results:
            writer.write(result)
    
    print(f"Processed {len(results)} questions. Results written to {out}")

if __name__ == '__main__':
    main()