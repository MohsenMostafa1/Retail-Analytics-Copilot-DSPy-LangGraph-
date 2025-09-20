import os
import re
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRetriever:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
    def load_documents(self):
        """Load and chunk documents"""
        self.chunks = []
        chunk_id = 0
        
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple paragraph-based chunking
                paragraphs = re.split(r'\n\s*\n', content)
                for i, para in enumerate(paragraphs):
                    if para.strip():
                        self.chunks.append({
                            'id': f"{filename.replace('.md', '')}::chunk{i}",
                            'content': para.strip(),
                            'source': filename,
                            'chunk_index': i
                        })
                        chunk_id += 1
        
        # Create TF-IDF matrix
        texts = [chunk['content'] for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks"""
        if not self.chunks:
            self.load_documents()
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(similarities[idx])
            results.append(chunk)
        
        return results