"""
Query Expansion Module for MIST

This module expands symptom-based queries into solution-aware queries
using LLM-based query reformulation. This bridges the lexical gap between
how users describe problems (symptoms) and how repair guides describe fixes.
"""

import os
import logging
from typing import List, Dict, Optional
import requests

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands symptom queries into solution-aware queries using LLM.
    
    This helps bridge the gap between:
    - Symptom language: "rough idle", "check engine light"
    - Solution language: "clean throttle body", "replace oxygen sensor"
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "openai/gpt-4o-mini",
        provider: str = "openrouter"
    ):
        """
        Initialize query expander.
        
        Args:
            api_key: API key for LLM provider
            model: Model to use for expansion
            provider: Provider name (openrouter, openai, etc.)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.provider = provider
        
        if provider == "openrouter":
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            self.api_url = "https://api.openai.com/v1/chat/completions"
        
        logger.info(f"Initialized QueryExpander: model={model}, provider={provider}")
    
    def expand_query(
        self,
        fault_codes: List[str],
        symptoms: str,
        max_expansions: int = 3
    ) -> List[str]:
        """
        Expand a symptom query into solution-aware queries.
        
        Args:
            fault_codes: List of fault codes
            symptoms: Symptom description
            max_expansions: Number of expansion variations to generate
        
        Returns:
            List of expanded queries (including original)
        """
        if not symptoms:
            # No symptoms to expand, return fault codes only
            return [f"Fault codes: {', '.join(fault_codes)}"]
        
        # Build the prompt
        fault_code_str = ', '.join(fault_codes)
        
        prompt = f"""Given the following automotive fault codes and symptoms, generate {max_expansions} different search queries that describe what repair actions might fix this problem.

Fault codes: {fault_code_str}
Symptoms: {symptoms}

Generate queries that:
1. Use repair/action language (e.g., "replace", "clean", "check", "inspect")
2. Include specific components that might be involved
3. Vary the specificity (one general, one specific, one with common fixes)

Return only the queries, one per line, no numbering, no extra text."""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an automotive repair expert. Convert symptom descriptions into repair action queries."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            expansion_text = data["choices"][0]["message"]["content"].strip()
            
            # Parse expansions
            expansions = [line.strip() for line in expansion_text.split('\n') if line.strip()]
            
            # Add original symptom query
            original = f"Fault codes: {fault_code_str}. Problem: {symptoms}"
            
            # Combine: original + expansions (limited to max_expansions)
            all_queries = [original] + expansions[:max_expansions]
            
            logger.debug(f"Generated {len(all_queries)} query variations")
            return all_queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            # Fallback to original query
            return [f"Fault codes: {fault_code_str}. Problem: {symptoms}"]
    
    def expand_query_batch(
        self,
        queries: List[Dict],
        max_expansions: int = 3
    ) -> List[Dict]:
        """
        Expand multiple queries in batch.
        
        Args:
            queries: List of dicts with 'fault_codes' and 'symptoms'
            max_expansions: Number of expansions per query
        
        Returns:
            List of dicts with added 'expanded_queries' field
        """
        results = []
        for query in queries:
            expanded = self.expand_query(
                query.get('fault_codes', []),
                query.get('symptoms', ''),
                max_expansions
            )
            results.append({
                **query,
                'expanded_queries': expanded,
                'original_query': f"Fault codes: {', '.join(query.get('fault_codes', []))}. Problem: {query.get('symptoms', '')}"
            })
        return results


class HybridRetriever:
    """
    Hybrid retriever that combines symptom-based and solution-based retrieval.
    """
    
    def __init__(
        self,
        encoder,
        query_expander: QueryExpander,
        vector_store=None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            encoder: Embedding encoder (e.g., OpenRouterEncoder)
            query_expander: QueryExpander instance
            vector_store: Optional vector store for retrieval
        """
        self.encoder = encoder
        self.query_expander = query_expander
        self.vector_store = vector_store
    
    def retrieve_with_expansion(
        self,
        fault_codes: List[str],
        symptoms: str,
        top_k: int = 10,
        expansion_weight: float = 0.3
    ) -> List[Dict]:
        """
        Retrieve using both original and expanded queries.
        
        Args:
            fault_codes: List of fault codes
            symptoms: Symptom description
            top_k: Number of results to return
            expansion_weight: Weight for expanded query results (0-1)
        
        Returns:
            Merged and ranked results
        """
        # Get expanded queries
        expanded_queries = self.query_expander.expand_query(fault_codes, symptoms)
        
        logger.info(f"Retrieving with {len(expanded_queries)} query variations")
        
        # Encode all queries
        query_embeddings = self.encoder.encode(expanded_queries)
        
        # For now, just return the expanded queries for testing
        # Full implementation would retrieve from vector store and merge
        results = []
        for i, (query, embedding) in enumerate(zip(expanded_queries, query_embeddings)):
            results.append({
                'query': query,
                'query_type': 'original' if i == 0 else 'expanded',
                'embedding': embedding.tolist()
            })
        
        return results


# Convenience functions
def create_query_expander(
    api_key: str = None,
    model: str = "openai/gpt-4o-mini"
) -> QueryExpander:
    """Create a query expander with OpenRouter."""
    return QueryExpander(api_key=api_key, model=model, provider="openrouter")


if __name__ == "__main__":
    # Test the query expander
    import sys
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    expander = create_query_expander(api_key=api_key)
    
    # Test expansion
    fault_codes = ["P0171", "P0174"]
    symptoms = "rough idle, check engine light on, poor fuel economy"
    
    print(f"Original: Fault codes: {', '.join(fault_codes)}. Problem: {symptoms}")
    print("\nExpanded queries:")
    
    expanded = expander.expand_query(fault_codes, symptoms, max_expansions=3)
    for i, query in enumerate(expanded):
        print(f"  {i}: {query}")
