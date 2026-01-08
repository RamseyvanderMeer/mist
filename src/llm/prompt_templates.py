"""
Prompt templates for clarification and query expansion.
"""
from typing import Dict, List


def get_clarification_prompt(fault_codes: List[str], obd_data: Dict, context: str) -> Dict[str, str]:
    """
    Get clarification prompt for LLM.
    
    Args:
        fault_codes: List of fault codes
        obd_data: OBD sensor data
        context: Current context (top candidates)
    
    Returns:
        Dict with "system" and "user" messages
    """
    system_prompt = """You are a diagnostic assistant helping to clarify automotive fault diagnosis.
Analyze the fault codes and OBD data, then ask 1-3 clarifying questions
to help narrow down the diagnosis.

Focus on:
- Missing critical information
- Ambiguous symptoms
- Vehicle-specific context

Return only the questions, numbered or bulleted."""
    
    user_prompt = f"""Fault Codes: {', '.join(fault_codes)}
OBD Data: {obd_data}
Current Context: {context}

Generate clarifying questions to improve diagnosis accuracy."""
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }


def get_query_expansion_prompt(original_query: str, user_responses: List[str]) -> Dict[str, str]:
    """
    Get query expansion prompt for LLM.
    
    Args:
        original_query: Original query text
        user_responses: List of user clarification responses
    
    Returns:
        Dict with "system" and "user" messages
    """
    system_prompt = """You are a query expansion assistant for automotive diagnostics.
Expand the original query with context from user responses."""
    
    user_prompt = f"""Original Query: {original_query}
User Responses: {', '.join(user_responses)}

Expand the query with relevant context."""
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }
