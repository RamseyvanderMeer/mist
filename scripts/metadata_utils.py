"""
Metadata truncation utilities for MIST indexing.

Provides functions to intelligently truncate metadata fields while preserving
semantic meaning for ChromaDB storage.
"""
import re
from typing import List, Dict, Any, Optional


def truncate_title(title: str, max_len: int = 200) -> str:
    """
    Truncate title intelligently to preserve meaning.
    
    - If title is a comma-separated list, keep first N items + count
    - Otherwise truncate at word boundary
    
    Args:
        title: Original title string
        max_len: Maximum length for truncated title
        
    Returns:
        Truncated title with indicator if truncated
        
    Examples:
        >>> truncate_title("A267*1V, A268*1V, A269*1V, A270*1V, A307*1B...", 200)
        "A267*1V, A268*1V, A269*1V, A270*1V, A307*1B (+42 more)"
        
        >>> truncate_title("BMW 7-Series E65/66", 200)
        "BMW 7-Series E65/66"
    """
    if not title:
        return ""
    
    if len(title) <= max_len:
        return title
    
    # Detect list patterns (part numbers, fault codes, etc.)
    if ',' in title:
        items = [item.strip() for item in title.split(',')]
        
        # Build truncated list
        result = []
        current_len = 0
        for item in items:
            # Reserve 20 chars for " (+N more)" suffix
            if current_len + len(item) + 2 > max_len - 20:
                break
            result.append(item)
            current_len += len(item) + 2
        
        remaining = len(items) - len(result)
        if remaining > 0:
            return f"{', '.join(result)} (+{remaining} more)"
        return ', '.join(result)
    
    # Regular truncation at word boundary
    truncated = title[:max_len]
    last_space = truncated.rfind(' ')
    
    # Only break at space if we keep at least 80% of max_len
    if last_space > max_len * 0.8:
        truncated = truncated[:last_space]
    
    return truncated + "..."


def extract_text_preview(text: str, max_len: int = 500) -> str:
    """
    Extract a meaningful preview from procedure text.
    
    - Removes CSS/font styling noise
    - Prioritizes problem/solution sections
    - Keeps first N chars of actual content
    
    Args:
        text: Full procedure text (XML-stripped)
        max_len: Maximum length for preview
        
    Returns:
        Cleaned text preview
        
    Examples:
        >>> extract_text_preview("@media screen { .standard_black {...", 500)
        "BMW 7-Series E65/66 rescue card information..."
    """
    if not text:
        return ""
    
    # Remove common CSS/JavaScript noise patterns
    noise_patterns = [
        r'@media\s+screen\s*\{[^}]*\}',  # @media blocks
        r'\.[a-zA-Z_][a-zA-Z0-9_-]*\s*\{[^}]*\}',  # CSS class definitions
        r'var\s+\w+\s*=\s*[^;]+;',  # var declarations
        r'function\s+\w+\s*\([^)]*\)\s*\{[^}]*\}',  # function definitions
        r'tr\s*\{[^}]*\}',  # table row CSS
        r'td\s*\{[^}]*\}',  # table cell CSS
        r'p\s*\{[^}]*\}',   # paragraph CSS
    ]
    
    cleaned = text
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Look for problem/solution keywords and prioritize that content
    problem_keywords = ['Problem:', 'Symptom:', 'Cause:', 'Condition:', 'When:', 'If:']
    for keyword in problem_keywords:
        idx = cleaned.find(keyword)
        if idx != -1 and idx < max_len:
            # Start from problem keyword if found early
            start = max(0, idx - 50)
            preview = cleaned[start:start + max_len]
            if len(cleaned) > start + max_len:
                preview += "..."
            return preview
    
    # Default: first N chars
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len] + "..."


def truncate_fault_codes(fault_codes: List[str], max_codes: int = 10) -> List[str]:
    """
    Limit fault codes to most relevant ones.
    
    - Prioritizes OBD-II codes (P, B, C, U prefixes)
    - Keeps first N codes after sorting
    
    Args:
        fault_codes: List of fault code strings
        max_codes: Maximum number of codes to keep
        
    Returns:
        Truncated list of fault codes (or ['P0000'] if empty)
        
    Examples:
        >>> truncate_fault_codes(['P0301', 'P0302', '2A87', ... 50 codes], 10)
        ['P0301', 'P0302', 'P0303', 'P0304', 'P0305', '2A87', '2A88', ...]
    """
    if not fault_codes:
        return ['P0000']  # Placeholder for ChromaDB
    
    # Remove placeholder if we have real codes
    if len(fault_codes) > 1 and 'P0000' in fault_codes:
        fault_codes = [c for c in fault_codes if c != 'P0000']
    
    if len(fault_codes) <= max_codes:
        return fault_codes
    
    # Sort: OBD-II codes first (P, B, C, U), then others
    obd_prefixes = ('P', 'B', 'C', 'U')
    sorted_codes = sorted(
        fault_codes, 
        key=lambda c: (not c.startswith(obd_prefixes), c)
    )
    
    return sorted_codes[:max_codes]


def build_metadata(
    doc: Dict[str, Any], 
    text: str,
    title_max_len: int = 200,
    text_max_len: int = 500,
    max_fault_codes: int = 10
) -> Dict[str, Any]:
    """
    Build optimized metadata for ChromaDB storage.
    
    Intelligently truncates fields to reduce metadata size while preserving
    semantic meaning for search/retrieval.
    
    Args:
        doc: Document dict with 'title', 'procedure_id', 'fault_codes', etc.
        text: Full procedure text (already XML-stripped)
        title_max_len: Maximum length for title
        text_max_len: Maximum length for text preview
        max_fault_codes: Maximum number of fault codes to keep
        
    Returns:
        Optimized metadata dict for ChromaDB
        
    Example:
        >>> doc = {
        ...     'title': 'BMW 7-Series E65/66',
        ...     'procedure_id': '2000004249159',
        ...     'fault_codes': ['P0301', 'P0302']
        ... }
        >>> meta = build_metadata(doc, "Procedure text here...")
        >>> print(meta['title'])
        'BMW 7-Series E65/66'
        >>> print(meta['text_preview'][:50])
        'Procedure text here...'
    """
    # Truncate title
    truncated_title = truncate_title(doc.get('title', ''), max_len=title_max_len)
    
    # Extract meaningful text preview
    text_preview = extract_text_preview(text, max_len=text_max_len)
    
    # Limit fault codes
    fault_codes = truncate_fault_codes(
        doc.get('fault_codes', []), 
        max_codes=max_fault_codes
    )
    
    # Build metadata (removed redundant fields)
    meta = {
        "title": truncated_title,
        "procedure_id": doc.get('procedure_id', ''),
        "text_preview": text_preview,
        "fault_codes": fault_codes,
    }
    
    return meta


def calculate_metadata_size(meta: Dict[str, Any]) -> int:
    """
    Calculate approximate metadata size in characters.
    
    Useful for debugging and monitoring truncation effectiveness.
    
    Args:
        meta: Metadata dict
        
    Returns:
        Approximate character count
    """
    return len(str(meta))


# Backwards compatibility: old field names
build_optimized_metadata = build_metadata
