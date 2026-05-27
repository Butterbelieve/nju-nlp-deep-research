"""
Query Expander — Programmatic keyword extraction and diverse query generation.

Generates multiple search queries from a question without LLM involvement,
improving BM25 search coverage for the Deep Research Agent.
"""

import re
from typing import Any, Dict, List

from .browsecomp_searcher import BrowseCompBM25Searcher
from .tools import snippetize


# Common English stop words to filter out
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "up", "its",
    "it", "he", "she", "they", "we", "you", "i", "me", "my", "his", "her",
    "their", "our", "your", "this", "that", "these", "those", "which", "who",
    "whom", "what", "whose", "also", "any",
})


def extract_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from a question, removing stop words."""
    # Extract quoted phrases first (often the most important clues)
    quoted = re.findall(r'"([^"]+)"', question)

    # Extract capitalized phrases (names, places, titles)
    capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)

    # Extract numbers and date-like patterns
    numbers = re.findall(r"\b(\d{1,4}s?)\b", question)

    # Extract hyphenated terms
    hyphenated = re.findall(r"\b(\w+-\w+(?:-\w+)*)\b", question)

    # Extract all remaining words, filter stop words and short words
    all_words = re.findall(r"\b([a-zA-Z]{3,})\b", question)
    content_words = [w for w in all_words if w.lower() not in STOP_WORDS]

    # Build keyword list with priority: quoted > capitalized > numbers > hyphenated > content
    keywords: List[str] = []
    seen: set = set()

    for phrase in quoted:
        if phrase.lower() not in seen:
            keywords.append(phrase)
            seen.add(phrase.lower())

    for phrase in capitalized:
        if phrase.lower() not in seen:
            keywords.append(phrase)
            seen.add(phrase.lower())

    for num in numbers:
        if num not in seen:
            keywords.append(num)
            seen.add(num)

    for term in hyphenated:
        if term.lower() not in seen:
            keywords.append(term)
            seen.add(term.lower())

    for word in content_words:
        if word.lower() not in seen:
            keywords.append(word)
            seen.add(word.lower())

    return keywords


def generate_diverse_queries(question: str, n: int = 5) -> List[str]:
    """Generate multiple diverse search queries from a question."""
    keywords = extract_keywords(question)

    if not keywords:
        # Fallback: use the full question with common words removed
        return [question]

    queries: List[str] = []
    seen_queries: set = set()

    def _add(q: str) -> None:
        q = q.strip()
        if q and q not in seen_queries and len(q) > 2:
            queries.append(q)
            seen_queries.add(q)

    # Query 1: All keywords joined
    if len(keywords) >= 2:
        _add(" ".join(keywords[:8]))

    # Query 2: Most distinctive keywords only (2-3 keywords)
    if len(keywords) >= 3:
        _add(" ".join(keywords[:3]))

    # Query 3: Different subset (middle + end keywords)
    if len(keywords) >= 5:
        mid = len(keywords) // 2
        _add(" ".join(keywords[mid:mid + 4]))

    # Query 4: Last few keywords (often the most specific)
    if len(keywords) >= 4:
        _add(" ".join(keywords[-3:]))

    # Query 5: First keyword + last keyword (broad but distinct)
    if len(keywords) >= 2:
        _add(f"{keywords[0]} {keywords[-1]}")

    # Query 6: Original question stripped of common words
    words = question.split()
    stripped = " ".join(w for w in words if w.lower() not in STOP_WORDS and len(w) > 2)
    _add(stripped)

    # If we still have fewer than n queries, add more variations
    if len(queries) < n and len(keywords) >= 4:
        # Pair combinations
        for i in range(0, min(len(keywords) - 1, 3)):
            _add(f"{keywords[i]} {keywords[i + 1]}")

    return queries[:n]


def batch_search(
    searcher: BrowseCompBM25Searcher,
    queries: List[str],
    top_k: int = 10,
    snippet_max_chars: int = 1200,
    max_total: int = 30,
) -> List[Dict[str, Any]]:
    """Execute multiple queries, merge and deduplicate results."""
    seen_docids: set = set()
    results: List[Dict[str, Any]] = []

    for query in queries:
        docs = searcher.search(query, k=top_k)
        for doc in docs:
            if doc["docid"] not in seen_docids:
                seen_docids.add(doc["docid"])
                results.append({
                    "docid": doc["docid"],
                    "score": doc["score"],
                    "snippet": snippetize(doc["text"], snippet_max_chars),
                    "url": doc.get("url", ""),
                })

    # Sort by BM25 score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_total]
