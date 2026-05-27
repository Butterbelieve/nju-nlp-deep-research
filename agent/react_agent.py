"""
Deep Research Agent V4 — Batch Search + ReAct Deep Dive.

Phase 1: Programmatic batch search (no LLM) -> broad coverage
Phase 2: ReAct loop with pre-searched context -> targeted follow-up

"""

import json
from typing import Any, Dict, List

from .browsecomp_searcher import BrowseCompBM25Searcher
from .query_expander import batch_search, generate_diverse_queries
from .tools import format_rag_context, get_agent_tool_specs_and_registry
from .vllm_client import VLLMClient


# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Deep Research Agent. Some documents have already been pre-searched for you. Review them carefully first.

## Available Tools
- search(query): Search the document collection using BM25 keyword matching. Returns top results with document IDs, scores, and snippets.
- get_document(docid): Retrieve the full text of a specific document by its ID.

## CRITICAL: BM25 Search Strategy
BM25 is a keyword-based search engine. It matches the words in your query against documents.

- Do NOT search with the full question. Extract the most distinctive keywords and search with those.
- If pre-searched documents don't contain the answer, search with DIFFERENT keywords.
- Try synonyms, related terms, or search for specific entities separately.
- Search from multiple angles.

## Rules
- Check the pre-searched documents carefully before searching again.
- Each search must use DIFFERENT keywords — do not repeat queries.
- You MUST provide your best guess even if uncertain. NEVER say "cannot be determined" or "information not available". Use whatever evidence you found to make your best educated guess.
- Focus on finding EXACT facts (names, dates, numbers, titles), not vague descriptions.

## Output Format
When you have enough evidence, output your final answer in EXACTLY this format:
Explanation: <brief explanation citing the evidence you found>
Exact Answer: <your precise answer>

While searching, write your reasoning and call tools. Do NOT output the final answer format until you have sufficient evidence."""

FORCE_ANSWER_PROMPT = (
    "You must now provide your final answer based on ALL the evidence you have gathered. "
    "You MUST give your best guess — do NOT say 'cannot be determined' or 'information not available'. "
    "Use the format:\nExplanation: <your reasoning based on evidence>\nExact Answer: <your best guess>"
)

REPHRASE_PROMPT = (
    "You already searched for very similar keywords and did not find useful results. "
    "You MUST try completely different keywords now — use synonyms, search for a different entity, "
    "or approach the question from a completely different angle."
)


# ── Context Management ─────────────────────────────────────────


def _extract_rounds(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split assistant/tool messages into rounds, each starting with an assistant message."""
    rounds: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "assistant":
            if current:
                rounds.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        rounds.append(current)
    return rounds


def _extract_previous_search_queries(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract all search queries from previous rounds."""
    queries = []
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                if name == "search":
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}
                    q = args.get("query", "")
                    if q:
                        queries.append(q.lower().strip())
    return queries


def _is_similar_query(new_query: str, previous_queries: List[str], threshold: float = 0.7) -> bool:
    """Check if new_query is too similar to any previous query (word overlap ratio)."""
    new_words = set(new_query.lower().split())
    if not new_words:
        return False
    for prev in previous_queries:
        prev_words = set(prev.lower().split())
        if not prev_words:
            continue
        overlap = len(new_words & prev_words) / max(len(new_words), len(prev_words))
        if overlap > threshold:
            return True
    return False


def _summarize_round(round_msgs: List[Dict[str, Any]]) -> str:
    """Compress one round (assistant + tool results) into a one-line summary."""
    queries = []
    docids_found = []
    key_snippets = []

    for msg in round_msgs:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                if name == "search":
                    queries.append(args.get("query", "?"))
                elif name == "get_document":
                    queries.append(f"get_doc({args.get('docid', '?')})")

        if msg["role"] == "tool":
            content = msg.get("content", "")
            try:
                parsed = json.loads(content) if isinstance(content, str) else content
                if isinstance(parsed, list):
                    for item in parsed[:3]:
                        if isinstance(item, dict):
                            docids_found.append(item.get("docid", ""))
                            snippet = item.get("snippet", "")[:120]
                            if snippet:
                                key_snippets.append(snippet)
                elif isinstance(parsed, dict) and "text" in parsed:
                    key_snippets.append(parsed["text"][:120])
            except (json.JSONDecodeError, TypeError):
                pass

    parts = []
    if queries:
        parts.append(f"Searched: {'; '.join(queries)}")
    if docids_found:
        parts.append(f"Found docs: {', '.join(docids_found[:5])}")
    if key_snippets:
        parts.append(f"Key info: {'; '.join(key_snippets[:2])}")
    return " | ".join(parts) if parts else "(no results)"


def manage_context(
    messages: List[Dict[str, Any]],
    max_recent_rounds: int = 4,
) -> List[Dict[str, Any]]:
    """Keep system + user + last N rounds intact, compress older rounds into a summary."""
    system_msg = messages[0]
    user_msg = messages[1]

    rounds = _extract_rounds(messages[2:])

    if len(rounds) <= max_recent_rounds:
        return messages

    old_rounds = rounds[:-max_recent_rounds]
    recent_rounds = rounds[-max_recent_rounds:]

    summary_lines = ["[Previous search history (summarized)]:"]
    for i, r in enumerate(old_rounds, 1):
        summary_lines.append(f"  Round {i}: {_summarize_round(r)}")

    summary_text = "\n".join(summary_lines)

    new_messages = [system_msg, user_msg]
    new_messages.append({"role": "assistant", "content": summary_text})
    for r in recent_rounds:
        new_messages.extend(r)

    return new_messages


# ── Tool Execution ─────────────────────────────────────────────


def execute_tool_call(
    tool_call: Dict[str, Any],
    tool_registry: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a single tool call and return the result."""
    func = tool_call.get("function", {})
    name = func.get("name", "")
    args_str = func.get("arguments", "{}")
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError:
        args = {"raw": args_str}

    if name not in tool_registry:
        return {"error": f"Unknown tool: {name}"}

    try:
        result = tool_registry[name](**args)
    except Exception as e:
        result = {"error": str(e)}

    return result


# ── Agent ──────────────────────────────────────────────────────


class DeepResearchAgent:
    """V4: Batch search + ReAct deep dive."""

    def __init__(
        self,
        client: VLLMClient,
        searcher: BrowseCompBM25Searcher,
        model_name: str = "qwen_auto",
        max_rounds: int = 5,
        max_tokens: int = 2048,
        search_top_k: int = 10,
        snippet_max_chars: int = 1200,
        max_recent_rounds: int = 4,
        temperature: float = 0.0,
        presearch_n_queries: int = 5,
        presearch_top_k: int = 10,
        presearch_max_docs: int = 30,
    ) -> None:
        self.client = client
        self.searcher = searcher
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_recent_rounds = max_recent_rounds
        self.presearch_n_queries = presearch_n_queries
        self.presearch_top_k = presearch_top_k
        self.presearch_max_docs = presearch_max_docs

        tool_specs, tool_registry = get_agent_tool_specs_and_registry(
            searcher=searcher,
            k=search_top_k,
            snippet_max_chars=snippet_max_chars,
        )
        self.tool_specs = tool_specs
        self.tool_registry = tool_registry

    def _build_initial_messages(
        self, question: str, presearch_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Format pre-search results as context
        evidence = format_rag_context(presearch_results)
        user_content = (
            f"Question: {question}\n\n"
            f"Pre-searched documents (from multiple queries):\n{evidence}\n\n"
            f"Based on the above documents, either provide your final answer "
            f"or use tools to search further or read specific documents in full."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _check_stalemate(
        self,
        messages: List[Dict[str, Any]],
        stale_threshold: int = 3,
    ) -> bool:
        """Check if the last N rounds found no new documents."""
        rounds = _extract_rounds(messages[2:])
        if len(rounds) < stale_threshold:
            return False

        recent_rounds = rounds[-stale_threshold:]
        all_docids: set = set()
        for r in recent_rounds:
            for msg in r:
                if msg["role"] == "tool":
                    content = msg.get("content", "")
                    try:
                        parsed = json.loads(content) if isinstance(content, str) else content
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "docid" in item:
                                    all_docids.add(item["docid"])
                    except (json.JSONDecodeError, TypeError):
                        pass

        return len(all_docids) == 0

    def _check_repeated_query(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if the most recent search query is too similar to a previous one."""
        previous_queries = _extract_previous_search_queries(messages)
        if len(previous_queries) < 2:
            return False
        latest = previous_queries[-1]
        earlier = previous_queries[:-1]
        return _is_similar_query(latest, earlier)

    def _extract_final_answer(self, content: str) -> str:
        """Extract Exact Answer from model output, with format filtering."""
        stripped = content.strip()
        # Filter out JSON-like garbage
        if stripped.startswith('{"') or stripped.startswith('["'):
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("{") or line.startswith("["):
                    continue
                if len(line) < 200 and not line.startswith('"'):
                    return line
            return ""

        # Try "Exact Answer: xxx" or "Answer: xxx"
        for prefix in ("exact answer:", "answer:"):
            for line in content.split("\n"):
                line = line.strip()
                if line.lower().startswith(prefix):
                    answer = line[len(prefix):].strip()
                    if answer and not answer.startswith('{"'):
                        return answer

        # Try extracting from the last non-empty line
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if lines and len(lines[-1]) < 100 and not lines[-1].startswith('{"'):
            return lines[-1]

        return content.strip()

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run Phase 1 (batch search) + Phase 2 (ReAct deep dive).

        Returns
        -------
        dict with keys: predicted_answer, status, messages
        """
        # ── Phase 1: Batch Search ──────────────────────────────
        queries = generate_diverse_queries(question, n=self.presearch_n_queries)
        presearch_results = batch_search(
            self.searcher, queries,
            top_k=self.presearch_top_k,
            max_total=self.presearch_max_docs,
        )

        if verbose:
            print(f"  [Pre-search] {len(queries)} queries -> {len(presearch_results)} unique docs")
            for q in queries:
                print(f"    Query: {q}")

        # ── Phase 2: ReACT Deep Dive ───────────────────────────
        messages = self._build_initial_messages(question, presearch_results)

        for round_id in range(1, self.max_rounds + 1):
            # Context management
            messages = manage_context(messages, self.max_recent_rounds)

            # No need to force first-round search — we already have pre-searched docs
            tool_choice = "auto"

            try:
                response = self.client.simple_chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=self.tool_specs,
                    tool_choice=tool_choice,
                )
            except Exception as e:
                if verbose:
                    print(f"  [Round {round_id}] LLM call failed: {e}")
                break

            message = response["choices"][0]["message"]
            raw_content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls") or []

            # Record assistant message
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": raw_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if verbose:
                if tool_calls:
                    tool_names = [
                        f"{tc['function']['name']}({tc['function']['arguments'][:50]})"
                        for tc in tool_calls
                    ]
                    print(f"  [Round {round_id}] Tools: {', '.join(tool_names)}")
                else:
                    print(f"  [Round {round_id}] Final answer")

            # No tool calls -> model gave final answer
            if not tool_calls:
                break

            # Execute tool calls
            for tool_call in tool_calls:
                result = execute_tool_call(tool_call, self.tool_registry)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # Check for repeated query
            if self._check_repeated_query(messages):
                if verbose:
                    print(f"  [Round {round_id}] Repeated query detected, injecting rephrase hint")
                messages.append({"role": "user", "content": REPHRASE_PROMPT})

            # Check for stalemate
            if self._check_stalemate(messages):
                if verbose:
                    print(f"  [Round {round_id}] Stalemate detected, forcing final answer")
                messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})
                try:
                    response = self.client.simple_chat(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=self.max_tokens,
                    )
                    final_content = response["choices"][0]["message"].get("content", "")
                    messages.append({"role": "assistant", "content": final_content})
                except Exception:
                    pass
                break
        else:
            # Max rounds reached, force answer
            if verbose:
                print(f"  Max rounds ({self.max_rounds}) reached, forcing final answer")
            messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})
            try:
                response = self.client.simple_chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
                final_content = response["choices"][0]["message"].get("content", "")
                messages.append({"role": "assistant", "content": final_content})
            except Exception:
                pass

        # Extract final answer
        predicted_answer = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                predicted_answer = self._extract_final_answer(msg["content"])
                if predicted_answer:
                    break

        status = "completed" if predicted_answer else "failed"

        return {
            "predicted_answer": predicted_answer,
            "status": status,
            "messages": messages,
        }
