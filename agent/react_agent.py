"""
Deep Research Agent V10 — Auto Deep-Read + LLM Summarization.

Round 1 forced search, Rounds 2-3 gap analysis, then auto deep-read
top docs + LLM summary injected into context. Rounds 4+ continue
with gap analysis. Forces answer on stalemate or max rounds.
"""

import json
import re
from typing import Any, Dict, List

from .browsecomp_searcher import BrowseCompBM25Searcher
from .tools import get_agent_tool_specs_and_registry
from .vllm_client import VLLMClient


# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Deep Research Agent. You must search documents to answer the question.

## Available Tools
- search(query): BM25 keyword search. Returns top results with doc IDs, scores, and snippets.
- get_document(docid): Retrieve the full text of a document by its ID.

## Search Strategy
- Start by searching for the most distinctive entity or clue in the question.
- After each search, identify what information you STILL NEED and search for that specific missing piece.
- Each search should target a SPECIFIC missing clue, not repeat the whole question.
- Search with SHORT keyword phrases (2-4 words).
- If a snippet mentions relevant information, use get_document to read the full text.

## Rules
- Search with DIFFERENT keywords each time. Do not repeat queries.
- You MUST provide your best guess. NEVER say "cannot be determined" or "information not available".
- Focus on EXACT facts (names, dates, numbers, titles), not vague descriptions.

## Output Format (MANDATORY)
You MUST end your final response with EXACTLY these two lines:
Explanation: <your reasoning based on the evidence found>
Exact Answer: <your precise answer>

While searching, call tools. Do NOT output the answer format until you have evidence or have exhausted searches."""

FORCE_ANSWER_PROMPT = (
    "You must now provide your final answer based on ALL the evidence you have gathered. "
    "You MUST give your best guess — do NOT say 'cannot be determined' or 'information not available'. "
    "Use the format:\nExplanation: <your reasoning based on evidence>\nExact Answer: <your best guess>"
)

GAP_ANALYSIS_PROMPT = (
    "Review the search results above carefully.\n"
    "- What specific clues from the question have you found evidence for?\n"
    "- What information is still missing?\n"
    "If critical information is missing, search with targeted keywords for the missing piece. "
    "If a snippet mentions relevant information but is incomplete, use get_document to read the full text.\n"
    "If you have enough evidence, provide your final answer using the mandatory format."
)

REPHRASE_PROMPT = (
    "You already searched for very similar keywords and did not find useful results. "
    "You MUST try completely different keywords now — use synonyms, search for a different entity, "
    "or approach the question from a completely different angle."
)

DEEP_READ_SUMMARY_SYSTEM = (
    "You are analyzing documents to find information relevant to a question. "
    "Extract ONLY the specific facts, names, dates, numbers, and details that are "
    "directly relevant to answering the question. Ignore irrelevant content."
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

_MAX_SEARCH_RESULTS_IN_CONTEXT = 5
_MAX_GETDOC_CHARS = 2000


def _truncate_tool_result(result: Any) -> Any:
    """Truncate tool results to keep context within token limits."""
    if isinstance(result, list):
        # search results: keep top N, strip long snippets
        truncated = []
        for item in result[:_MAX_SEARCH_RESULTS_IN_CONTEXT]:
            if isinstance(item, dict):
                item = dict(item)
                if "snippet" in item and len(item["snippet"]) > 400:
                    item["snippet"] = item["snippet"][:400]
                truncated.append(item)
        return truncated
    if isinstance(result, dict) and "text" in result:
        # get_document: truncate full text
        truncated = dict(result)
        if len(truncated["text"]) > _MAX_GETDOC_CHARS:
            truncated["text"] = truncated["text"][:_MAX_GETDOC_CHARS]
        return truncated
    return result


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
    """V10: Auto deep-read + LLM summarization after Round 3."""

    def __init__(
        self,
        client: VLLMClient,
        searcher: BrowseCompBM25Searcher,
        model_name: str = "qwen_auto",
        max_rounds: int = 10,
        max_tokens: int = 4096,
        search_top_k: int = 20,
        snippet_max_chars: int = 600,
        max_recent_rounds: int = 3,
        temperature: float = 0.0,
        deep_read_after_round: int = 3,
        deep_read_top_n: int = 5,
        deep_read_doc_max_chars: int = 3000,
    ) -> None:
        self.client = client
        self.searcher = searcher
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_recent_rounds = max_recent_rounds
        self.deep_read_after_round = deep_read_after_round
        self.deep_read_top_n = deep_read_top_n
        self.deep_read_doc_max_chars = deep_read_doc_max_chars

        tool_specs, tool_registry = get_agent_tool_specs_and_registry(
            searcher=searcher,
            k=search_top_k,
            snippet_max_chars=snippet_max_chars,
        )
        self.tool_specs = tool_specs
        self.tool_registry = tool_registry

    def _build_initial_messages(self, question: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
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

    def _check_repeated_query_and_stale(
        self, messages: List[Dict[str, Any]]
    ) -> bool:
        """Check if the latest search query is repeated AND returned no new documents."""
        previous_queries = _extract_previous_search_queries(messages)
        if len(previous_queries) < 2:
            return False
        latest = previous_queries[-1]
        earlier = previous_queries[:-1]
        if not _is_similar_query(latest, earlier):
            return False

        # Query is repeated — only flag if the latest search returned no new docids
        already_seen: set = set()
        for msg in messages:
            if msg["role"] == "tool":
                content = msg.get("content", "")
                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "docid" in item:
                                already_seen.add(item["docid"])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Check the last tool result(s) for new docids
        latest_docids: set = set()
        for msg in reversed(messages):
            if msg["role"] == "tool":
                content = msg.get("content", "")
                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "docid" in item:
                                latest_docids.add(item["docid"])
                except (json.JSONDecodeError, TypeError):
                    pass
                break  # only check the most recent tool result

        new_docs = latest_docids - already_seen
        return len(new_docs) == 0

    def _collect_top_docids(
        self, messages: List[Dict[str, Any]], top_n: int,
    ) -> List[tuple[str, float]]:
        """Collect top-N unique (docid, score) pairs from all search results in messages."""
        doc_scores: dict[str, float] = {}
        for msg in messages:
            if msg["role"] != "tool":
                continue
            content = msg.get("content", "")
            try:
                parsed = json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                docid = item.get("docid")
                score = item.get("score", 0.0)
                if docid and (docid not in doc_scores or score > doc_scores[docid]):
                    doc_scores[docid] = score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_n]

    def _run_deep_read(
        self,
        question: str,
        messages: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Auto deep-read top docs and inject LLM summary into context."""
        top_docs = self._collect_top_docids(messages, self.deep_read_top_n)
        if not top_docs:
            if verbose:
                print("  [Deep-Read] No documents found to read")
            return messages

        if verbose:
            docids_str = ", ".join(d[0] for d in top_docs)
            print(f"  [Deep-Read] Reading docs: {docids_str}")

        # Fetch full text for each doc
        doc_texts: list[str] = []
        for docid, score in top_docs:
            try:
                result = self.tool_registry["get_document"](docid=docid)
                text = result.get("text", "") if isinstance(result, dict) else str(result)
                if len(text) > self.deep_read_doc_max_chars:
                    text = text[: self.deep_read_doc_max_chars]
                doc_texts.append(f"[Document: {docid} (score: {score:.2f})]\n{text}")
            except Exception as e:
                if verbose:
                    print(f"  [Deep-Read] Failed to read {docid}: {e}")

        if not doc_texts:
            return messages

        # LLM summarization
        docs_block = "\n---\n".join(doc_texts)
        summary_messages = [
            {"role": "system", "content": DEEP_READ_SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {question}\n\nDocuments:\n{docs_block}\n\nExtract all information relevant to the question.",
            },
        ]

        try:
            response = self.client.simple_chat(
                model=self.model_name,
                messages=summary_messages,
                temperature=0.0,
                max_tokens=1024,
                extra_payload={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            summary_text = response["choices"][0]["message"].get("content", "")
        except Exception as e:
            if verbose:
                print(f"  [Deep-Read] LLM summarization failed: {e}")
            return messages

        if not summary_text.strip():
            return messages

        # Inject summary as user message
        analysis_msg = {
            "role": "user",
            "content": (
                f"[Document Analysis - Key findings from top documents]:\n"
                f"{summary_text}\n\n"
                "Continue searching if needed, or provide your final answer based on this analysis."
            ),
        }
        messages.append(analysis_msg)

        if verbose:
            preview = summary_text[:120].replace("\n", " ")
            print(f"  [Deep-Read] Summary injected: {preview}...")

        return messages

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
            answer = lines[-1]
            # Filter out "cannot be determined" type answers
            if any(phrase in answer.lower() for phrase in (
                "cannot be determined", "not available", "not found",
                "information cannot", "information is not",
            )):
                return ""
            return answer

        return content.strip()

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run gap-driven ReAct loop.

        Returns
        -------
        dict with keys: predicted_answer, status, messages
        """
        messages = self._build_initial_messages(question)

        for round_id in range(1, self.max_rounds + 1):
            # Context management
            messages = manage_context(messages, self.max_recent_rounds)

            # Only force search on round 1; later rounds use gap analysis
            if round_id == 1:
                tool_choice = "required"
            else:
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

            # Execute tool calls and truncate results to prevent context overflow
            for tool_call in tool_calls:
                result = execute_tool_call(tool_call, self.tool_registry)
                truncated = _truncate_tool_result(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(truncated, ensure_ascii=False),
                })

            # Check for repeated query with no new results
            if self._check_repeated_query_and_stale(messages):
                if verbose:
                    print(f"  [Round {round_id}] Repeated query with no new results, injecting rephrase hint")
                # Replace gap analysis with rephrase prompt (gap analysis not yet injected)
                messages.append({"role": "user", "content": REPHRASE_PROMPT})
            else:
                # Inject gap analysis prompt after each round with tool results
                messages.append({"role": "user", "content": GAP_ANALYSIS_PROMPT})
                if verbose:
                    print(f"  [Round {round_id}] Injected gap analysis prompt")

            # V10: Auto deep-read after specified round
            if round_id == self.deep_read_after_round:
                messages = self._run_deep_read(question, messages, verbose)

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

        # Extract final answer — search all assistant messages for content
        predicted_answer = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                predicted_answer = self._extract_final_answer(msg["content"])
                if predicted_answer:
                    break

        # If still empty, try extracting from any assistant content with substance
        if not predicted_answer:
            for msg in reversed(messages):
                if msg["role"] == "assistant" and msg.get("content"):
                    content = msg["content"].strip()
                    content = re.sub(r"ჵ.*? mówi", "", content, flags=re.DOTALL)
                    content = re.sub(r"</?think>", "", content)
                    for line in content.split("\n"):
                        line = line.strip()
                        if line and len(line) > 3 and not line.startswith("Okay") and not line.startswith("Let me"):
                            predicted_answer = line
                            break
                    if predicted_answer:
                        break

        status = "completed" if predicted_answer else "failed"

        return {
            "predicted_answer": predicted_answer,
            "status": status,
            "messages": messages,
        }
