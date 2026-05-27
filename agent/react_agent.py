"""
Deep Research Agent — 基于 ReAct 模式的多轮检索问答 Agent.

Thought -> Action (tool call) -> Observation (tool result) -> Thought -> ... -> Final Answer

"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher
from .tools import get_agent_tool_specs_and_registry
from .vllm_client import VLLMClient


# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Deep Research Agent. Your task is to find precise, factual answers to questions by searching through a document collection. You MUST search before answering — never answer from your own knowledge.

## Available Tools
- search(query): Search the document collection using BM25 keyword matching. Returns top results with document IDs, scores, and snippets.
- get_document(docid): Retrieve the full text of a specific document by its ID.

## CRITICAL: BM25 Search Strategy
BM25 is a keyword-based search engine. It matches the words in your query against documents. Your search quality depends entirely on choosing the right keywords.

- Do NOT search with the full question sentence. Extract the most distinctive keywords and search with those.
  Example: "A football match between 1990-1994 with a Brazilian referee and 4 yellow cards" -> search "Brazil referee football 1990 1994 yellow card"
- If search returns no relevant results, try different keywords:
  - Use synonyms or related terms
  - Try shorter queries with only the most unique keywords
  - Try longer queries with more context
  - Search for specific entities separately (person name, place, date)
- Search from multiple angles. If searching by person name fails, try searching by event, location, or time period.

## Search Workflow
1. Analyze the question. Identify key entities: names, dates, locations, events, numbers.
2. Extract the most distinctive keywords and call search.
3. If search results contain a relevant document, use get_document to read its full text.
4. If search results are irrelevant, reformulate your query with different keywords and search again.
5. Cross-reference information from multiple documents when possible.
6. Search at least 2 times before giving a final answer.

## Rules
- You MUST call search at least once before answering. NEVER answer without searching.
- Each search must use different keywords — do not repeat the same query.
- If results are irrelevant, change your keywords instead of giving up.
- Focus on finding EXACT facts (names, dates, numbers, titles), not vague descriptions.

## Output Format
When you have enough evidence, output your final answer in EXACTLY this format:
Explanation: <brief explanation citing the evidence you found>
Exact Answer: <your precise answer>

While searching, write your reasoning and call tools. Do NOT output the final answer format until you have sufficient evidence."""


# ── 上下文管理 ──────────────────────────────────────────────────


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


# ── 工具执行 ────────────────────────────────────────────────────


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


# ── Agent 主类 ──────────────────────────────────────────────────


class DeepResearchAgent:
    """ReAct-based multi-round retrieval Deep Research Agent."""

    def __init__(
        self,
        client: VLLMClient,
        searcher: BrowseCompBM25Searcher,
        model_name: str = "qwen_auto",
        max_rounds: int = 8,
        max_tokens: int = 2048,
        search_top_k: int = 10,
        snippet_max_chars: int = 1200,
        max_recent_rounds: int = 4,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_recent_rounds = max_recent_rounds

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
            {"role": "user", "content": f"{question}\n\nPlease start by searching for relevant information using the search tool. Do NOT answer directly."},
        ]

    def _check_stalemate(
        self,
        messages: List[Dict[str, Any]],
        stale_threshold: int = 2,
    ) -> bool:
        """Check if the last N rounds found no new documents (information saturation)."""
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

    def _extract_final_answer(self, content: str) -> str:
        """Extract Exact Answer from model output, supporting multiple formats."""
        # Try "Exact Answer: xxx" or "Answer: xxx"
        for prefix in ("exact answer:", "answer:"):
            for line in content.split("\n"):
                line = line.strip()
                if line.lower().startswith(prefix):
                    return line[len(prefix):].strip()

        # Try extracting from the last non-empty line if content looks like a short answer
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if lines and len(lines[-1]) < 100:
            return lines[-1]

        return content.strip()

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the ReAct loop and return the full trajectory.

        Returns
        -------
        dict with keys: query, predicted_answer, status, messages
        """
        messages = self._build_initial_messages(question)

        for round_id in range(1, self.max_rounds + 1):
            # Context management: compress old rounds
            messages = manage_context(messages, self.max_recent_rounds)

            # Round 1: force tool call to prevent answering without search
            tool_choice = "required" if round_id == 1 else "auto"

            # Call LLM
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

            # Check for stalemate
            if self._check_stalemate(messages):
                if verbose:
                    print(f"  [Round {round_id}] Stalemate detected, forcing final answer")
                messages.append({
                    "role": "user",
                    "content": (
                        "You have searched multiple times without finding new relevant information. "
                        "Based on the evidence you have gathered so far, please provide your best answer now. "
                        "Use the format:\nExplanation: ...\nExact Answer: ..."
                    ),
                })
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
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the maximum number of search rounds. "
                    "Based on the evidence you have gathered, please provide your best answer now. "
                    "Use the format:\nExplanation: ...\nExact Answer: ..."
                ),
            })
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
                break

        status = "completed" if predicted_answer else "failed"

        return {
            "predicted_answer": predicted_answer,
            "status": status,
            "messages": messages,
        }
