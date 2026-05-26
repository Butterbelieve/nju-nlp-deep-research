"""
Deep Research Agent — 基于 ReAct 模式的多轮检索问答 Agent。

核心流程：
  Thought → Action (tool call) → Observation (tool result) → Thought → ... → Final Answer

实验目标：在 BrowseComp-Plus 语料上实现多轮检索，达到 12%+ 正确率。
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher
from .tools import get_agent_tool_specs_and_registry
from .vllm_client import VLLMClient


# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Deep Research Agent. Your task is to find precise, factual answers to questions by searching through a document collection.

## Available Tools
- search(query): Search the document collection using BM25. Returns top results with document IDs, scores, and snippets.
- get_document(docid): Retrieve the full text of a specific document by its ID.

## Strategy
1. Carefully analyze the question. Identify key entities, dates, locations, and specific details.
2. Start with a targeted search using the most distinctive keywords from the question.
3. If the initial search doesn't find relevant documents, try:
   - Different keywords, synonyms, or related terms
   - Break complex questions into simpler sub-questions and search for each
   - Search for specific entities (names, places, dates) mentioned in the question
   - Try broader or narrower queries
4. When you find a potentially relevant document in search results, use get_document to read the full text for details.
5. Cross-reference information from multiple documents when possible.
6. Continue searching until you find strong, specific evidence for an answer.

## Important Rules
- Always search before answering. Do NOT answer from your own knowledge.
- Each search should be different from previous ones — do not repeat the same query.
- If search results are irrelevant, reformulate your query rather than giving up.
- Focus on finding EXACT facts (names, dates, numbers, titles) rather than vague descriptions.

## Output Format
When you have found enough evidence, provide your final answer in EXACTLY this format:
Explanation: <brief explanation citing the evidence you found>
Exact Answer: <your precise answer>

When you are still searching, write your reasoning and call tools. Do NOT give a final answer until you have sufficient evidence."""


# ── 上下文管理 ──────────────────────────────────────────────────


def _extract_rounds(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """将 assistant/tool 消息拆分成轮次，每轮以 assistant 开头。"""
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
    """将一轮 (assistant + tool results) 压缩为一行摘要。"""
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
    max_recent_rounds: int = 3,
) -> List[Dict[str, Any]]:
    """保留 system + user + 最近 N 轮完整对话，旧轮次压缩为摘要。"""
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
    """执行单个工具调用，返回结果。"""
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
    """基于 ReAct 模式的多轮检索 Deep Research Agent。"""

    def __init__(
        self,
        client: VLLMClient,
        searcher: BrowseCompBM25Searcher,
        model_name: str = "qwen_auto",
        max_rounds: int = 5,
        max_tokens: int = 1024,
        search_top_k: int = 5,
        snippet_max_chars: int = 1200,
        max_recent_rounds: int = 3,
        temperature: float = 0.3,
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
            {"role": "user", "content": question},
        ]

    def _check_stalemate(
        self,
        messages: List[Dict[str, Any]],
        stale_threshold: int = 2,
    ) -> bool:
        """检查最近 N 轮搜索是否都没有找到新文档（信息饱和）。"""
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

        # 如果最近 stale_threshold 轮都没有检索到任何文档
        return len(all_docids) == 0

    def _extract_final_answer(self, content: str) -> str:
        """从模型输出中提取 Exact Answer。"""
        # 尝试匹配 "Exact Answer: xxx"
        for line in content.split("\n"):
            line = line.strip()
            if line.lower().startswith("exact answer:"):
                return line[len("exact answer:"):].strip()

        # 如果没有格式化输出，返回整个内容
        return content.strip()

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        运行 ReAct 循环，返回完整轨迹。

        Returns
        -------
        dict with keys: query, predicted_answer, status, messages
        """
        messages = self._build_initial_messages(question)

        for round_id in range(1, self.max_rounds + 1):
            # 上下文管理：压缩旧轮次
            messages = manage_context(messages, self.max_recent_rounds)

            # 调用 LLM
            try:
                response = self.client.simple_chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=self.tool_specs,
                    tool_choice="auto",
                )
            except Exception as e:
                if verbose:
                    print(f"  [Round {round_id}] LLM call failed: {e}")
                break

            message = response["choices"][0]["message"]
            raw_content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls") or []

            # 记录 assistant 消息
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

            # 没有工具调用 → 模型给出最终答案
            if not tool_calls:
                break

            # 执行工具调用
            for tool_call in tool_calls:
                result = execute_tool_call(tool_call, self.tool_registry)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # 检查信息饱和
            if self._check_stalemate(messages):
                if verbose:
                    print(f"  [Round {round_id}] Stalemate detected, forcing final answer")
                # 强制让模型给出答案
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
            # 达到最大轮次，强制回答
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

        # 提取最终答案
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
