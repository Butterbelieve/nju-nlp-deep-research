# NLP 课程项目：Deep Research Agent 实验指南

本实验的目标是基于 `BrowseComp-Plus` 搭建一个能够进行多轮检索、维护中间状态、并基于证据给出最终答案的 agent。

你们拿到的材料只有三部分：

- `agent/` 文件夹
- 数据集与语料
- `agent_vllm.ipynb` 和 `agent_vllm_weather.ipynb`

## 1. 实验目标

本实验重点考察：

1. 多轮检索 loop 与停止条件
2. 历史与上下文管理
3. Prompt 设计
4. 基于证据的最终回答

也就是说，你们要做的不是单次 `search -> answer`，而是一个能逐步推进问题求解的 Deep Research Agent。

## 2. 你们会拿到什么

### 2.1 notebook

- [agent_vllm.ipynb](agent_vllm.ipynb)
  - 单步 baseline
- [agent_vllm_weather.ipynb](agent_vllm_weather.ipynb)
  - 本地工具调用链路演示

### 2.2 `agent/` 目录

- [agent/browsecomp_searcher.py](agent/browsecomp_searcher.py)
  - 本地 BM25 检索实现
- [agent/build_bm25_index.py](agent/build_bm25_index.py)
  - 构建 BM25 索引
- [agent/tools.py](agent/tools.py)
  - 检索工具定义
- [agent/vllm_client.py](agent/vllm_client.py)
  - 调用 vLLM 服务
- [agent/dataset_utils.py](agent/dataset_utils.py)
  - 读取数据集
- [agent/pangu_tool_parser.py](agent/pangu_tool_parser.py)
  - Pangu 工具调用 parser
- [agent/pangu_chat_template.jinja](agent/pangu_chat_template.jinja)
  - Pangu chat template
- [agent/eval.py](agent/eval.py)
  - 自动评估脚本，使用 LLM 判断预测答案与标准答案是否一致

### 2.3 数据

- `browsecomp-plus-corpus/`
  - 全量离线语料
- `browsecomp_plus_hard50.jsonl`
  - 课堂调试样例集

## 3. 下载模型

首先在终端中克隆所需模型（二选一，推荐 Qwen）：

```bash
cd nlp-exp
# Qwen3-8B（推荐）
git clone https://atomgit.com/hf_mirrors/MindSpore-Lab/Qwen3-8B.git

# openPangu-Embedded-7B（备选）
git clone https://atomgit.com/ascend-tribe/openPangu-Embedded-7B-DeepDiver.git
```

## 4. 启动 vLLM 服务

### 4.1 Qwen 路线

git config --global http.version HTTP/1.1

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

### 4.2 Pangu 路线

```bash
vllm serve ./openPangu-Embedded-7B-DeepDiver \
  --served-model-name pangu_auto \
  --enable-auto-tool-choice \
  --tool-parser-plugin agent/pangu_tool_parser.py \
  --tool-call-parser pangu_deepdiver \
  --chat-template agent/pangu_chat_template.jinja \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

服务地址默认写成：

```text
http://127.0.0.1:8000/v1
```

说明：

- `vLLM` 需要在 `Jupyter` 之外单独启动
- 启动后需要一直保持在终端中运行
- notebook 只负责调用已经启动好的 `vLLM` 服务

## 5. Jupyter 使用方式

打开课程提供的 Jupyter 环境后，建议先查看：

1. agent_vllm_weather.ipynb
2. agent_vllm.ipynb

说明：

- 依赖安装与 notebook 内环境准备步骤，已经写在 notebook 中
- 请先在终端启动并保持 `vLLM` 运行，再打开 notebook

## 6. 推荐实验流程

建议按下面顺序完成：

1. 跑通 `agent_vllm.ipynb`
2. 理解单步 baseline
3. 参考 `agent_vllm_weather.ipynb` agent实现
4. 自己实现多轮 agent loop
5. 加入 query 改写、状态管理和停止条件
6. 在 `hard50` 或开发集上调试
7. 导出统一格式结果

## 7. 实验要求

### 6.1 必做内容

你们需要在 baseline 基础上完成：

1. 多轮检索 loop
2. 明确的停止条件
3. 历史与上下文管理
4. Prompt 设计
5. 基于证据的最终回答

### 6.2 限制

- 不允许替换检索器
- 不允许使用额外外部检索服务（google/bing）
- 不允许引入 benchmark 外部知识库

### 6.3 推荐改进方向

- query reformulation
- 维护已确认事实与待确认子问题
- 历史信息压缩
- 避免重复搜索
- 最终答案前做证据校验

## 8. 自动评估

我们提供了自动评估脚本 `agent/eval.py`，使用 LLM 自动判断你的预测答案与标准答案是否一致，并输出准确率（acc）和详细评估结果。

使用方法：

```bash
python -m agent.eval \
  --submission runs/submission.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/eval_results.jsonl
```

你也可以在 notebook 中直接调用 `run_evaluation()` 函数进行评估。

## 9. 统一输出与提交格式

为了方便老师审核和自动化评估，所有同学都需要按统一格式提交结果。

### 9.1 提交文件

最终提交文件命名格式：

```text
学号-姓名-submission-最终得分.jsonl
```

### 9.2 单题结果格式

`submission.jsonl` 采用 `JSON Lines` 格式：

- 每一行对应一道题
- 每一行都是一个完整 JSON 对象
- 每个 JSON 对象同时包含最终答案和完整轨迹

单题至少包含：

```json
{
  "query_id": "442",
  "predicted_answer": "THE DAWN OF AUSTRALIAN COLONISATION",
  "status": "completed",
  "messages": [
    {
      "role": "system",
      "content": "你是一个 Deep Research Agent ..."
    },
    {
      "role": "user",
      "content": "原始题目文本"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "search",
            "arguments": "{\"query\": \"...\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "工具返回结果或结果摘要"
    },
    {
      "role": "assistant",
      "content": "Explanation: ...\nExact Answer: THE DAWN OF AUSTRALIAN COLONISATION\nConfidence: 72%"
    }
  ]
}
```

其中建议作为必填字段：

- `query_id`
- `status`
- `predicted_answer`
- `messages`

除了上面这些字段，不要求额外提交派生统计字段。像下面这些信息都可以由评测脚本从完整 `messages` 中还原或统计：

- 工具调用次数
- 检索到过的文档 id
- 最终一轮 assistant 输出
- 每一轮 tool call / tool result 对应关系

### 9.3 消息格式

为了能完整还原整个对话过程，建议直接按时间顺序记录 `messages`。

建议支持下面几类 message：

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "search",
        "arguments": "{\"query\": \"...\"}"
      }
    }
  ]
}
```

推荐约定如下：

- `system` / `user` / `assistant` / `tool` 都统一放进 `messages`
- `assistant` 如果发起工具调用，就在该条 message 中记录 `tool_calls`
- `tool` message 记录工具返回内容，并通过 `tool_call_id` 对应到上一条 `assistant.tool_calls`
- 最终答案就是最后一条 `assistant` message 的 `content`

如果你们实现了显式状态管理，建议额外加入但不要重复正文：

- `state_summary`
- `current_subgoal`
- `next_action_plan`

这样老师可以同时检查：

- 最终答案是否正确
- 整个对话过程是否能被完整还原
- 检索和决策过程是否合理

## 10. 一句话提醒

这次实验的目标不是让模型搜一次就猜答案，而是让它在固定检索环境中，逐步推进、保留证据、最后输出一个可以检查的答案。

tip：使用 `agent/eval.py` 可以自动评估你的 agent 效果，无需手动比对答案。
