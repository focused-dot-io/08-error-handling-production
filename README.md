# Error Handling and Production Patterns

A document processing pipeline demonstrating four classified error-handling patterns in LangGraph: transient errors with RetryPolicy, LLM-recoverable errors via ToolNode with handle_tool_errors, user-fixable errors using interrupt() for human-in-the-loop correction, and unexpected errors that bubble up. The pipeline extracts contract clauses, validates them against policy, and generates executive summaries.

## Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)
- [LangSmith API key](https://smith.langchain.com/)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `LANGSMITH_API_KEY` | Yes | LangSmith API key for tracing and evals |
| `LANGSMITH_TRACING` | Yes | Set to `true` to enable tracing |

## Running

```bash
python pipeline.py
```

To run evaluations:

```bash
python evals.py
```

## Article

[Error Handling and Production Patterns for LangGraph](#)
