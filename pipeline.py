"""Document processing pipeline with classified error handling.

Demonstrates four error-handling patterns in LangGraph:
1. Transient errors → RetryPolicy
2. LLM-recoverable errors → ToolNode(handle_tool_errors=True) + loop back
3. User-fixable errors → interrupt()
4. Unexpected errors → let them bubble up
"""

import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RetryPolicy, interrupt
from langsmith import traceable

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)


# --- State ---


class PipelineState(TypedDict):
    document: str
    messages: Annotated[list[AnyMessage], operator.add]
    extracted_clauses: list[dict]
    validation_errors: list[str]
    retry_count: int
    final_summary: str


# --- Retry policies ---

api_retry = RetryPolicy(
    max_attempts=5,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
)

llm_retry = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=5.0,
    jitter=True,
)


# --- Tools ---


@tool
def extract_clause(text: str, clause_type: str) -> dict:
    """Extract a specific clause from contract text.

    Args:
        text: The contract text to search.
        clause_type: One of 'termination', 'liability', 'indemnification', 'payment'.
    """
    valid_types = {"termination", "liability", "indemnification", "payment"}
    if clause_type not in valid_types:
        raise ValueError(
            f"Invalid clause_type '{clause_type}'. Must be one of: {valid_types}"
        )
    return {
        "clause_type": clause_type,
        "text": f"Extracted {clause_type} clause from document.",
        "confidence": 0.92,
    }


@tool
def check_compliance(clause: str, regulation: str) -> dict:
    """Check if a clause complies with a specific regulation.

    Args:
        clause: The clause text to check.
        regulation: The regulation identifier (e.g., 'GDPR-Art17', 'SOX-302').
    """
    if not clause.strip():
        raise ValueError("Empty clause text provided. Extract the clause first.")
    return {"compliant": True, "regulation": regulation, "notes": "No issues found."}


tools = [extract_clause, check_compliance]
tool_node = ToolNode(tools, handle_tool_errors=True)


# --- Nodes ---


@traceable(name="agent", run_type="chain")
def agent_node(state: PipelineState) -> dict:
    system = SystemMessage(
        content="You are a contract analysis agent. Use the provided tools to "
        "extract and validate clauses. If a tool returns an error, read "
        "the error message carefully and adjust your arguments. "
        "Available clause types: termination, liability, indemnification, payment."
    )
    messages = [system] + state["messages"]
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}


def should_continue(state: PipelineState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "validate"


def post_tool(state: PipelineState) -> dict:
    tool_results = []
    for msg in state["messages"]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if "Extracted" in msg.content:
                tool_results.append(
                    {
                        "clause_type": "extracted",
                        "text": msg.content,
                        "confidence": 0.92,
                    }
                )
    if tool_results:
        return {"extracted_clauses": tool_results}
    return {}


@traceable(name="validate_document", run_type="chain")
def validate_node(state: PipelineState) -> dict:
    clauses = state.get("extracted_clauses", [])
    errors = []

    if not clauses:
        errors.append("No clauses extracted from document.")

    required_types = {"termination", "payment"}
    found_types = {c["clause_type"] for c in clauses if isinstance(c, dict)}
    missing = required_types - found_types
    if missing:
        errors.append(f"Missing required clause types: {missing}")

    low_confidence = [
        c for c in clauses if isinstance(c, dict) and c.get("confidence", 1.0) < 0.7
    ]
    if low_confidence:
        types = [c["clause_type"] for c in low_confidence]
        errors.append(f"Low confidence extractions for: {types}")

    if errors:
        human_input = interrupt(
            {
                "type": "validation_errors",
                "errors": errors,
                "message": "Document validation failed. Please review and provide corrections.",
                "document_preview": state["document"][:500],
            }
        )
        return {
            "extracted_clauses": human_input.get("corrected_clauses", clauses),
            "validation_errors": [],
        }

    return {"validation_errors": []}


@traceable(name="summarize", run_type="chain")
def summarize_node(state: PipelineState) -> dict:
    clauses_text = "\n".join(
        f"- {c['clause_type']}: {c['text']}" for c in state["extracted_clauses"]
    )
    response = llm.invoke(
        [
            SystemMessage(
                content="Summarize the following contract clauses into a concise executive summary. "
                "Flag any compliance concerns."
            ),
            HumanMessage(content=f"Contract clauses:\n{clauses_text}"),
        ]
    )
    return {"final_summary": response.content}


# --- Graph ---

builder = StateGraph(PipelineState)

builder.add_node("agent", agent_node, retry=llm_retry)
builder.add_node("tools", tool_node, retry=api_retry)
builder.add_node("post_tool", post_tool)
builder.add_node("validate", validate_node)
builder.add_node("summarize", summarize_node, retry=llm_retry)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "validate": "validate"}
)
builder.add_edge("tools", "post_tool")
builder.add_edge("post_tool", "agent")
builder.add_edge("validate", "summarize")
builder.add_edge("summarize", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# --- Run ---

if __name__ == "__main__":
    from langsmith import tracing_context

    document = (
        "This contract between Party A and Party B includes: "
        "Termination: Either party may terminate with 30 days notice. "
        "Payment: Net 30 terms apply."
    )

    config = {"configurable": {"thread_id": "contract-review-1"}}

    with tracing_context(
        metadata={"document_length": len(document), "pipeline_version": "v3"},
        tags=["production", "error-handling-v3"],
    ):
        result = graph.invoke(
            {
                "document": document,
                "messages": [
                    HumanMessage(
                        content=f"Process this contract:\n\n{document}"
                    )
                ],
                "extracted_clauses": [],
                "validation_errors": [],
                "retry_count": 0,
                "final_summary": "",
            },
            config,
        )

    print("Summary:", result.get("final_summary", "No summary generated."))
