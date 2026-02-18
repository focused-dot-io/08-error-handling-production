"""Evaluation suite for the document processing pipeline.

Tests error classification accuracy, recovery efficiency, and output quality.
"""

from langchain_core.messages import HumanMessage
from langsmith import Client, evaluate
from openevals.llm import create_llm_as_judge

from pipeline import graph

ls_client = Client()

DATASET_NAME = "error-handling-evals"

if not ls_client.has_dataset(dataset_name=DATASET_NAME):
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Document processing pipeline error handling evaluation",
    )
    ls_client.create_examples(
        dataset_id=dataset.id,
    inputs=[
        {
            "document": (
                "This contract between Party A and Party B includes: "
                "Termination: Either party may terminate with 30 days notice. "
                "Payment: Net 30 terms apply."
            ),
            "thread_id": "eval-1",
        },
        {
            "document": (
                "This agreement covers liability limitations and "
                "indemnification clauses only."
            ),
            "thread_id": "eval-2",
        },
        {"document": "", "thread_id": "eval-3"},
    ],
    outputs=[
        {"should_succeed": True, "required_clauses": ["termination", "payment"]},
        {"should_succeed": False, "missing_clauses": ["termination", "payment"]},
        {"should_succeed": False, "error_type": "empty_document"},
    ],
)


QUALITY_PROMPT = """\
Document: {inputs}
Pipeline output: {outputs}

Rate 0.0-1.0 on:
- Completeness: Did the summary cover all extracted clauses?
- Accuracy: Are the clause descriptions faithful to the source?
- Error handling: If the document was incomplete, did the pipeline flag it appropriately?

Return ONLY: {{"score": <float>, "reasoning": "<explanation>"}}"""

quality_judge = create_llm_as_judge(
    prompt=QUALITY_PROMPT,
    model="anthropic:claude-sonnet-4-5-20250929",
    feedback_key="quality",
    continuous=True,
)


def error_classification(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the pipeline correctly classify errors?"""
    should_succeed = reference_outputs.get("should_succeed", True)
    has_summary = bool(outputs.get("final_summary"))
    has_errors = bool(outputs.get("validation_errors"))

    if should_succeed:
        score = 1.0 if has_summary and not has_errors else 0.0
    else:
        score = 1.0 if has_errors or not has_summary else 0.0
    return {"key": "error_classification", "score": score}


def recovery_efficiency(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """How many retries did it take? Lower is better."""
    retry_count = outputs.get("retry_count", 0)
    if retry_count == 0:
        score = 1.0
    elif retry_count <= 2:
        score = 0.7
    else:
        score = 0.3
    return {"key": "recovery_efficiency", "score": score}


def target(inputs: dict) -> dict:
    config = {"configurable": {"thread_id": inputs["thread_id"]}}
    try:
        result = graph.invoke(
            {
                "document": inputs["document"],
                "messages": [
                    HumanMessage(
                        content=f"Process this contract:\n\n{inputs['document']}"
                    )
                ],
                "extracted_clauses": [],
                "validation_errors": [],
                "retry_count": 0,
                "final_summary": "",
            },
            config,
        )
        return {
            "final_summary": result.get("final_summary", ""),
            "validation_errors": [str(e) for e in result.get("validation_errors", [])],
            "retry_count": result.get("retry_count", 0),
        }
    except Exception as e:
        return {
            "final_summary": "",
            "validation_errors": [str(e)],
            "retry_count": 0,
        }


if __name__ == "__main__":
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[quality_judge, error_classification, recovery_efficiency],
        experiment_prefix="error-handling-v1",
        max_concurrency=2,
    )
    print("\nEvaluation complete. Check LangSmith for results.")
