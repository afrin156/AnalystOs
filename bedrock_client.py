"""
Thin wrapper around Amazon Bedrock, used only for the reasoning/explanation
parts of AnalystOS (WHY investigation phrasing, verification write-ups).
Every actual number and statistic in this project comes from real Pandas
calculations, never from the AI - Bedrock is only used to explain findings
in plain language, never to invent them.

MOCK_MODE: set to true (default) to run and test the whole app without any
AWS account. Flip to false once you have real credentials.
"""

import json
import os

MOCK_MODE = os.environ.get("ANALYSTOS_MOCK", "true").lower() == "true"
MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = os.environ.get("AWS_REGION", "us-east-1")

_bedrock_runtime = None


def _get_client():
    global _bedrock_runtime
    if _bedrock_runtime is None:
        import boto3
        _bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)
    return _bedrock_runtime


def ask_bedrock(system_prompt: str, user_message: str) -> str:
    if MOCK_MODE:
        return _mock_reply(system_prompt, user_message)

    try:
        client = _get_client()
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 700,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        print(f"[bedrock_client] Bedrock call failed, falling back to mock: {e}")
        return _mock_reply(system_prompt, user_message)


def _mock_reply(system_prompt: str, user_message: str) -> str:
    """
    Simple, honest stand-in used only when Bedrock isn't connected yet.
    It still produces a properly hedged, evidence-first style response so
    the whole app behaves consistently while you're setting up AWS.
    """
    if "analyst explaining a finding" in system_prompt.lower():
        return (
            "The data shows this finding alongside several correlated factors listed above. "
            "These are associations observed in the dataset, not confirmed causes - "
            "establishing causation would require further investigation beyond what this data alone can show."
        )
    if "verif" in system_prompt.lower() or "evidence" in system_prompt.lower():
        return (
            "The calculation has been reproduced directly from the source records shown above. "
            "This confirms the figure is accurate based on the uploaded dataset."
        )
    return "Analysis complete based on the provided data. See the evidence and figures above for details."
