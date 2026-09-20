"""
Next Best Question Agent + "I Don't Know" Mode.

After a finding is investigated, this suggests what to look into next -
based on what other findings exist, not randomly generated. It also
implements the "insufficient evidence" behavior: if a user asks a
question that references something not present in the dataset (e.g.
"did the new CEO cause this?" when there's no CEO/leadership column),
it says so plainly instead of the AI making something up.
"""

import pandas as pd


def suggest_next_questions(all_findings: list, current_finding_id: str) -> list:
    suggestions = []
    for f in all_findings:
        if f["id"] != current_finding_id:
            suggestions.append(f"Why is {f['title'].lower()}?")

    generic = [
        "Are these patterns consistent with previous periods?",
        "Which customers are most affected by this change?",
    ]

    return (suggestions + generic)[:4]


def check_question_answerable(question: str, df_columns: list) -> dict:
    """
    Very simple keyword-based check: if the question references concepts
    that have no matching column in the dataset, flag it as
    unanswerable rather than letting an AI invent a confident-sounding
    non-answer.
    """
    question_lower = question.lower()
    columns_lower = [c.lower() for c in df_columns]

    unmappable_concepts = {
        "ceo": "leadership/management change",
        "competitor": "competitor activity",
        "weather": "weather conditions",
        "economy": "macroeconomic conditions",
        "marketing spend": "marketing/advertising spend",
        "advertising": "advertising data",
    }

    for keyword, concept in unmappable_concepts.items():
        if keyword in question_lower:
            has_related_column = any(keyword.split()[0] in c for c in columns_lower)
            if not has_related_column:
                return {
                    "answerable": False,
                    "reason": f"'{concept}' is not present as a column in this dataset.",
                    "available_columns": df_columns,
                }

    return {"answerable": True}
