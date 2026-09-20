"""
Supervisor Agent.

Orchestrates the full pipeline in order, exactly matching the
architecture: Data Understanding -> Data Health -> Discovery ->
(on demand) Investigation / Simulation / Verification -> Business
Impact -> Next Questions. Each step is a separate, testable module -
this file just calls them in the right sequence and passes state
between them.
"""

import pandas as pd

from agents.data_understanding import understand, find_key_columns
from agents.data_health import check_health
from agents.discovery import discover
from agents.investigation import investigate
from agents.verification import verify
from agents.business_impact import translate_impact
from agents.next_questions import suggest_next_questions
from agents.memory import record_run


def run_full_analysis(df: pd.DataFrame, dataset_name: str) -> dict:
    understanding = understand(df)
    key_columns = find_key_columns(df, understanding)

    health = check_health(df, key_columns)
    findings = discover(df, key_columns)

    enriched_findings = []
    for finding in findings:
        investigation_result = investigate(df, finding, key_columns)
        verification_result = verify(df, finding)
        impact_result = translate_impact(df, finding, key_columns)

        enriched_findings.append({
            **finding,
            "investigation": investigation_result,
            "verification": verification_result,
            "business_impact": impact_result,
        })

    for f in enriched_findings:
        f["next_questions"] = suggest_next_questions(findings, f["id"])

    metric_col = key_columns.get("primary_metric")
    metric_total = float(df[metric_col].sum()) if metric_col and metric_col in df.columns else 0
    top_finding_title = enriched_findings[0]["title"] if enriched_findings else "No significant findings"
    memory_comparison = record_run(dataset_name, metric_total, top_finding_title)

    return {
        "understanding": understanding,
        "key_columns": key_columns,
        "health": health,
        "findings": enriched_findings,
        "memory_comparison": memory_comparison,
    }
