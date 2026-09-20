"""
Business Impact Agent.

Translates a statistical finding into plain business language - what's
the actual rupee impact, who's affected, how urgent is it. A manager
shouldn't need to understand percentages and standard deviations to
know if something matters.
"""

import pandas as pd


def translate_impact(df: pd.DataFrame, finding: dict, key_columns: dict) -> dict:
    metric_col = finding["metric"]

    if finding["id"] == "period_change":
        evidence = finding["evidence"]
        rupee_impact = abs(evidence["second_half_total"] - evidence["first_half_total"])
        risk = "HIGH" if finding["magnitude"] > 20 else "MEDIUM" if finding["magnitude"] > 10 else "LOW"
        return {
            "revenue_impact": round(rupee_impact, 2),
            "affected_scope": "entire dataset, second half of period",
            "risk_level": risk,
        }

    if finding["id"].startswith("segment_deviation"):
        segment_col = finding["segment_column"]
        segment_val = finding["segment_value"]
        subset = df[df[segment_col] == segment_val]

        evidence = finding["evidence"]
        rupee_gap_per_record = abs(evidence["overall_average"] - evidence["segment_average"])
        total_impact = rupee_gap_per_record * evidence["segment_record_count"]

        id_col = next((c for c in df.columns if "id" in c.lower()), None)
        affected_customers = subset[id_col].nunique() if id_col else evidence["segment_record_count"]

        risk = "HIGH" if finding["magnitude"] > 30 else "MEDIUM" if finding["magnitude"] > 15 else "LOW"

        return {
            "revenue_impact": round(total_impact, 2),
            "affected_segment": f"{segment_col} = {segment_val}",
            "affected_records": affected_customers,
            "risk_level": risk,
        }

    return {"revenue_impact": None, "risk_level": "UNKNOWN"}
