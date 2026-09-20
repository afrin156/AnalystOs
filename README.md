# AnalystOS

An agentic AI data analyst that autonomously discovers patterns in your
data, investigates why they happened, simulates decisions, and verifies
every conclusion against the source data before presenting it.

**Core principle: no evidence, no strong conclusion.** Every number this
app shows you is a real, reproducible Pandas calculation - Bedrock is
used only to phrase explanations in plain language, never to invent
statistics.

## How it works (Supervisor + specialist agents)

```
Upload CSV
    |
    v
Data Understanding Agent   --> infers what each column means
    |
    v
Data Health Agent          --> real checks: missing values, duplicates,
    |                           invalid dates, negative values, outliers
    v
Autonomous Discovery Agent --> finds significant patterns without being asked
    |
    v
For each finding:
    Investigation Agent    --> finds correlated factors (never claims causation)
    Verification Agent     --> re-derives the numbers independently, proves they match
    Business Impact Agent  --> translates into rupee terms + risk level
    Next Questions Agent   --> suggests what to investigate next

Separately, on demand:
    Simulation Agent       --> real linear regression for "what if" scenarios
    Memory Agent           --> compares this run against your history
    "I Don't Know" check   --> flags questions the dataset can't answer
```

`agents/supervisor.py` is the orchestrator that runs this sequence -
matching the Supervisor + specialist agent architecture pattern AWS
documents for Bedrock agent workloads.

## Project structure

```
analystos/
├── app.py                     # Flask app - all API endpoints
├── lambda_handler.py          # Wraps app.py for AWS Lambda deployment
├── bedrock_client.py          # Bedrock wrapper, mock mode for local testing
├── template.yaml              # AWS SAM deployment config
├── agents/
│   ├── supervisor.py          # Orchestrates the full pipeline
│   ├── data_understanding.py  # Infers column roles
│   ├── data_health.py         # Missing/duplicate/outlier checks
│   ├── discovery.py           # Autonomous pattern discovery
│   ├── investigation.py       # WHY - finds correlated factors
│   ├── verification.py        # Re-derives numbers, proves they match
│   ├── simulation.py          # What-if scenarios via real regression
│   ├── business_impact.py     # Translates findings to business language
│   ├── next_questions.py      # Suggests follow-ups + "insufficient evidence" mode
│   └── memory.py               # Local history, compares runs over time
├── sample_data/
│   └── company_sales.csv      # Realistic sample with a real discoverable pattern
├── templates/
│   └── index.html             # Analyst workspace dashboard UI
└── requirements.txt
```

## Running it locally (no AWS needed yet)

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`, click **"Load sample dataset"** to see it
work immediately, or upload your own CSV. Runs in mock mode by default
(`ANALYSTOS_MOCK=true`) - all the Pandas-based analysis (health,
discovery, verification, simulation) is fully real either way; only the
plain-language explanation text is a mock stand-in until Bedrock is
connected.

## Switching to real Bedrock

```bash
export ANALYSTOS_MOCK=false
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python app.py
```

If the Bedrock call fails for any reason, it automatically falls back
to the mock explanation instead of crashing - a safety net for demo day.

## Deploying (Ship It track)

```bash
sam build
sam deploy --guided
```

This deploys behind Lambda + API Gateway and creates an S3 bucket for
uploads. Gives you a live URL to submit.

## What's real right now, and what's honestly simplified

**Fully real, tested, and working:**
- Data understanding, health checks, autonomous discovery - all genuine
  Pandas calculations, not AI-generated
- Investigation - finds real correlated factors via groupby comparisons
- Verification - independently re-derives every claimed number
- Simulation - real linear regression with honest confidence scores
  based on actual R²
- Memory - real local history, compares each run against past runs
- "I don't know" mode - flags genuinely unanswerable questions

**Simplified for the time available - documented honestly, not hidden:**
- **Memory is a local JSON file**, not DynamoDB. Swapping this is
  straightforward: replace the read/write in `agents/memory.py` with
  DynamoDB calls (same pattern as the AWS Bedrock client's structure).
- **No Streamlit / no separate multi-agent AgentCore orchestration** -
  the Supervisor pattern is implemented directly in Python
  (`supervisor.py`), which is functionally the same orchestration
  concept AgentCore provides, without the additional AWS service to
  configure under time pressure.
- **Discovery checks two dimensions** (time period, one categorical/geo
  column) rather than an exhaustive combinatorial search - deliberately
  scoped so it's fast and explainable rather than a black box.
- **No file storage between sessions** - each upload analyzes in
  memory. For production, uploads would go to S3 (the SAM template
  already provisions the bucket for this).

## Bugs found and fixed while building this (for transparency)

- Column type detection initially missed categorical columns because
  pandas' newer string dtype isn't exactly `"object"` - fixed to check
  both.
- The injected sample-data pattern was originally too narrow (a
  three-way intersection) to be discoverable by any single-dimension
  check - regenerated with a broader, realistic pattern.
- The WHY/Investigation agent initially only handled segment-type
  findings, leaving period-based findings with zero contributing
  factors - added a dedicated period-change investigation path.
- Several numeric results were numpy types (`np.float64`, `np.True_`)
  which aren't JSON-serializable by Flask - cast to native Python types
  at the source.

## For your submission

**The demo sentence:** "I didn't ask AnalystOS what to analyze. I
uploaded the data, and it found the problem itself." Click "Load sample
dataset," show the auto-discovered findings, click one to show
contributing factors + verification + business impact, then run a
what-if simulation. That's your 3-minute demo.
