# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Vision Judges vs Text-Only Judges
# MAGIC Builds three trace-based judges with `make_judge()` that *look at the images* in the trace via
# MAGIC `get_span_image` (MLflow >= 3.15), plus text-only counterparts, then runs both suites over the
# MAGIC agent traces and compares what each catches.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.15" openai
# MAGIC %restart_python

# COMMAND ----------

import mlflow, json, re
import pandas as pd
from typing import Literal
from mlflow.genai.judges import make_judge
from mlflow.entities import Trace

CATALOG, SCHEMA = "labuser_anubhav_awasthi", "mlflow_multimodal"
EXP_PATH = "/Users/anubhav.awasthi@databricks.com/Mlflow/claims_agent_demo/claims_agent_exp"
JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4-5"

print("MLflow version:", mlflow.__version__)
exp = mlflow.set_experiment(EXP_PATH)

# COMMAND ----------

# Load the exact traces produced by the agent run (02) and confirm image attachments are present
agent_runs = {r["claim_id"]: r["trace_id"] for r in
              [x.asDict() for x in spark.table(f"{CATALOG}.{SCHEMA}.agent_runs").collect()]}
traces = {cid: mlflow.get_trace(tid) for cid, tid in sorted(agent_runs.items())}
eval_data = pd.DataFrame({"trace": list(traces.values())})
print("traces loaded:", list(traces.keys()))

att = re.findall(r"mlflow-attachment://[^\"\\\s]+", traces["CLM-1004"].to_json())
print("attachment refs in CLM-1004 trace:", len(att), att[:2])
assert att, "no attachments found in trace — judges will have nothing to look at"

# COMMAND ----------

# ---------- VISION JUDGES (trace-based, can call get_span_image) ----------

damage_fidelity_vision = make_judge(
    name="damage_fidelity_vision",
    instructions=(
        "You are auditing an auto insurance FNOL agent. Analyze the {{ trace }}.\n\n"
        "The agent classified the claim's damage severity (see the process_claim span output: "
        "'severity' field, one of minor/moderate/severe). Your job is to verify this severity "
        "against the ACTUAL DAMAGE PHOTOS in the trace.\n\n"
        "Use the get_span_image tool to fetch and inspect EVERY damage photo image attached to the "
        "trace spans. The ingest_evidence span contains ALL photos the claimant submitted; the "
        "analyze_damage_photo spans contain only the ones the agent actually reviewed. Ignore the "
        "claim form document image.\n\n"
        "Severity rubric: minor = cosmetic (scratches, small dents, scuffs); moderate = significant "
        "panel damage, broken lights/glass, still drivable; severe = structural/frame damage, "
        "crushed panels, likely airbag deployment, not safely drivable.\n\n"
        "Adjacent-category tolerance: borderline calls one category apart (minor vs moderate, "
        "moderate vs severe) should 'pass'. Return 'fail' only when the photos CLEARLY contradict "
        "the agent — e.g., structural/severe damage classified as minor, or cosmetic damage "
        "classified as severe. In your rationale, describe what you SAW in the photos."
    ),
    feedback_value_type=Literal["pass", "fail"],
    model=JUDGE_MODEL,
)

extraction_accuracy_vision = make_judge(
    name="extraction_accuracy_vision",
    instructions=(
        "You are auditing document extraction in an insurance FNOL agent. Analyze the {{ trace }}.\n\n"
        "The agent extracted fields from a scanned claim form (see the extract_form_fields span "
        "output). Use the get_span_image tool to fetch the claim form document image (a scanned "
        "'AUTOMOBILE LOSS NOTICE' form) from the extract_form_fields or ingest_evidence span, "
        "read it carefully, and compare every extracted field value "
        "(policy_number, claimant_name, date_of_loss, vehicle, vin, estimate_amount) against what "
        "is actually printed on the form.\n\n"
        "Return 'pass' only if ALL extracted fields match the form exactly (ignoring case and "
        "whitespace). Return 'fail' if any field was misread, and name the incorrect fields and the "
        "correct values you can read on the form."
    ),
    feedback_value_type=Literal["pass", "fail"],
    model=JUDGE_MODEL,
)

process_check_vision = make_judge(
    name="process_completeness",
    instructions=(
        "You are auditing the PROCESS of an insurance FNOL agent, not its conclusion. "
        "Analyze the {{ trace }}.\n\n"
        "The process_claim span's inputs contain 'photo_paths' — the list of damage photos the "
        "claimant submitted. Count them. Then count how many analyze_damage_photo spans the agent "
        "actually executed.\n\n"
        "Return 'pass' if the agent analyzed EVERY submitted photo (one analyze_damage_photo span "
        "per photo path). Return 'fail' if any submitted photo was never analyzed, and say which "
        "photo(s) were skipped."
    ),
    feedback_value_type=Literal["pass", "fail"],
    model=JUDGE_MODEL,
)

# COMMAND ----------

# ---------- TEXT-ONLY JUDGES (see only inputs/outputs, the common pattern today) ----------

damage_fidelity_text = make_judge(
    name="damage_fidelity_text",
    instructions=(
        "You are auditing an auto insurance FNOL agent.\n\n"
        "Claim intake data: {{ inputs }}\n\n"
        "Agent output: {{ outputs }}\n\n"
        "The agent classified damage severity (minor/moderate/severe; rubric: minor = cosmetic; "
        "moderate = significant panel damage, drivable; severe = structural damage, not safely "
        "drivable). Note that you cannot see the damage photos; the agent may have relied on photo "
        "analysis that is not available to you. Assess whether the output is internally consistent "
        "and plausible. Return 'fail' only if the output is internally inconsistent or clearly "
        "contradicted by the text available to you; otherwise return 'pass'."
    ),
    feedback_value_type=Literal["pass", "fail"],
    model=JUDGE_MODEL,
)

extraction_accuracy_text = make_judge(
    name="extraction_accuracy_text",
    instructions=(
        "You are auditing document extraction in an insurance FNOL agent.\n\n"
        "Claim intake data: {{ inputs }}\n\n"
        "Agent output (includes extracted_fields from the scanned claim form): {{ outputs }}\n\n"
        "Assess whether the extracted field values (policy_number, claimant_name, date_of_loss, "
        "vehicle, vin, estimate_amount) are complete and plausible for this claim. Return 'pass' "
        "if extraction looks correct and complete, 'fail' otherwise."
    ),
    feedback_value_type=Literal["pass", "fail"],
    model=JUDGE_MODEL,
)

process_check_text = make_judge(
    name="process_completeness_text",
    instructions=(
        "You are auditing the process of an insurance FNOL agent.\n\n"
        "Claim intake data (includes the list of submitted photo_paths): {{ inputs }}\n\n"
        "Agent output: {{ outputs }}\n\n"
        "You cannot see the agent's internal steps — only its inputs and final output. Based on "
        "this text alone, is there explicit evidence that the agent ignored submitted evidence? "
        "Return 'fail' only if the output itself indicates evidence was skipped or ignored; "
        "otherwise return 'pass'."
    ),
    feedback_value_type=Literal["pass", "fail"],
    model=JUDGE_MODEL,
)

# COMMAND ----------

# Evaluate: same traces, two judge suites

with mlflow.start_run(run_name="eval_vision_judges"):
    res_vision = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[damage_fidelity_vision, extraction_accuracy_vision, process_check_vision],
    )
print("vision eval run:", res_vision.run_id)

with mlflow.start_run(run_name="eval_text_judges"):
    res_text = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[damage_fidelity_text, extraction_accuracy_text, process_check_text],
    )
print("text eval run:", res_text.run_id)

# COMMAND ----------

# Build the comparison table
def collect(run_id, names):
    rows = {}
    tdf = mlflow.search_traces(run_id=run_id, max_results=50)
    if "trace_id" in tdf.columns:
        tids = list(tdf["trace_id"])
    else:
        tids = [(Trace.from_json(t) if isinstance(t, str) else t).info.trace_id for t in tdf["trace"]]
    for tid in tids:
        trace = mlflow.get_trace(tid)
        js = trace.to_json()
        m = re.search(r"CLM-\d{4}", js)
        claim = m.group(0) if m else str(tid)[-8:]
        vals = rows.setdefault(claim, {})
        for a in (trace.info.assessments or []):
            if a.name in names:
                try:
                    vals[a.name] = (a.feedback.value, (a.rationale or "")[:400])
                except Exception:
                    vals[a.name] = (str(a), "")
    return rows

vision_names = ["damage_fidelity_vision", "extraction_accuracy_vision", "process_completeness"]
text_names = ["damage_fidelity_text", "extraction_accuracy_text", "process_completeness_text"]
v = collect(res_vision.run_id, vision_names)
t = collect(res_text.run_id, text_names)

gt = {r["claim_id"]: r for r in [x.asDict() for x in spark.table(f"{CATALOG}.{SCHEMA}.claims").collect()]}
agent = {r["claim_id"]: r for r in [x.asDict() for x in spark.table(f"{CATALOG}.{SCHEMA}.agent_runs").collect()]}

comp = []
for cid in sorted(set(list(v.keys()) + list(t.keys()))):
    row = {"claim_id": cid,
           "seeded_failure": gt.get(cid, {}).get("seeded_failure"),
           "agent_severity": agent.get(cid, {}).get("agent_severity"),
           "gt_severity": gt.get(cid, {}).get("gt_severity")}
    for n in text_names:
        row["TEXT:" + n.replace("_text", "")] = t.get(cid, {}).get(n, ("", ""))[0]
    for n in vision_names:
        row["VISION:" + n.replace("_vision", "")] = v.get(cid, {}).get(n, ("", ""))[0]
    comp.append(row)
comp_df = pd.DataFrame(comp)
display(comp_df)

# COMMAND ----------

# Show the vision judges' image-citing rationales for the seeded failures
rationale_rows = []
for cid in sorted(v.keys()):
    for n, (val, rat) in v.get(cid, {}).items():
        rationale_rows.append({"claim_id": cid, "judge": n, "verdict": str(val), "rationale": rat})
    print("=" * 100)
    print(cid, "| seeded:", gt.get(cid, {}).get("seeded_failure"))
    for n, (val, rat) in v.get(cid, {}).items():
        print(f"\n  [{n}] -> {val}\n  {rat}")

spark.createDataFrame(comp_df.astype(str)).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.judge_comparison")
spark.createDataFrame(pd.DataFrame(rationale_rows).astype(str)).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.judge_rationales")
print("DONE_EVAL_OK")
