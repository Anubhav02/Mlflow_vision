# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — FNOL Claims Agent with Multimodal Tracing
# MAGIC A first-notice-of-loss intake agent: analyzes damage photos, extracts fields from the scanned
# MAGIC claim form, produces a severity estimate and a routing decision. Instrumented with MLflow
# MAGIC autologging so photos and form scans land in traces as attachments (MLflow >= 3.15).
# MAGIC
# MAGIC The agent contains three *realistic* production bugs (deliberately kept for the eval demo):
# MAGIC 1. "Express lane": claims whose description sounds like a minor fender bender skip photo
# MAGIC    review entirely and are classified from the claimant's own words (cost-saving triage).
# MAGIC 2. Only the FIRST photo of each claim is analyzed (cost-saving shortcut).
# MAGIC 3. The claim form is downscaled before extraction (another cost-saving shortcut).
# MAGIC
# MAGIC All submitted evidence is still stored on the trace by `ingest_evidence` — so a vision-capable
# MAGIC judge can see photos the agent itself never opened.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.15" openai
# MAGIC %restart_python

# COMMAND ----------

import mlflow, json, base64, io
from openai import OpenAI
from PIL import Image

CATALOG, SCHEMA = "labuser_anubhav_awasthi", "mlflow_multimodal"
EXP_PATH = "/Users/anubhav.awasthi@databricks.com/Mlflow/claims_agent_demo/claims_agent_exp"
MODEL = "databricks-claude-sonnet-4-5"

print("MLflow version:", mlflow.__version__)
mlflow.set_experiment(EXP_PATH)
mlflow.openai.autolog()

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
WORKSPACE_HOST = spark.conf.get("spark.databricks.workspaceUrl")
client = OpenAI(
    api_key=ctx.apiToken().get(),
    base_url=f"https://{WORKSPACE_HOST}/serving-endpoints",
)

# COMMAND ----------

def _img_content(path, max_px=None):
    im = Image.open(path).convert("RGB")
    if max_px:
        im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


@mlflow.trace(span_type="LLM", name="analyze_damage_photo")
def analyze_damage_photo(photo_path: str) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "Briefly describe the vehicle damage visible in this claim photo in 2-3 sentences. "
                "End by categorizing it as exactly one of: cosmetic only (scratches, small dents, scuffs); "
                "significant panel damage (broken lights/glass, large dents, still drivable); or "
                "structural/severe (frame damage, crushed panels, not safely drivable)."
            )},
            _img_content(photo_path),
        ]}],
        max_tokens=200,
    )
    return r.choices[0].message.content


@mlflow.trace(span_type="LLM", name="extract_form_fields")
def extract_form_fields(form_path: str) -> dict:
    # BUG(3): form downscaled to 768px to save tokens — fine for sparse forms, lossy for dense ones
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "Extract these fields from the scanned claim form image. Respond with ONLY JSON: "
                '{"claim_number": "", "policy_number": "", "claimant_name": "", "date_of_loss": "", '
                '"vehicle": "", "vin": "", "estimate_amount": "", "loss_description": ""}. '
                "If a value is unreadable, give your best guess."
            )},
            _img_content(form_path, max_px=768),
        ]}],
        max_tokens=300,
    )
    txt = r.choices[0].message.content
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(txt)


@mlflow.trace(span_type="LLM", name="assess_severity")
def assess_severity(description: str, photo_analysis: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": (
            "You are an FNOL intake assistant. Classify claim severity as minor, moderate, or severe "
            "based on the available information.\n\n"
            "Rubric: minor = cosmetic (scratches, small dents, scuffs); moderate = significant panel "
            "damage, broken lights/glass, still drivable; severe = structural/frame damage, crushed "
            "panels, airbag deployment likely, not safely drivable.\n\n"
            f"CLAIMANT DESCRIPTION: {description}\n\n"
            f"AUTOMATED PHOTO NOTES: {photo_analysis}\n\n"
            'Respond with ONLY JSON: {"severity": "minor|moderate|severe", "rationale": "<one sentence>"}'
        )}],
        max_tokens=150,
    )
    txt = r.choices[0].message.content
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(txt)


@mlflow.trace(name="ingest_evidence")
def ingest_evidence(photo_paths: list, form_path: str) -> dict:
    # Store all submitted evidence on the trace (auto-extracted into attachments, rendered in the UI)
    return {
        "photos": [{"path": p, "image": _img_content(p)} for p in photo_paths],
        "form": {"path": form_path, "image": _img_content(form_path)},
    }


# BUG(1): "express lane" — parking-lot fender benders are high-volume, low-value claims,
# so intake skips photo review for them and trusts the claimant's description.
EXPRESS_LANE_KEYWORDS = ["fender bender"]


@mlflow.trace(name="process_claim")
def process_claim(claim_id: str, description: str, photo_paths: list, form_path: str) -> dict:
    ingest_evidence(photo_paths, form_path)
    fields = extract_form_fields(form_path)
    if any(k in description.lower() for k in EXPRESS_LANE_KEYWORDS):
        photo_analysis = "(photo review skipped — express lane triage)"
    else:
        # BUG(2): only the first photo is analyzed
        photo_analysis = analyze_damage_photo(photo_paths[0])
    assessment = assess_severity(description, photo_analysis)
    route = "fast_track" if assessment["severity"] == "minor" else "adjuster_review"
    return {
        "claim_id": claim_id,
        "severity": assessment["severity"],
        "routing": route,
        "rationale": assessment["rationale"],
        "extracted_fields": fields,
    }

# COMMAND ----------

# Run the agent over all 8 claims
claims = [r.asDict() for r in spark.table(f"{CATALOG}.{SCHEMA}.claims").orderBy("claim_id").collect()]
results, trace_ids = [], {}

for c in claims:
    out = process_claim(c["claim_id"], c["description"], list(c["photo_paths"]), c["form_path"])
    tid = mlflow.get_last_active_trace_id()
    trace_ids[c["claim_id"]] = tid
    results.append(out)
    print(f'{c["claim_id"]}: agent={out["severity"]}/{out["routing"]}  gt={c["gt_severity"]}/{c["gt_route"]}  trace={tid}')

# COMMAND ----------

# Attach ground-truth expectations to each trace (used by evaluation + review workflows)
for c in claims:
    tid = trace_ids[c["claim_id"]]
    mlflow.log_expectation(trace_id=tid, name="expected_severity", value=c["gt_severity"])
    mlflow.log_expectation(trace_id=tid, name="expected_route", value=c["gt_route"])
    mlflow.log_expectation(trace_id=tid, name="expected_form_fields", value=json.loads(c["form_fields"]))

import pandas as pd
summary = pd.DataFrame([{
    "claim_id": c["claim_id"], "gt_severity": c["gt_severity"],
    "agent_severity": r["severity"], "agent_route": r["routing"],
    "correct": c["gt_severity"] == r["severity"],
    "seeded_failure": c["seeded_failure"], "trace_id": trace_ids[c["claim_id"]],
} for c, r in zip(claims, results)])
display(summary)

spark.createDataFrame(summary).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.agent_runs")
print("DONE_AGENT_OK")
