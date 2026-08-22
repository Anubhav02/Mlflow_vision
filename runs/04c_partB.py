# Databricks notebook source
# MAGIC %md
# MAGIC # 04c — Calibration Part B (job driver): agreement, MemAlign alignment, before/after, register

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.15" openai jinja2 tqdm "pydantic<2.13" dspy "litellm<1.80"
# MAGIC %restart_python

# COMMAND ----------

import mlflow, json
import pandas as pd
from typing import Literal
from mlflow.genai.judges import make_judge

CATALOG, SCHEMA = "labuser_anubhav_awasthi", "mlflow_multimodal"
EXP_PATH = "/Users/anubhav.awasthi@databricks.com/Mlflow/claims_agent_demo/claims_agent_exp"
JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4-5"
exp = mlflow.set_experiment(EXP_PATH)
print("mlflow", mlflow.__version__)

DAMAGE_JUDGE_INSTRUCTIONS = (
    "You are auditing an auto insurance FNOL agent's damage severity classification.\n\n"
    "Claim intake data and evidence notes: {{ inputs }}\n\n"
    "Agent output: {{ outputs }}\n\n"
    "The inputs contain the claimant's description and 'photo_notes' - factual damage notes written "
    "by a separate perception system that inspected EVERY photo the claimant submitted. Treat the "
    "photo notes as reliable evidence of what the photos show.\n\n"
    "Severity rubric: minor = cosmetic (scratches, small dents, scuffs); moderate = significant "
    "panel damage, broken lights/glass, still drivable; severe = structural/frame damage, crushed "
    "panels, likely airbag deployment, not safely drivable.\n\n"
    "Adjacent-category tolerance: borderline calls one category apart should 'pass'. Return 'fail' "
    "only when the photo notes CLEARLY contradict the agent's severity. In your rationale, cite the "
    "specific evidence you relied on."
)
damage_fidelity = make_judge(name="damage_fidelity", instructions=DAMAGE_JUDGE_INSTRUCTIONS,
                             feedback_value_type=Literal["pass", "fail"], model=JUDGE_MODEL)

# COMMAND ----------

# B-1: collect both verdicts per trace, compute agreement
io_rows = [r.asDict() for r in spark.table(f"{CATALOG}.{SCHEMA}.judge_io").orderBy("claim_id").collect()]

records = []
for row in io_rows:
    trace = mlflow.get_trace(row["trace_id"])
    judge_v = judge_r = human_v = human_r = None
    for a in (trace.info.assessments or []):
        if a.name != "damage_fidelity":
            continue
        st = str(a.source.source_type)
        try:
            val = str(a.feedback.value)
        except Exception:
            continue
        if "HUMAN" in st:
            human_v, human_r = val, a.rationale
        elif "LLM_JUDGE" in st or "CODE" in st:
            judge_v, judge_r = val, a.rationale
    records.append(dict(claim_id=row["claim_id"], trace_id=row["trace_id"],
                        judge=judge_v, judge_rationale=judge_r,
                        human=human_v, human_rationale=human_r))

lab = pd.DataFrame(records)
labeled = lab.dropna(subset=["judge", "human"]).reset_index(drop=True)
print(f"{len(labeled)}/{len(lab)} traces have BOTH judge and human verdicts")
assert len(labeled) >= 10, "need >=10 labeled traces"
assert labeled["human"].nunique() > 1, "need a mix of pass/fail labels"

agree = float((labeled["judge"] == labeled["human"]).mean())
p_j = labeled["judge"].value_counts(normalize=True)
p_h = labeled["human"].value_counts(normalize=True)
pe = sum(float(p_j.get(k, 0)) * float(p_h.get(k, 0)) for k in ["pass", "fail"])
kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
conf = pd.crosstab(labeled["judge"], labeled["human"], rownames=["judge"], colnames=["adjuster"])
print(f"BASELINE agreement {agree:.0%} kappa {kappa:.2f}")
print(conf)

disagreements = [dict(claim_id=r["claim_id"], judge=r["judge"], adjuster=r["human"],
                      judge_rationale=(r["judge_rationale"] or "")[:400],
                      adjuster_rationale=(r["human_rationale"] or "")[:400])
                 for _, r in labeled[labeled["judge"] != labeled["human"]].iterrows()]
spark.createDataFrame(labeled.astype(str)).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.adjuster_labels")

# COMMAND ----------

# B-2: stratified split, MemAlign, before/after
from mlflow.genai.judges.optimizers import MemAlignOptimizer

labeled = labeled.sort_values("claim_id").reset_index(drop=True)
holdout_ids, align_ids = [], []
for verdict in ["pass", "fail"]:
    grp = list(labeled[labeled["human"] == verdict]["claim_id"])
    n_hold = max(1, round(len(grp) * 0.3)) if grp else 0
    holdout_ids += grp[:n_hold]; align_ids += grp[n_hold:]
print("align:", align_ids, "\nholdout:", holdout_ids)
assert len(align_ids) >= 10, f"only {len(align_ids)} alignment traces"

align_tids = set(labeled[labeled["claim_id"].isin(align_ids)]["trace_id"])
all_traces = mlflow.search_traces(experiment_ids=[exp.experiment_id], max_results=1000, return_type="list")
align_traces = [t for t in all_traces if t.info.trace_id in align_tids]
print("resolved", len(align_traces), "alignment traces")
assert len(align_traces) == len(align_tids)

import inspect
sig = inspect.signature(MemAlignOptimizer.__init__)
print("MemAlignOptimizer params:", list(sig.parameters))
emb_param = next((p for p in sig.parameters if "embed" in p.lower()), None)

aligned_judge, optimizer_used = None, None
attempts = []
if emb_param:
    attempts += [{ "reflection_lm": JUDGE_MODEL, emb_param: "databricks:/databricks-gte-large-en" },
                 { "reflection_lm": JUDGE_MODEL, emb_param: "databricks/databricks-gte-large-en" }]
attempts += [{ "reflection_lm": JUDGE_MODEL }]

for kw in attempts:
    try:
        optimizer = MemAlignOptimizer(**kw)
        aligned_judge = damage_fidelity.align(align_traces, optimizer)
        optimizer_used = f"MemAlign({kw})"
        break
    except Exception as e:
        print("MemAlign attempt failed:", kw, "->", str(e)[:250])

if aligned_judge is None:
    from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer
    optimizer = SIMBAAlignmentOptimizer(model=JUDGE_MODEL)
    aligned_judge = damage_fidelity.align(align_traces, optimizer)
    optimizer_used = "SIMBA"

print("OPTIMIZER USED:", optimizer_used)
aligned_instructions = str(getattr(aligned_judge, "instructions", ""))[:8000]
print(aligned_instructions[:2000])

# COMMAND ----------

def run_judge(judge, claim_ids):
    out = {}
    for row in io_rows:
        if row["claim_id"] in claim_ids:
            fb = judge(inputs=json.loads(row["inputs"]), outputs=json.loads(row["outputs"]))
            out[row["claim_id"]] = (str(fb.value), fb.rationale or "")
    return out

human_by_cid = dict(zip(labeled["claim_id"], labeled["human"]))
all_ids = set(labeled["claim_id"])
base_out = run_judge(damage_fidelity, all_ids)      # fresh baseline pass (same judge as Part A)
alig_out = run_judge(aligned_judge, all_ids)

rows = []
for cid in sorted(all_ids):
    rows.append(dict(claim_id=cid, split=("holdout" if cid in holdout_ids else "align"),
                     adjuster=human_by_cid[cid],
                     baseline=base_out[cid][0], aligned=alig_out[cid][0],
                     baseline_ok=base_out[cid][0] == human_by_cid[cid],
                     aligned_ok=alig_out[cid][0] == human_by_cid[cid],
                     aligned_rationale=alig_out[cid][1][:400]))
res = pd.DataFrame(rows)
hold = res[res["split"] == "holdout"]
metrics = {
    "n_labeled": int(len(labeled)),
    "baseline_agreement_stored": agree,
    "baseline_kappa": kappa,
    "confusion": {f"judge={a}|adjuster={b}": int(conf.loc[a, b]) if (a in conf.index and b in conf.columns) else 0
                  for a in ["pass", "fail"] for b in ["pass", "fail"]},
    "holdout_ids": sorted(holdout_ids), "align_ids": sorted(align_ids),
    "holdout_baseline_agreement": float(hold["baseline_ok"].mean()),
    "holdout_aligned_agreement": float(hold["aligned_ok"].mean()),
    "full_baseline_agreement": float(res["baseline_ok"].mean()),
    "full_aligned_agreement": float(res["aligned_ok"].mean()),
}
print(json.dumps(metrics, indent=2))
spark.createDataFrame(res.astype(str)).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.alignment_results")

# COMMAND ----------

reg_info = None
try:
    registered = aligned_judge.register(experiment_id=exp.experiment_id)
    reg_info = str(getattr(registered, "name", "damage_fidelity"))
except Exception as e:
    reg_info = "REGISTER_FAILED: " + str(e)[:300]
print("registered:", reg_info)

payload = {
    "metrics": metrics,
    "disagreements": disagreements,
    "per_claim": rows,
    "aligned_instructions": aligned_instructions,
    "registered": reg_info,
    "optimizer_used": optimizer_used,
}
dbutils.notebook.exit(json.dumps(payload))
