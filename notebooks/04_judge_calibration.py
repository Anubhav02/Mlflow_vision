# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Who Audits the Judge? Calibrating the Damage Judge Against a Senior Adjuster
# MAGIC
# MAGIC Part 2 of the blog series. This notebook:
# MAGIC 1. **Expands the claims portfolio** (8 → 20) so alignment has enough labeled traces (MLflow requires ≥10).
# MAGIC 2. Runs the same buggy FNOL agent (from 02) over the new claims.
# MAGIC 3. Splits the severity judge into **perception** (frozen vision pass) + **judgment** (field-based,
# MAGIC    alignable) — because `judge.align()` currently supports only field-based judges, not `{{ trace }}` judges.
# MAGIC 4. Runs the baseline `damage_fidelity` judge over all traces and logs its verdicts as feedback.
# MAGIC 5. Creates a **label schema + labeling session** (Review Queue) assigned to a senior adjuster.
# MAGIC    ⏸️ **HUMAN-IN-THE-LOOP PAUSE**: the adjuster labels traces in the Review App.
# MAGIC 6. Measures **judge↔human agreement** (raw agreement, Cohen's κ, confusion matrix, disagreement quotes).
# MAGIC 7. **Aligns** the judge with `MemAlignOptimizer`, measures before/after agreement on a holdout,
# MAGIC    and **registers** the aligned judge.
# MAGIC
# MAGIC Run Part A (cells up to the PAUSE), label in the Review App, then run Part B.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.15" openai jinja2 tqdm "pydantic<2.13" dspy "litellm<1.80"
# MAGIC # note: pydantic/litellm pins work around a litellm incompatibility with pydantic 2.13 (Aug 2026)
# MAGIC %restart_python

# COMMAND ----------

import mlflow, json, base64, io, os, glob, hashlib, random, re
import pandas as pd
from typing import Literal
from openai import OpenAI
from PIL import Image

CATALOG, SCHEMA = "labuser_anubhav_awasthi", "mlflow_multimodal"
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/claims_data"
EXP_PATH = "/Users/anubhav.awasthi@databricks.com/Mlflow/claims_agent_demo/claims_agent_exp"
MODEL = "databricks-claude-sonnet-4-5"
JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4-5"
ADJUSTER = "anubhav.awasthi@databricks.com"   # plays the senior adjuster in the Review App

print("MLflow version:", mlflow.__version__)
exp = mlflow.set_experiment(EXP_PATH)

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
WORKSPACE_HOST = spark.conf.get("spark.databricks.workspaceUrl")
client = OpenAI(api_key=ctx.apiToken().get(), base_url=f"https://{WORKSPACE_HOST}/serving-endpoints")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A-1 — Expand the portfolio: 12 new claims from unused candidate photos

# COMMAND ----------

# Classify unused candidate photos to establish ground truth (same rubric as 01)
def classify(photo_path):
    b64 = base64.b64encode(open(photo_path, "rb").read()).decode()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "You are a senior auto insurance adjuster. Assess the vehicle damage in this photo. "
                "Respond with ONLY a JSON object: {\"severity\": \"minor|moderate|severe\", "
                "\"visible_damage\": \"<one sentence>\", \"confidence\": \"high|medium|low\", "
                "\"is_car_damage_photo\": true|false}. "
                "minor = cosmetic (scratches, small dents, scuffs); "
                "moderate = significant panel damage, broken lights/glass, drivable; "
                "severe = structural/frame damage, crushed panels, airbag deployment likely, not safely drivable."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=200,
    )
    txt = r.choices[0].message.content.strip()
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(txt)

# Skip photos already used by round-1 claims (compare content hashes)
used_md5 = {hashlib.md5(open(p, "rb").read()).hexdigest() for p in glob.glob(f"{VOL_ROOT}/photos/*.jpg")}
cands = [p for p in sorted(glob.glob(f"{VOL_ROOT}/candidates/*.jpg"))
         if hashlib.md5(open(p, "rb").read()).hexdigest() not in used_md5]
random.seed(43)
random.shuffle(cands)
print(f"{len(cands)} unused candidate photos")

NEED = {"minor": 5, "moderate": 3, "severe": 4}
labels, by_sev = {}, {"minor": [], "moderate": [], "severe": []}
from concurrent.futures import ThreadPoolExecutor

def classify_safe(p):
    try:
        return classify(p)
    except Exception as e:
        print("ERR", os.path.basename(p), e); return None

for i in range(0, len(cands), 12):
    if all(len(by_sev[s]) >= NEED[s] for s in NEED):
        break
    batch = cands[i:i + 12]
    with ThreadPoolExecutor(8) as ex:
        for p, v in zip(batch, ex.map(classify_safe, batch)):
            ok = v and v.get("is_car_damage_photo") and (
                v.get("confidence") == "high"
                or (v.get("severity") == "severe" and v.get("confidence") == "medium"))
            if ok and len(by_sev[v["severity"]]) < NEED[v["severity"]]:
                labels[p] = v; by_sev[v["severity"]].append(p)
                print(os.path.basename(p), "->", v["severity"], "|", v["visible_damage"])

print("distribution:", {k: len(v) for k, v in by_sev.items()})
assert all(len(by_sev[s]) >= NEED[s] for s in NEED), f"not enough photos: { {k: len(v) for k, v in by_sev.items()} }"

# COMMAND ----------

# 12 new claims. Two are seeded failures (downplayed severe damage phrased to trigger the express lane).
mi, mo, se = by_sev["minor"], by_sev["moderate"], by_sev["severe"]
new_claims = [
    dict(claim_id="CLM-2001", photos=[mi[0]], gt_severity="minor",
         description="Grocery run gone wrong — clipped a cart corral, scratched the rear door."),
    dict(claim_id="CLM-2002", photos=[mi[1]], gt_severity="minor",
         description="Low brick wall got my bumper while reversing out of a tight driveway."),
    dict(claim_id="CLM-2003", photos=[mi[2]], gt_severity="minor",
         description="Neighbor's kid's bike fell against the car, small dent and paint scuffs on the panel."),
    dict(claim_id="CLM-2004", photos=[mi[3]], gt_severity="minor",
         description="Gravel truck on the highway, chips and scratches along the hood edge."),
    dict(claim_id="CLM-2005", photos=[mi[4]], gt_severity="minor",
         description="Door dinged the garage frame pulling in, cosmetic scrape near the handle."),
    dict(claim_id="CLM-2006", photos=[mo[0]], gt_severity="moderate",
         description="Rear-ended at a stop sign, trunk won't close properly and the light is cracked."),
    dict(claim_id="CLM-2007", photos=[mo[1]], gt_severity="moderate",
         description="Sideswiped on the interstate, driver side panels dented, still drove it home."),
    dict(claim_id="CLM-2008", photos=[mo[2]], gt_severity="moderate",
         description="Hit a deer at 40 mph. Front grille and hood damaged, car still runs."),
    dict(claim_id="CLM-2009", photos=[se[0]], gt_severity="severe",
         description="T-boned at an intersection, doors crushed in, car was towed to the yard."),
    dict(claim_id="CLM-2010", photos=[se[1]], gt_severity="severe",
         description="Lost control on ice into a ditch, frame looks bent, airbags deployed."),
    # -- seeded failures: severe damage, claimant downplays with express-lane wording --
    dict(claim_id="CLM-2011", photos=[se[2]], gt_severity="severe",
         description="Minor fender bender at the mall, other driver barely touched me. "
                     "Just need a quick estimate, hoping for fast-track."),
    dict(claim_id="CLM-2012", photos=[se[3]], gt_severity="severe",
         description="Small fender bender on my street, mostly paint transfer I think. Quick fix please."),
]

import shutil
for c in new_claims:
    dsts, evid = [], []
    for j, src in enumerate(c["photos"]):
        dst = f"{VOL_ROOT}/photos/{c['claim_id']}_photo{j}.jpg"
        shutil.copy(src, dst); dsts.append(dst); evid.append(labels[src]["visible_damage"])
    c["photos"], c["photo_evidence"] = dsts, evid

# COMMAND ----------

# Generate claim forms (same generator as 01, compact version) + synthetic form fields
from PIL import ImageDraw, ImageFont, ImageFilter
import numpy as np

def _font(size, bold=False):
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
                 "/usr/share/fonts/truetype/liberation/LiberationSans-%s.ttf" % ("Bold" if bold else "Regular")]:
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()

NAMES = ["Priya Nair", "Marcus Webb", "Sofia Andersson", "James O'Rourke", "Keiko Tanaka", "Daniel Osei",
         "Ingrid Bauer", "Carlos Mendes", "Fatima Al-Sayed", "Peter Novak", "Grace Kim", "Omar Haddad"]
VEHICLES = ["2023 Toyota RAV4", "2021 Honda Accord", "2022 Chevrolet Equinox", "2020 Nissan Rogue",
            "2024 Kia Sportage", "2021 Ford Fusion", "2023 Hyundai Elantra", "2019 Jeep Cherokee",
            "2022 Subaru Forester", "2020 Mazda 3", "2023 VW Jetta", "2021 Toyota Highlander"]
EST = {"minor": (600, 1800), "moderate": (3500, 9000), "severe": (12000, 24000)}

random.seed(44)
def synth_fields(i, c):
    rng = random.Random(100 + i)
    vin = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(17))
    lo, hi = EST[c["gt_severity"]]
    return dict(policy_number=f"PA-{rng.randint(10000,99999)}-{rng.randint(10,99)}",
                claimant_name=NAMES[i], date_of_loss=f"11/{rng.randint(1,28):02d}/2026",
                vehicle=VEHICLES[i], vin=vin,
                estimate_amount=f"${rng.randint(lo, hi):,}.00",
                loss_description=c["description"][:130])

def make_form(claim_id, fields):
    W, H = 1100, 1450
    img = Image.new("RGB", (W, H), (252, 252, 250)); d = ImageDraw.Draw(img)
    d.rectangle([40, 40, W - 40, 130], outline=(60, 60, 60), width=3)
    d.text((60, 55), "AUTOMOBILE LOSS NOTICE", font=_font(34, True), fill=(30, 30, 30))
    d.text((60, 98), "FIRST NOTICE OF LOSS  |  FORM AL-2 (2026/01)", font=_font(16), fill=(90, 90, 90))
    d.text((W - 320, 60), "CLAIM NO.", font=_font(14), fill=(90, 90, 90))
    d.text((W - 320, 80), claim_id, font=_font(26, True), fill=(20, 20, 80))
    lab_f, val_f = _font(15), _font(24)
    rows = [[("POLICY NUMBER", fields["policy_number"]), ("DATE OF LOSS", fields["date_of_loss"])],
            [("INSURED / CLAIMANT NAME", fields["claimant_name"]), ("PHONE", "(555) 013-77" + claim_id[-2:])],
            [("VEHICLE (YEAR MAKE MODEL)", fields["vehicle"]), ("VIN", fields["vin"])],
            [("REPAIR ESTIMATE", fields["estimate_amount"]), ("DEDUCTIBLE", "$500.00")]]
    y = 170
    for row in rows:
        x, cw = 40, (W - 80) // len(row)
        for lab, val in row:
            d.rectangle([x, y, x + cw, y + 95], outline=(120, 120, 120), width=2)
            d.text((x + 12, y + 8), lab, font=lab_f, fill=(100, 100, 100))
            d.text((x + 12, y + 42), val, font=val_f, fill=(25, 25, 90)); x += cw
        y += 95
    d.rectangle([40, y, W - 40, y + 170], outline=(120, 120, 120), width=2)
    d.text((52, y + 8), "DESCRIPTION OF LOSS", font=lab_f, fill=(100, 100, 100))
    desc = fields["loss_description"]
    d.text((52, y + 42), desc[:70], font=val_f, fill=(25, 25, 90))
    if len(desc) > 70:
        d.text((52, y + 80), desc[70:140], font=val_f, fill=(25, 25, 90))
    y += 200
    d.text((52, y), "SIGNATURE OF INSURED:", font=lab_f, fill=(100, 100, 100))
    d.line([290, y + 18, 640, y + 18], fill=(25, 25, 90), width=2)
    nm = fields["claimant_name"].split()
    d.text((300, y - 8), nm[0] + " " + nm[-1], font=_font(26), fill=(25, 25, 90))
    img = img.rotate(0.7, expand=False, fillcolor=(245, 245, 243))
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.default_rng(7).integers(-9, 9, arr.shape)
    img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.55))
    out = f"{VOL_ROOT}/forms/{claim_id}_form.png"; img.save(out); return out

for i, c in enumerate(new_claims):
    c["form_fields"] = synth_fields(i, c)
    c["form_path"] = make_form(c["claim_id"], c["form_fields"])

route = {"minor": "fast_track", "moderate": "adjuster_review", "severe": "adjuster_review"}
rows = [dict(claim_id=c["claim_id"], description=c["description"], photo_paths=c["photos"],
             photo_evidence=c["photo_evidence"], form_path=c["form_path"],
             gt_severity=c["gt_severity"], gt_route=route[c["gt_severity"]],
             form_fields=json.dumps(c["form_fields"]),
             seeded_failure={"CLM-2011": "downplayed_severe_damage",
                             "CLM-2012": "downplayed_severe_damage"}.get(c["claim_id"]))
        for c in new_claims]
spark.createDataFrame(rows).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.claims_round2")
print("claims_round2 written:", len(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A-2 — Run the (still buggy) agent over the new claims

# COMMAND ----------

mlflow.openai.autolog()   # attachments on, same as notebook 02

def _img_content(path, max_px=None):
    im = Image.open(path).convert("RGB")
    if max_px: im.thumbnail((max_px, max_px))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"}}

@mlflow.trace(span_type="LLM", name="analyze_damage_photo")
def analyze_damage_photo(photo_path: str) -> str:
    r = client.chat.completions.create(model=MODEL, max_tokens=200,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "Briefly describe the vehicle damage visible in this claim photo in 2-3 sentences. "
                "End by categorizing it as exactly one of: cosmetic only (scratches, small dents, scuffs); "
                "significant panel damage (broken lights/glass, large dents, still drivable); or "
                "structural/severe (frame damage, crushed panels, not safely drivable).")},
            _img_content(photo_path)]}])
    return r.choices[0].message.content

@mlflow.trace(span_type="LLM", name="extract_form_fields")
def extract_form_fields(form_path: str) -> dict:
    r = client.chat.completions.create(model=MODEL, max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "Extract these fields from the scanned claim form image. Respond with ONLY JSON: "
                '{"claim_number": "", "policy_number": "", "claimant_name": "", "date_of_loss": "", '
                '"vehicle": "", "vin": "", "estimate_amount": "", "loss_description": ""}. '
                "If a value is unreadable, give your best guess.")},
            _img_content(form_path, max_px=768)]}])           # BUG(3) kept
    txt = r.choices[0].message.content
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])

@mlflow.trace(span_type="LLM", name="assess_severity")
def assess_severity(description: str, photo_analysis: str) -> dict:
    r = client.chat.completions.create(model=MODEL, max_tokens=150,
        messages=[{"role": "user", "content": (
            "You are an FNOL intake assistant. Classify claim severity as minor, moderate, or severe "
            "based on the available information.\n\n"
            "Rubric: minor = cosmetic (scratches, small dents, scuffs); moderate = significant panel "
            "damage, broken lights/glass, still drivable; severe = structural/frame damage, crushed "
            "panels, airbag deployment likely, not safely drivable.\n\n"
            f"CLAIMANT DESCRIPTION: {description}\n\nAUTOMATED PHOTO NOTES: {photo_analysis}\n\n"
            'Respond with ONLY JSON: {"severity": "minor|moderate|severe", "rationale": "<one sentence>"}')}])
    txt = r.choices[0].message.content
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])

@mlflow.trace(name="ingest_evidence")
def ingest_evidence(photo_paths: list, form_path: str) -> dict:
    return {"photos": [{"path": p, "image": _img_content(p)} for p in photo_paths],
            "form": {"path": form_path, "image": _img_content(form_path)}}

EXPRESS_LANE_KEYWORDS = ["fender bender"]                     # BUG(1) kept

@mlflow.trace(name="process_claim")
def process_claim(claim_id, description, photo_paths, form_path):
    ingest_evidence(photo_paths, form_path)
    fields = extract_form_fields(form_path)
    if any(k in description.lower() for k in EXPRESS_LANE_KEYWORDS):
        photo_analysis = "(photo review skipped — express lane triage)"
    else:
        photo_analysis = analyze_damage_photo(photo_paths[0])  # BUG(2) kept
    assessment = assess_severity(description, photo_analysis)
    return {"claim_id": claim_id, "severity": assessment["severity"],
            "routing": "fast_track" if assessment["severity"] == "minor" else "adjuster_review",
            "rationale": assessment["rationale"], "extracted_fields": fields}

r2 = [r.asDict() for r in spark.table(f"{CATALOG}.{SCHEMA}.claims_round2").orderBy("claim_id").collect()]
results, trace_ids = [], {}
for c in r2:
    out = process_claim(c["claim_id"], c["description"], list(c["photo_paths"]), c["form_path"])
    tid = mlflow.get_last_active_trace_id()
    trace_ids[c["claim_id"]] = tid; results.append(out)
    mlflow.log_expectation(trace_id=tid, name="expected_severity", value=c["gt_severity"])
    print(f'{c["claim_id"]}: agent={out["severity"]}/{out["routing"]}  gt={c["gt_severity"]}  trace={tid}')

summary2 = pd.DataFrame([{"claim_id": c["claim_id"], "gt_severity": c["gt_severity"],
                          "agent_severity": r["severity"], "agent_route": r["routing"],
                          "correct": c["gt_severity"] == r["severity"],
                          "seeded_failure": c["seeded_failure"], "trace_id": trace_ids[c["claim_id"]]}
                         for c, r in zip(r2, results)])
spark.createDataFrame(summary2).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.agent_runs_round2")
display(summary2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A-3 — Perception/judgment split: frozen vision pass + field-based `damage_fidelity` judge
# MAGIC
# MAGIC `judge.align()` currently supports only field-based judges (`{{ inputs }}`/`{{ outputs }}`), not
# MAGIC `{{ trace }}` judges. So we split Part 1's vision judge in two:
# MAGIC - **Perception** (frozen): a vision pass that turns every submitted photo into neutral damage notes.
# MAGIC - **Judgment** (alignable): a field-based judge that applies the severity policy to those notes.
# MAGIC
# MAGIC This split is also good governance: the calibratable "policy" layer is text-auditable, and each
# MAGIC layer can be regression-tested independently.

# COMMAND ----------

mlflow.openai.autolog(disable=True)   # perception calls should not create new traces

def perceive(photo_paths):
    """Frozen perception pass — one neutral damage note per submitted photo."""
    notes = []
    for p in photo_paths:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        r = client.chat.completions.create(model=MODEL, max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "Describe the vehicle damage visible in this photo in 2-3 factual sentences. "
                    "Mention affected components, deformation, broken glass, airbag deployment if visible, "
                    "and whether the vehicle looks drivable. Do NOT classify severity.")},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
        notes.append(r.choices[0].message.content)
    return notes

# All 20 claims: round 1 + round 2
all_claims = ([r.asDict() for r in spark.table(f"{CATALOG}.{SCHEMA}.claims").collect()] +
              [r.asDict() for r in spark.table(f"{CATALOG}.{SCHEMA}.claims_round2").collect()])
all_runs = ({r["claim_id"]: r for r in [x.asDict() for x in spark.table(f"{CATALOG}.{SCHEMA}.agent_runs").collect()]} |
            {r["claim_id"]: r for r in [x.asDict() for x in spark.table(f"{CATALOG}.{SCHEMA}.agent_runs_round2").collect()]})

judge_io = []
for c in sorted(all_claims, key=lambda x: x["claim_id"]):
    cid = c["claim_id"]
    notes = perceive(list(c["photo_paths"]))
    run = all_runs[cid]
    inputs = {"claim_id": cid, "claimant_description": c["description"],
              "photo_notes": notes, "num_photos_submitted": len(list(c["photo_paths"]))}
    outputs = {"severity": run["agent_severity"], "routing": run["agent_route"]}
    judge_io.append(dict(claim_id=cid, trace_id=run["trace_id"],
                         inputs=json.dumps(inputs), outputs=json.dumps(outputs)))
    print(cid, "| photos:", len(notes))

# Persist judge inputs so Part B can run after the labeling pause (fresh cluster, fresh state)
spark.createDataFrame(judge_io).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.judge_io")
print("judge_io written:", len(judge_io))

# COMMAND ----------

# Baseline field-based judge — the SAME rubric the Part 1 vision judge used
from mlflow.genai.judges import make_judge
from mlflow.entities import AssessmentSource, AssessmentSourceType

DAMAGE_JUDGE_INSTRUCTIONS = (
    "You are auditing an auto insurance FNOL agent's damage severity classification.\n\n"
    "Claim intake data and evidence notes: {{ inputs }}\n\n"
    "Agent output: {{ outputs }}\n\n"
    "The inputs contain the claimant's description and 'photo_notes' — factual damage notes written "
    "by a separate perception system that inspected EVERY photo the claimant submitted. Treat the "
    "photo notes as reliable evidence of what the photos show.\n\n"
    "Severity rubric: minor = cosmetic (scratches, small dents, scuffs); moderate = significant "
    "panel damage, broken lights/glass, still drivable; severe = structural/frame damage, crushed "
    "panels, likely airbag deployment, not safely drivable.\n\n"
    "Adjacent-category tolerance: borderline calls one category apart should 'pass'. Return 'fail' "
    "only when the photo notes CLEARLY contradict the agent's severity. In your rationale, cite the "
    "specific evidence you relied on."
)

damage_fidelity = make_judge(
    name="damage_fidelity",
    instructions=DAMAGE_JUDGE_INSTRUCTIONS,
    feedback_value_type=Literal["pass", "fail"],
    model=JUDGE_MODEL,
)
print("damage_fidelity judge defined")

# COMMAND ----------

# Run the baseline judge on every claim and attach its verdict to the AGENT trace.
# (Alignment requires judge + human assessments with the same name on the same trace.)
# ⚠️ Part A only — do NOT re-run this cell after the labeling pause (it would double-log feedback).
io_rows = [r.asDict() for r in spark.table(f"{CATALOG}.{SCHEMA}.judge_io").orderBy("claim_id").collect()]
for row in io_rows:
    fb = damage_fidelity(inputs=json.loads(row["inputs"]), outputs=json.loads(row["outputs"]))
    mlflow.log_feedback(
        trace_id=row["trace_id"], name="damage_fidelity", value=fb.value, rationale=fb.rationale,
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="damage_fidelity_baseline"),
    )
    print(f'{row["claim_id"]}: {fb.value} | {(fb.rationale or "")[:110]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A-4 — The Review Queue: label schema + labeling session for the senior adjuster

# COMMAND ----------

import mlflow.genai.labeling as labeling
import mlflow.genai.label_schemas as schemas
from mlflow.genai.label_schemas import InputCategorical

# Schema name MUST equal the judge name — that's what makes the labels usable for alignment.
schemas.create_label_schema(
    name="damage_fidelity",
    type="feedback",
    title="Is the agent's severity classification consistent with the submitted damage photos?",
    input=InputCategorical(options=["pass", "fail"]),
    instruction=(
        "You are auditing the agent, not the claimant. Open the ingest_evidence span to view every "
        "submitted photo. 'pass' = severity is defensible given the photos (borderline one-category "
        "calls are acceptable). 'fail' = the photos clearly contradict the severity. Always explain "
        "your reasoning in the comment — especially WHY you disagree with a plausible-looking call."),
    enable_comment=True,
    overwrite=True,
)

session = labeling.create_labeling_session(
    name="adjuster_audit_2026_08",
    assigned_users=[ADJUSTER],
    label_schemas=["damage_fidelity"],
)
print("Labeling session:", session.name, "| run_id:", session.mlflow_run_id)

# Add all 20 agent traces to the queue
tids = [r["trace_id"] for r in io_rows]
traces_for_queue = [mlflow.get_trace(t) for t in tids]
session.add_traces(traces_for_queue)
print(f"Added {len(traces_for_queue)} traces to the labeling session")

try:
    print("Review App URL:", session.url)
except AttributeError:
    print("Open the Review App from the experiment's 'Labeling sessions' tab")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⏸️ PAUSE — Adjuster labels in the Review App
# MAGIC
# MAGIC **What to do now** (wearing the senior-adjuster hat):
# MAGIC 1. Open the Review App link above (or Experiment → Labeling sessions → `adjuster_audit_2026_08`).
# MAGIC 2. For each claim: look at the photos (rendered inline — this is Part 1's `ingest_evidence`
# MAGIC    pattern paying off), read the agent's severity + routing, answer pass/fail, and **write a
# MAGIC    comment explaining your reasoning**. Rationales matter: MemAlign learns from them.
# MAGIC 3. Label with an adjuster's eye, not the judge's rubric. Where they genuinely differ, let them
# MAGIC    differ — the whole point is to measure and close that gap. Typical adjuster instincts that
# MAGIC    diverge from the drivability rubric:
# MAGIC    - **Repair economics over drivability**: glass and panel damage that photographs dramatically
# MAGIC      can still be a routine repair; conversely "cosmetic" damage on certain panels is not cheap.
# MAGIC    - **Over-calls are not free**: severity inflation routes fast-track claims to adjusters and
# MAGIC      inflates reserves — an audit fails a clear over-call even though it's "safe".
# MAGIC    - **Routing is part of the call**: severe-per-photos + fast_track is always a fail.
# MAGIC 4. Label ALL 20 traces (alignment needs ≥10, and a healthy pass/fail mix).
# MAGIC
# MAGIC Then run Part B below. Fresh cluster is fine — first re-run three cells: the `%pip` cell, the
# MAGIC setup/config cell, and the `damage_fidelity` judge-definition cell (NOT the baseline-run loop,
# MAGIC which would double-log feedback). Everything else Part B needs is reloaded from tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B-1 — Collect labels, measure judge↔human agreement

# COMMAND ----------

# (Part B is standalone-safe: re-run the setup cell at the top first if this is a fresh cluster.)
io_rows = [r.asDict() for r in spark.table(f"{CATALOG}.{SCHEMA}.judge_io").orderBy("claim_id").collect()]
by_tid = {r["trace_id"]: r for r in io_rows}

records = []
for row in io_rows:
    trace = mlflow.get_trace(row["trace_id"])
    judge_v = judge_r = human_v = human_r = None
    for a in (trace.info.assessments or []):
        if a.name != "damage_fidelity":
            continue
        st = str(a.source.source_type)
        if "HUMAN" in st:
            human_v, human_r = str(a.feedback.value), a.rationale
        elif "LLM_JUDGE" in st or "CODE" in st:
            judge_v, judge_r = str(a.feedback.value), a.rationale
    records.append(dict(claim_id=row["claim_id"], trace_id=row["trace_id"],
                        judge=judge_v, judge_rationale=judge_r,
                        human=human_v, human_rationale=human_r))

lab = pd.DataFrame(records)
labeled = lab.dropna(subset=["judge", "human"]).reset_index(drop=True)
print(f"{len(labeled)}/{len(lab)} traces have BOTH judge and human verdicts")
assert len(labeled) >= 10, "Alignment needs >=10 labeled traces — finish labeling in the Review App first"
assert labeled["human"].nunique() > 1, "Need a mix of pass and fail labels for alignment"

# Agreement metrics: raw agreement + Cohen's kappa + confusion matrix
agree = (labeled["judge"] == labeled["human"]).mean()
po = agree
p_j = labeled["judge"].value_counts(normalize=True)
p_h = labeled["human"].value_counts(normalize=True)
pe = sum(p_j.get(k, 0) * p_h.get(k, 0) for k in ["pass", "fail"])
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
conf = pd.crosstab(labeled["judge"], labeled["human"], rownames=["judge"], colnames=["adjuster"])

print(f"\nBASELINE agreement: {agree:.0%}   Cohen's kappa: {kappa:.2f}   disagreements: {(labeled['judge'] != labeled['human']).sum()}/{len(labeled)}")
print(conf)

print("\n" + "=" * 100 + "\nDISAGREEMENTS (blog money quotes):")
for _, r in labeled[labeled["judge"] != labeled["human"]].iterrows():
    print(f"\n--- {r['claim_id']}  judge={r['judge']}  adjuster={r['human']}")
    print(f"  JUDGE:    {(r['judge_rationale'] or '')[:300]}")
    print(f"  ADJUSTER: {(r['human_rationale'] or '(no comment)')[:300]}")

spark.createDataFrame(labeled.astype(str)).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.adjuster_labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B-2 — Align the judge with MemAlign, evaluate before/after on a holdout

# COMMAND ----------

from mlflow.genai.judges.optimizers import MemAlignOptimizer

# Deterministic split: ~70% align / 30% holdout, stratified by human verdict
labeled = labeled.sort_values("claim_id").reset_index(drop=True)
holdout_ids, align_ids = [], []
for verdict in ["pass", "fail"]:
    grp = list(labeled[labeled["human"] == verdict]["claim_id"])
    n_hold = max(1, round(len(grp) * 0.3))
    holdout_ids += grp[:n_hold]; align_ids += grp[n_hold:]
print(f"align on {len(align_ids)}: {align_ids}\nholdout {len(holdout_ids)}: {holdout_ids}")
assert len(align_ids) >= 10, f"only {len(align_ids)} alignment traces — label more, or shrink the holdout"

align_tids = set(labeled[labeled["claim_id"].isin(align_ids)]["trace_id"])
all_traces = mlflow.search_traces(experiment_ids=[exp.experiment_id], max_results=500, return_type="list")
align_traces = [t for t in all_traces if t.info.trace_id in align_tids]
print(f"resolved {len(align_traces)} alignment traces")

# MemAlign's default embedder is OpenAI — point it at a Databricks embedding endpoint instead
optimizer = MemAlignOptimizer(reflection_lm=JUDGE_MODEL,
                              embedding_model="databricks:/databricks-gte-large-en")
aligned_judge = damage_fidelity.align(align_traces, optimizer)
print("\nAligned judge ready. Instructions/memories:\n")
print(aligned_judge.instructions[:4000])

# COMMAND ----------

# Before/after: run both judges on the holdout, compare with the adjuster
def run_judge(judge, claim_ids):
    out = {}
    for row in io_rows:
        if row["claim_id"] in claim_ids:
            fb = judge(inputs=json.loads(row["inputs"]), outputs=json.loads(row["outputs"]))
            out[row["claim_id"]] = (str(fb.value), fb.rationale)
    return out

human_by_cid = dict(zip(labeled["claim_id"], labeled["human"]))
base_out = run_judge(damage_fidelity, set(holdout_ids))
alig_out = run_judge(aligned_judge, set(holdout_ids))

rows = []
for cid in sorted(holdout_ids):
    rows.append(dict(claim_id=cid, adjuster=human_by_cid[cid],
                     baseline=base_out[cid][0], aligned=alig_out[cid][0],
                     baseline_ok=base_out[cid][0] == human_by_cid[cid],
                     aligned_ok=alig_out[cid][0] == human_by_cid[cid],
                     aligned_rationale=(alig_out[cid][1] or "")[:400]))
res = pd.DataFrame(rows)
display(res)
print(f"\nHOLDOUT agreement — baseline: {res['baseline_ok'].mean():.0%}   aligned: {res['aligned_ok'].mean():.0%}")

# Also re-score the full labeled set for the blog's headline number
base_full = run_judge(damage_fidelity, set(labeled["claim_id"]))
alig_full = run_judge(aligned_judge, set(labeled["claim_id"]))
full_base = sum(base_full[c][0] == human_by_cid[c] for c in human_by_cid) / len(human_by_cid)
full_alig = sum(alig_full[c][0] == human_by_cid[c] for c in human_by_cid) / len(human_by_cid)
print(f"FULL-SET agreement — baseline: {full_base:.0%}   aligned: {full_alig:.0%}")

spark.createDataFrame(res.astype(str)).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.alignment_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B-3 — Register the aligned judge (the governance receipt)

# COMMAND ----------

registered = aligned_judge.register(experiment_id=exp.experiment_id)
print("Registered aligned judge:", getattr(registered, "name", "damage_fidelity"))

# Optional: promote the adjuster's labels into a reusable evaluation dataset
# session.sync(to_dataset=f"{CATALOG}.{SCHEMA}.claims_eval_dataset")   # syncs Expectations if collected

print("DONE_CALIBRATION_OK")
