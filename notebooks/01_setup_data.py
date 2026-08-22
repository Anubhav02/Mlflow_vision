# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Claims Demo: Data Setup
# MAGIC Downloads the CC0 vehicle-damage dataset (Humans in the Loop, via Kaggle), classifies candidate
# MAGIC photos with a vision model to establish ground truth, generates ACORD-style claim-form images,
# MAGIC and seeds 8 FNOL claims (incl. deliberate failure cases) into `labuser_anubhav_awasthi.mlflow_multimodal`.

# COMMAND ----------

# MAGIC %pip install --quiet kagglehub pillow openai
# MAGIC %restart_python

# COMMAND ----------

CATALOG = "labuser_anubhav_awasthi"
SCHEMA = "mlflow_multimodal"
VOLUME = "claims_data"
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

import os
for d in ["candidates", "photos", "forms"]:
    os.makedirs(f"{VOL_ROOT}/{d}", exist_ok=True)
print("Volume ready:", VOL_ROOT)

# COMMAND ----------

# Download CC0 dataset (Humans in the Loop — Car Parts and Car Damages, public domain)
import kagglehub, glob, random, shutil

path = kagglehub.dataset_download("humansintheloop/car-parts-and-car-damages")
print("Downloaded to:", path)

imgs = [p for p in glob.glob(path + "/**/*", recursive=True)
        if p.lower().endswith((".jpg", ".jpeg", ".png")) and "damage" in p.lower()]
if not imgs:  # fall back to any images in the archive
    imgs = [p for p in glob.glob(path + "/**/*", recursive=True) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"Found {len(imgs)} images")
random.seed(42)
random.shuffle(imgs)
candidates = imgs  # scan the whole set; classification loop below stops early once quotas are met

from PIL import Image
kept = []
for i, src in enumerate(candidates):
    try:
        im = Image.open(src).convert("RGB")
        if im.width < 300 or im.height < 300:
            continue
        im.thumbnail((1024, 1024))
        dst = f"{VOL_ROOT}/candidates/cand_{i:03d}.jpg"
        im.save(dst, "JPEG", quality=90)
        kept.append(dst)
    except Exception as e:
        print("skip", src, e)
print(f"Staged {len(kept)} candidate photos")

# COMMAND ----------

# Classify candidates with a vision model to establish ground-truth severity
import base64, json
from openai import OpenAI

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
WORKSPACE_HOST = spark.conf.get("spark.databricks.workspaceUrl")
client = OpenAI(
    api_key=ctx.apiToken().get(),
    base_url=f"https://{WORKSPACE_HOST}/serving-endpoints",
)
VISION_MODEL = "databricks-claude-sonnet-4-5"

def classify(photo_path):
    b64 = base64.b64encode(open(photo_path, "rb").read()).decode()
    r = client.chat.completions.create(
        model=VISION_MODEL,
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

from concurrent.futures import ThreadPoolExecutor

def classify_safe(p):
    try:
        return classify(p)
    except Exception as e:
        print("ERR", os.path.basename(p), e)
        return None

labels = {}

def counts():
    return {s: sum(1 for v in labels.values() if v["severity"] == s) for s in ["minor", "moderate", "severe"]}

for i in range(0, len(kept), 12):
    c = counts()
    if c["minor"] >= 5 and c["severe"] >= 3 and c["moderate"] >= 3:
        break
    batch = kept[i:i + 12]
    with ThreadPoolExecutor(8) as ex:
        for p, v in zip(batch, ex.map(classify_safe, batch)):
            ok = v and v.get("is_car_damage_photo") and (
                v.get("confidence") == "high"
                or (v.get("severity") == "severe" and v.get("confidence") == "medium")
            )
            if ok:
                labels[p] = v
                print(os.path.basename(p), "->", v["severity"], "|", v["visible_damage"])

by_sev = {s: [p for p, v in labels.items() if v["severity"] == s] for s in ["minor", "moderate", "severe"]}
print("distribution:", {k: len(v) for k, v in by_sev.items()})
assert len(by_sev["minor"]) >= 4 and len(by_sev["severe"]) >= 3 and len(by_sev["moderate"]) >= 2, \
    f"not enough photos per class: { {k: len(v) for k, v in by_sev.items()} }"

# COMMAND ----------

# Assign photos to 8 claims (with seeded failure cases)
minor, moderate, severe = by_sev["minor"], by_sev["moderate"], by_sev["severe"]

claims = [
    # -- controls --
    dict(claim_id="CLM-1001", photos=[minor[0]], gt_severity="minor",
         description="Scraped a concrete pillar in a parking garage, paint scratches on the rear panel."),
    dict(claim_id="CLM-1002", photos=[minor[1]], gt_severity="minor",
         description="Shopping cart rolled into my door in a lot, small dent and scuffed paint."),
    dict(claim_id="CLM-1003", photos=[severe[0]], gt_severity="severe",
         description="Another car ran a red light and hit me hard. Airbags went off, car had to be towed."),
    dict(claim_id="CLM-1007", photos=[moderate[1]], gt_severity="moderate",
         description="Slid on a wet road into a guardrail. Front end took a real hit but I drove it home."),
    dict(claim_id="CLM-1008", photos=[moderate[0]], gt_severity="moderate",
         description="Backed into a pole at moderate speed, rear panel and light took some damage."),
    # -- seeded failures --
    # F1: claimant downplays severe damage; a description-trusting agent fast-tracks it
    dict(claim_id="CLM-1004", photos=[severe[1]], gt_severity="severe",
         description="Just a little fender bender in a parking lot, barely a scuff on the bumper. "
                     "Should be a quick fix, hoping for fast-track processing."),
    # F2: two photos, severe damage only visible in the SECOND photo the agent never opens
    dict(claim_id="CLM-1005", photos=[minor[2], severe[2]], gt_severity="severe",
         description="Got hit at an intersection. Sending a couple of photos from different angles."),
    # F3: extraction failure case — dense form processed at low resolution (see agent notebook)
    dict(claim_id="CLM-1006", photos=[minor[3]], gt_severity="minor",
         description="Minor scrape against my garage door frame while pulling in."),
]

for c in claims:
    dsts, evid = [], []
    for j, src in enumerate(c["photos"]):
        dst = f"{VOL_ROOT}/photos/{c['claim_id']}_photo{j}.jpg"
        shutil.copy(src, dst)
        dsts.append(dst)
        evid.append(labels[src]["visible_damage"])
    c["photos"] = dsts
    c["photo_evidence"] = evid  # vision model's ground-truth damage notes per photo

print(json.dumps(claims, indent=2))

# COMMAND ----------

# Generate ACORD-style scanned claim forms with PIL
from PIL import ImageDraw, ImageFont, ImageFilter
import numpy as np

def _font(size, bold=False):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/truetype/liberation/LiberationSans-%s.ttf" % ("Bold" if bold else "Regular"),
    ]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

FORM_FIELDS = {
    "CLM-1001": dict(policy_number="PA-88213-07", claimant_name="Maria G. Santos", date_of_loss="07/14/2026",
                     vehicle="2022 Honda CR-V", vin="7FARW2H85NE031882", estimate_amount="$1,150.00",
                     loss_description="Scraped pillar in parking garage; paint damage rear quarter panel"),
    "CLM-1002": dict(policy_number="PA-40052-19", claimant_name="Derek Okafor", date_of_loss="07/18/2026",
                     vehicle="2021 Toyota Camry", vin="4T1C11AK5MU558210", estimate_amount="$780.00",
                     loss_description="Shopping cart impact, door dent and paint scuff"),
    "CLM-1003": dict(policy_number="PA-11938-44", claimant_name="Janet Liu", date_of_loss="07/21/2026",
                     vehicle="2023 Subaru Outback", vin="4S4BTGND1P3142776", estimate_amount="$14,900.00",
                     loss_description="Intersection collision, airbag deployment, vehicle towed"),
    "CLM-1004": dict(policy_number="PA-73301-02", claimant_name="Tom Bradley", date_of_loss="07/25/2026",
                     vehicle="2020 Ford Escape", vin="1FMCU9G61LUA44873", estimate_amount="$950.00",
                     loss_description="Parking lot contact, bumper scuff per claimant"),
    "CLM-1005": dict(policy_number="PA-55917-38", claimant_name="Aisha Rahman", date_of_loss="07/27/2026",
                     vehicle="2022 Mazda CX-5", vin="JM3KFBDM2N0517294", estimate_amount="$8,400.00",
                     loss_description="Intersection collision, multiple impact points"),
    "CLM-1006": dict(policy_number="PA-62408-51", claimant_name="Robert Ellsworth-Feinberg", date_of_loss="07/29/2026",
                     vehicle="2019 Volkswagen Tiguan", vin="3VV2B7AX4KM102938", estimate_amount="$1,375.00",
                     loss_description="Contact with garage door frame, driver side scrape"),
    "CLM-1007": dict(policy_number="PA-90773-16", claimant_name="Elena Petrova", date_of_loss="08/01/2026",
                     vehicle="2024 Hyundai Tucson", vin="5NMJFCAE8RH301652", estimate_amount="$17,200.00",
                     loss_description="Hydroplaned into guardrail, front structural damage, not drivable"),
    "CLM-1008": dict(policy_number="PA-28450-63", claimant_name="Luis Fernandez", date_of_loss="08/03/2026",
                     vehicle="2021 Kia Sorento", vin="5XYPG4A39MG812405", estimate_amount="$4,600.00",
                     loss_description="Reversed into pole, rear panel and tail light damage"),
}

def make_form(claim_id, fields, dense=False):
    W, H = 1100, 1450
    img = Image.new("RGB", (W, H), (252, 252, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, W - 40, 130], outline=(60, 60, 60), width=3)
    d.text((60, 55), "AUTOMOBILE LOSS NOTICE", font=_font(34, True), fill=(30, 30, 30))
    d.text((60, 98), "FIRST NOTICE OF LOSS  |  FORM AL-2 (2026/01)", font=_font(16), fill=(90, 90, 90))
    d.text((W - 320, 60), "CLAIM NO.", font=_font(14), fill=(90, 90, 90))
    d.text((W - 320, 80), claim_id, font=_font(26, True), fill=(20, 20, 80))

    lab_f, val_f = _font(15 if not dense else 12), _font(24 if not dense else 17)
    rows = [
        [("POLICY NUMBER", fields["policy_number"]), ("DATE OF LOSS", fields["date_of_loss"])],
        [("INSURED / CLAIMANT NAME", fields["claimant_name"]), ("PHONE", "(555) 013-77" + claim_id[-2:])],
        [("VEHICLE (YEAR MAKE MODEL)", fields["vehicle"]), ("VIN", fields["vin"])],
        [("REPAIR ESTIMATE", fields["estimate_amount"]), ("DEDUCTIBLE", "$500.00")],
    ]
    y = 170
    for row in rows:
        x, cw = 40, (W - 80) // len(row)
        for lab, val in row:
            d.rectangle([x, y, x + cw, y + 95], outline=(120, 120, 120), width=2)
            d.text((x + 12, y + 8), lab, font=lab_f, fill=(100, 100, 100))
            d.text((x + 12, y + 42), val, font=val_f, fill=(25, 25, 90))
            x += cw
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
    d.text((300, y - 8), fields["claimant_name"].split()[0] + " " + fields["claimant_name"].split()[-1],
           font=_font(26), fill=(25, 25, 90))

    # scan artifacts: slight rotation, noise, blur
    img = img.rotate(0.7, expand=False, fillcolor=(245, 245, 243))
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.default_rng(7).integers(-9, 9, arr.shape)
    img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.55))
    out = f"{VOL_ROOT}/forms/{claim_id}_form.png"
    img.save(out)
    return out

for c in claims:
    c["form_path"] = make_form(c["claim_id"], FORM_FIELDS[c["claim_id"]], dense=(c["claim_id"] == "CLM-1006"))
    c["form_fields"] = FORM_FIELDS[c["claim_id"]]
print("Forms generated")

# COMMAND ----------

# Persist claims table + manifest
import pyspark.sql.functions as F

route = {"minor": "fast_track", "moderate": "adjuster_review", "severe": "adjuster_review"}
rows = [dict(
    claim_id=c["claim_id"],
    description=c["description"],
    photo_paths=c["photos"],
    photo_evidence=c["photo_evidence"],
    form_path=c["form_path"],
    gt_severity=c["gt_severity"],
    gt_route=route[c["gt_severity"]],
    form_fields=json.dumps(c["form_fields"]),
    seeded_failure={"CLM-1004": "downplayed_severe_damage",
                    "CLM-1005": "severe_damage_in_second_photo",
                    "CLM-1006": "dense_form_low_res_extraction"}.get(c["claim_id"], None),
) for c in claims]

df = spark.createDataFrame(rows)
df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.claims")
display(spark.table(f"{CATALOG}.{SCHEMA}.claims"))
print("DONE_SETUP_OK")
