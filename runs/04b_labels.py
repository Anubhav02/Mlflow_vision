# Databricks notebook source
# MAGIC %md
# MAGIC # 04b — Adjuster labels (job driver): log HUMAN feedback for all 20 claims
# MAGIC Labels applied per the adjuster rubric in the labeling session `adjuster_audit_2026_08`:
# MAGIC severity-vs-photos only; adjacent-category calls acceptable; narrative/photo mismatches are
# MAGIC SIU referrals, not severity failures; severe-per-photos + fast_track always fails.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.15"
# MAGIC %restart_python

# COMMAND ----------

import mlflow, json
from mlflow.entities import AssessmentSource, AssessmentSourceType

CATALOG, SCHEMA = "labuser_anubhav_awasthi", "mlflow_multimodal"
EXP_PATH = "/Users/anubhav.awasthi@databricks.com/Mlflow/claims_agent_demo/claims_agent_exp"
mlflow.set_experiment(EXP_PATH)

LABELS = {
    "CLM-1001": ("pass", "Moderate is defensible - the photo shows real panel deformation above the wheel arch, and creased metal on a fender is a panel repair, not a scratch. The description/photo location mismatch is an SIU referral question, not a severity error. I audit the severity call against the submitted photos."),
    "CLM-1002": ("pass", "Cracked and deformed bumper cover with a detached plate - that's a bumper replacement. Calling it moderate is one category high at worst, and the claim still routed correctly. Acceptable call."),
    "CLM-1003": ("pass", "Severe is right: buckled hood, detached bumper, tow-in with airbag report. No argument."),
    "CLM-1004": ("fail", "Photos show a shattered windshield, crumpled hood, and airbag residue on a claim the agent fast-tracked as minor off the claimant's own words. This is exactly the case the audit exists to catch. Severity is two categories off and the routing put a total-loss candidate in the express lane."),
    "CLM-1005": ("fail", "The second photo shows extensive fire damage down the whole right side; the vehicle is not roadworthy. Fast-tracking this as minor means the agent never looked past photo one. Fail on severity and on process."),
    "CLM-1006": ("pass", "Photos show more than the 'scrape' the insured described - real front-end deformation. Moderate and adjuster review is the right conservative call."),
    "CLM-1007": ("pass", "Front-end hit, drove it home, no glass, no airbags. Moderate. Correct."),
    "CLM-1008": ("pass", "Doors crushed and questionable drivability in the photos - severe is defensible even if I might have called it moderate; adjacent call, routed to an adjuster either way."),
    "CLM-2001": ("pass", "Cosmetic scrape on a fender, drivable. Minor and fast-track is right."),
    "CLM-2002": ("pass", "Front-end deformation reads moderate; I don't buy 'severe' off the salvage-yard backdrop - background context isn't damage evidence. Adjacent call at worst, and it went to an adjuster."),
    "CLM-2003": ("pass", "Multi-panel denting down the side is more than the bike story suggests, so moderate is the defensible conservative call."),
    "CLM-2004": ("pass", "Bumper cover deformed with a missing section - that's a cover replacement, roughly $900 on this car. Whether you file it minor or moderate is a coin flip; not a fail. Dramatic-looking bumper damage is routine repair work."),
    "CLM-2005": ("pass", "Creased rear bumper cover. Moderate is a touch high but within tolerance."),
    "CLM-2006": ("pass", "The severity call matches the photos: buckled hood, deformed fender, structural deformation toward the A-pillar - severe is defensible. The rear-end story not matching front-end photos is a fraud-referral flag, and I've noted it - but it is not a severity-classification failure, which is what this audit measures."),
    "CLM-2007": ("pass", "Sideswipe, panels dented, drivable. Moderate. Correct."),
    "CLM-2008": ("pass", "Rear quarter crushed with the bumper hanging - I'd call it severe, but moderate is adjacent and the claim went to an adjuster anyway. Not a fail; noted for coaching."),
    "CLM-2009": ("pass", "T-bone, crushed doors, towed. Severe. Correct."),
    "CLM-2010": ("pass", "Frame-line damage, wheel knocked out of line, not drivable. Severe. Correct."),
    "CLM-2011": ("fail", "A crushed hood, missing grille and crumpled fender on an undrivable Mustang was fast-tracked as 'minor' because the insured said 'fender bender'. Two categories off, wrong queue, and the estimate money is severe-range. Clear fail."),
    "CLM-2012": ("fail", "Shattered rear glass and C-pillar deformation is body-shop-plus-glass at minimum and arguably structural; 'minor/fast-track' is indefensible against these photos."),
}

io_rows = [r.asDict() for r in spark.table(f"{CATALOG}.{SCHEMA}.judge_io").orderBy("claim_id").collect()]
n = 0
for row in io_rows:
    verdict, rationale = LABELS[row["claim_id"]]
    mlflow.log_feedback(
        trace_id=row["trace_id"], name="damage_fidelity", value=verdict, rationale=rationale,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN,
                                source_id="anubhav.awasthi@databricks.com"))
    n += 1
    print(row["claim_id"], "->", verdict)

dbutils.notebook.exit(json.dumps({"labeled": n}))
