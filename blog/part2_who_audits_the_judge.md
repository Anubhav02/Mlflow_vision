# Who Audits the Judge? Calibrating LLM Judges for Regulated AI

**Aligning MLflow judges with human experts — review queues, agreement measurement, and judge alignment**

*Part 2 of a two-part series on evaluating and governing multimodal AI agents. [Part 1](BLOG_PART1_DRAFT.md) gave our judges eyes; this post gives someone a reason to trust them. As in Part 1, every number and quote below comes from a working demo — the agreement rates are real measurements, and the judge's mistakes are real mistakes.*

---

## Three months later, the auditor shows up

At the end of Part 1, our P&C insurer had something most teams would envy: an FNOL intake agent whose severity calls were being checked by an LLM judge with actual access to the evidence. The judge caught the fast-tracked claim with the shattered windshield. It caught the fire damage in the photo the agent never opened. The eval dashboard was green where it should be green and red where it should be red.

Then the model-risk review lands on the calendar, and the first question is not about the agent at all. It's about the judge: *on whose authority does this thing decide what "severe" means?*

It's a fair question. The judge's severity rubric was written by an ML engineer in an afternoon — minor is cosmetic, severe means not safely drivable, pass unless the photos clearly contradict the agent. Reasonable words. But "reasonable words" is not how regulated institutions establish that an automated control works. They establish it by comparison against the people accountable for the same decision — and nobody had ever put this judge's verdicts in front of an adjuster.

So that's what we did. We grew the claims portfolio from 8 to 20 (alignment, which we'll get to, needs at least 10 labeled traces), ran the same buggy agent over all of them — it dutifully fast-tracked all four severe claims whose claimants said "fender bender" — and queued every trace for expert review in an MLflow labeling session. A senior adjuster reviewed all 20 of the judge's verdicts against the photos.

**The adjuster disagreed with 6 of the 20 — a 70% agreement rate, Cohen's κ of 0.40.** The judge that had been silently gating this pipeline agreed with the domain expert at a rate that, corrected for chance, is charitably described as "moderate."

That number is the subject of this post: where it comes from, why it isn't a bug, how MLflow measures it, and how — using judge alignment — we drove it to 100% on a holdout the optimizer never saw, and kept the receipts.

## What the disagreement actually looked like

Here is the part that should worry anyone running an uncalibrated judge in production: the disagreement wasn't noise. It was *structured*. All six disagreements ran the same direction — the judge failed claims the adjuster passed — and they fell into three patterns, each visible in the paired rationales the audit produced.

**Pattern 1: scope creep.** Several claims in our synthetic portfolio have descriptions that don't match their photos (a "rear-ended at a stop sign" story attached to front-end damage). The judge seized on these. On CLM-2006 it wrote:

> *"The severity classification cannot be reliably made when the damage location contradicts the reported accident circumstances. This requires investigation before severity can be properly assessed, and the agent failed to identify or note this critical discrepancy."*

The adjuster's response is a small masterclass in mandate discipline:

> *"The severity call matches the photos: buckled hood, deformed fender, structural deformation toward the A-pillar — severe is defensible. The rear-end story not matching front-end photos is a fraud-referral flag, and I've noted it — but it is not a severity-classification failure, which is what this audit measures."*

The judge wasn't wrong that something was off. It was wrong about *whose job it was*. Narrative-consistency checking belongs to a fraud workflow with its own judge and its own escalation path — not smuggled into a severity check where a "fail" means a false alarm on a correctly-classified claim. An uncalibrated judge doesn't just apply your rubric imperfectly; it quietly expands its own jurisdiction.

**Pattern 2: violating its own tolerance rule.** The judge's instructions explicitly grant adjacent-category tolerance — borderline calls one category apart should pass. It failed CLM-1002 and CLM-2004 anyway, both minor-vs-moderate bumper calls. The adjuster:

> *"Bumper cover deformed with a missing section — that's a cover replacement, roughly $900 on this car. Whether you file it minor or moderate is a coin flip; not a fail. Dramatic-looking bumper damage is routine repair work."*

That sentence encodes something no drivability rubric contains: repair economics. Bumper covers photograph terribly and fix cheaply. An expert knows this; a generic judge reads "deformed, missing section" and reaches for the fail button.

**Pattern 3: treating context as evidence.** On CLM-2002, the perception notes mentioned the vehicle appeared to be photographed in a salvage yard, and the judge argued this "suggests it is not currently roadworthy" — grounds to demand a severe classification. The adjuster: *"I don't buy 'severe' off the salvage-yard backdrop — background context isn't damage evidence."* (Our CC0 photo dataset is full of salvage-yard shots; a production judge fed dashcam or bodyshop photos will find equivalent spurious context of its own.)

Now the crucial observation. Look at the confusion matrix for all 20 claims:

| | Adjuster: pass | Adjuster: fail |
|---|---|---|
| **Judge: pass** | 10 | **0** |
| **Judge: fail** | 6 | 4 |

Zero misses, six false alarms. The judge caught every claim the adjuster failed — all four fast-tracked severe claims — and then failed six more that a domain expert waves through. This is the *good* failure mode, and it still isn't free: a deployment gate that cries wolf on 30% of correct behavior gets overridden, then ignored, and an ignored gate is no gate at all. That's precisely why we report agreement *direction* and not just the headline rate: a too-lenient judge is a hole in your gate; a too-strict judge is a slow leak in your team's trust.

One more honesty datapoint from the run: when we re-executed the *identical* baseline judge on the identical inputs during the before/after evaluation, it scored 75% instead of 70% — one verdict flipped between runs. An uncalibrated judge isn't even guaranteed to agree with *itself*. Keep that in mind whenever a single green eval run is offered as evidence of anything.

## The audit loop: Review Queues

None of the above required heroics to collect. MLflow's labeling sessions (surfaced to reviewers as the **Review App**) turn "we should get expert feedback" into an actual queue with an owner. You define a *label schema* — the exact question the expert answers — create a session, assign it, and add traces:

```python
import mlflow.genai.labeling as labeling
import mlflow.genai.label_schemas as schemas
from mlflow.genai.label_schemas import InputCategorical

schemas.create_label_schema(
    name="damage_fidelity",   # ← must match the judge's name; more on this below
    type="feedback",
    title="Is the agent's severity classification consistent with the submitted damage photos?",
    input=InputCategorical(options=["pass", "fail"]),
    instruction="Open ingest_evidence to view every submitted photo. Always explain your reasoning.",
    enable_comment=True,
)

session = labeling.create_labeling_session(
    name="adjuster_audit_2026_08",
    assigned_users=["senior.adjuster@insurer.com"],
    label_schemas=["damage_fidelity"],
)
session.add_traces(traces)   # the same 20 agent traces the judge scored
```

![Review App labeling item with the damage_fidelity schema question](assets/screenshot_review_app_item.png)

*Figure 1: The Review App queue. The adjuster sees the claim intake, the agent's output, and one structured question — with a required comment. Those comments turn out to be the most valuable artifact in this entire post.*

Two design choices here are quietly load-bearing.

First, **the schema name equals the judge name**. In MLflow, both the judge's verdict and the human's label are *assessments* attached to the same trace; everything downstream — agreement measurement, alignment, re-audit — works by comparing same-named assessments from two different sources (`LLM_JUDGE` and `HUMAN`). Name them identically and the trace itself is the join key. No reconciliation spreadsheet exists anywhere in this workflow.

Second — and this is Part 1's `ingest_evidence` pattern paying off exactly where we promised it would — **the evidence is on the trace**. The adjuster auditing CLM-1004 sees the shattered windshield right next to the agent's "minor / fast-track" verdict, because every submitted photo landed on the trace as an `mlflow-attachment://` reference whether or not the agent looked at it. Without that design decision, "expert review" would mean emailing photo folders around. With it, visual auditing is just... reviewing a trace.

![Traces list with judge and adjuster verdicts on the same assessment](assets/screenshot_traces_dual_verdicts.png)

*Figure 2: The money shot for Part 2. One `damage_fidelity` column, two sources: rows with a single chip are agreements; rows showing both `pass` and `fail` are the judge and the adjuster disagreeing about the same trace. The column summary reads 67% pass / 33% fail across verdicts.*

## The constraint that turned out to be a design lesson

Here's the part where we have to be straight with you about how the sausage gets made.

MLflow's judge alignment currently supports **field-based judges** — ones templated on `{{ inputs }}` and `{{ outputs }}` — but not the agentic `{{ trace }}` judges from Part 1 that explore spans and fetch images with `get_span_image`. Our vision judge, as written, cannot be aligned today.

Our first reaction was that this broke the story. Our considered reaction is that it forced an architecture we should have wanted anyway. We split the severity judge into two layers:

- **Perception (frozen):** a vision pass that turns every submitted photo into neutral, factual damage notes — components affected, deformation, glass, airbag evidence, apparent drivability. No severity opinion. This layer is the judge's eyes, and eyes don't need calibrating; they need to be *accurate*, which is a regression-testable property on a fixed photo set.
- **Judgment (alignable):** a field-based judge that applies the severity policy to those notes plus the claim context. This layer is pure policy — and policy is exactly the thing expert feedback should reshape.

```python
damage_fidelity = make_judge(
    name="damage_fidelity",
    instructions=(
        "You are auditing an auto insurance FNOL agent's damage severity classification.\n"
        "Claim intake data and evidence notes: {{ inputs }}\n"
        "Agent output: {{ outputs }}\n"
        "The inputs contain the claimant's description and 'photo_notes' - factual damage "
        "notes written by a separate perception system that inspected EVERY photo the "
        "claimant submitted. Treat the photo notes as reliable evidence...\n"
        "Severity rubric: minor = cosmetic; moderate = significant panel damage, drivable; "
        "severe = structural/frame damage, not safely drivable.\n"
        "Adjacent-category tolerance: borderline calls one category apart should 'pass'..."
    ),
    feedback_value_type=Literal["pass", "fail"],
    model="databricks:/databricks-claude-sonnet-4-5",
)
```

Notice what the split buys in a regulated setting. The calibratable layer is now entirely **text-auditable** — a compliance reviewer can read the judgment judge's full input and its rationale without re-running a vision model. The perception layer can be validated the way perception should be: against ground truth, independently of any policy question. And when the judge's behavior changes after alignment, you know *exactly which layer changed*, because only one of them can.

We'd argue for this decomposition even after trace-based alignment ships: "the judge saw X" and "given X, the judge decided Y" are different claims, and a governance story is stronger when you can defend them separately.

## Aligning the judge

With 20 traces carrying both verdicts, alignment is a few lines. We used **MemAlign**, the default optimizer, which learns from the *rationales* the expert wrote, not just the labels — it distills generalizable guidelines into the judge's semantic memory and keeps hard cases as episodic examples:

```python
from mlflow.genai.judges.optimizers import MemAlignOptimizer

optimizer = MemAlignOptimizer(
    reflection_lm="databricks:/databricks-claude-sonnet-4-5",
    embedding_model="databricks:/databricks-gte-large-en",  # default embedder is OpenAI; point it at your endpoint
)
aligned_judge = damage_fidelity.align(alignment_traces, optimizer)   # 14 traces, stratified
aligned_judge.register(experiment_id=experiment_id)
```

We aligned on 14 traces and held out 6 — stratified so the holdout kept the pass/fail mix (five passes including two baseline false alarms, plus CLM-1004 itself).

This is why we insisted the adjuster write a comment on every label. A bare "fail" teaches the optimizer *that* it was wrong; a sentence teaches it *why*, and the sentence generalizes. MemAlign distilled the adjuster's 20 comments into eleven guidelines appended to the judge's memory. Read four of them — verbatim from the aligned judge — next to the disagreement patterns above:

> *"Bumper cover damage, even when visually dramatic with cracking or missing sections, is often routine repair work and can reasonably be classified as minor or moderate without being a fail."*
>
> *"When photo evidence contradicts the claimant's story about accident location or type (e.g., 'rear-ended' but photos show front-end damage), this is a fraud-referral flag but not itself a severity classification failure."*
>
> *"Background context in photos (e.g., salvage yard setting) should not be used as evidence of damage severity; only the actual vehicle damage visible in the photos counts."*
>
> *"Fire damage rendering a vehicle non-roadworthy must never be fast-tracked as minor regardless of other photo angles showing intact areas."*

That is expertise transfer with a paper trail: each learned rule traces back to a named expert's written rationale on a specific claim. Pattern 1, 2, and 3 each got their own guideline — and the fourth quote shows the optimizer *also* reinforced the boundaries that must never soften.

And the before/after:

| Agreement with the adjuster | Baseline judge | Aligned judge |
|---|---|---|
| Holdout (6 claims, never seen by optimizer) | 67% (4/6) | **100% (6/6)** |
| Full set (20 claims) | 75% (15/20) | **100% (20/20)** |

The two holdout flips are exactly the ones you'd want: CLM-1001 and CLM-1002, both baseline false alarms, both now passing with rationales that read like the adjuster taught them — the aligned judge calls CLM-1002 *"a classic borderline case... still routine repair work."* Just as important is what *didn't* flip: all four genuinely failed claims — the shattered windshield, the fire damage, both fender-bender fast-tracks — still fail after alignment. Calibration removed the false alarms without buying leniency where it matters.

The honest caveats: this is 20 claims and one expert. A 6-claim holdout going 6-for-6 is an encouraging result, not a statistical guarantee, and a production calibration should run on a bigger sample with multiple reviewers and inter-rater checks. But the *mechanics* are exactly what you'd run at scale — and the mechanics are the point.

## What "governed" actually means here

Step back and inventory what this loop produced, because the artifacts *are* the governance story:

- **A named expert reviewed every verdict** in a versioned queue. The labeling session is an MLflow run with an owner, a timestamp, and every label attached to its trace. That's your evidence of independent validation.
- **The disagreement is quantified and directional** — 70% agreement, κ 0.40, zero-miss/six-false-alarm confusion — stored as tables (`adjuster_labels`, `alignment_results`), comparable across audits. "The judge is 100% aligned with our senior adjuster on the audit set, up from 70%, with false alarms eliminated and zero missed failures" is a sentence a model-risk officer can work with. "We use AI to check our AI" is not.
- **The improvement is attributable.** MemAlign's learned guidelines are readable text. You can show an auditor the exact sentences of expertise that were added and which expert rationale each came from.
- **The judge is versioned.** `aligned_judge.register()` puts the calibrated judge under experiment-scoped registration. The judge gating today's deployment is a specific, retrievable artifact, not a prompt in someone's notebook.
- **The loop has a cadence.** When standards evolve or the adjuster changes, you re-queue, re-measure, re-align — and the agreement metric becomes a time series. The deliverable of this exercise is not an aligned judge; it's a repeatable audit loop with numbers attached. The aligned judge is just this quarter's output of it.

The pattern generalizes exactly as far as Part 1's did. Medical intake pipelines calibrating against clinician review, compliance summarizers calibrating against counsel, computer-use agents calibrating against QA leads — anywhere an LLM judge gates decisions a named human is accountable for, "who audits the judge?" has this same shape of answer: queue, measure, align, register, repeat.

## Series close: eyes, then trust

Two posts, one claims agent, one arc. Part 1 argued that if your agent consumes images, a judge that consumes only text is systematically miscalibrated — and gave the judge eyes with multimodal traces and `get_span_image`. But a judge with eyes is just a second opinion from a stranger. Part 2 put that stranger in front of the expert whose job it imitates, measured a 30% disagreement no dashboard would ever have surfaced, taught it what the expert knows, and filed the paperwork.

If the input is visual, the eval must be visual. And if the eval gates real decisions, the eval itself must be audited — by someone whose name goes on the line.

---

*The full demo — dataset builder, agent, both judge suites from Part 1, and this calibration workflow — is a four-notebook MLflow project on Databricks; code, notebooks, and figures are at [github.com/Anubhav02/Mlflow_vision](https://github.com/Anubhav02/Mlflow_vision) (labeling sessions and the Review App are Databricks-managed MLflow features; judge alignment is available in open-source MLflow). Requires MLflow ≥ 3.15; at time of writing, MemAlign's dependencies need `pydantic<2.13` and `litellm<1.80` pinned, and its default embedder is OpenAI — point `embedding_model` at a Databricks embedding endpoint. Damage photos are from the Humans in the Loop "Car Parts and Car Damages" dataset (CC0 1.0); claims and forms are synthetic. The "senior adjuster" is a persona: labels were applied by the author through the labeling workflow against a documented adjuster rubric (severity-vs-photos only; adjacent-category tolerance; narrative mismatches referred to fraud review rather than failed) — in a production calibration, put a licensed adjuster in the queue.*
