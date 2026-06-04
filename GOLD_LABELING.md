# GOLD_LABELING.md

## Purpose

This document defines the formal protocol for retrospective PAS gold labeling in
this project. Gold labels are used for scientific evaluation of the extractor.
They are not clinical conclusions and are not intended for patient care.

Gold labeling must be performed as retrospective clinical adjudication, not as
MRI extraction. MRI is an important diagnostic input, but it is not the highest
source of truth when operative or discharge records provide more reliable
information.

Before creating or changing any `gold_*` labels, the annotating agent must read
this file and follow it. If this file conflicts with older repository
instructions, this file has priority for gold labeling.

## Source Priority

Use source fields in this order:

1. `Ход Вмешательства`
2. `ДиагнозыВыпЭпикриза`
3. `ПоказанияКОперации`
4. `КровопотеряОперация` / `КровопотеряРоды`
5. `МРТ_Описание` / `МРТ_Заключение`

Use MRI fields as diagnostic evidence, especially for imaging signs. However,
when operation notes or discharge diagnoses clearly contradict MRI suspicion,
prefer operation/discharge evidence for final retrospective gold labels.

## Label Categories

Gold labels are split into two groups.

Retrospective outcome/anatomy labels:

- `gold_invasion_type`
- `gold_invasion_confidence`
- `gold_bladder_involvement`
- `gold_parametrium_involvement`
- `gold_posterior_wall_involvement`
- `gold_vascular_intervention`
- `gold_blood_loss_ml`
- `gold_massive_blood_loss`
- `gold_blood_loss_class`
- `gold_readiness_level`
- `gold_risk_group`

These must be adjudicated primarily from:

1. `Ход Вмешательства`
2. `ДиагнозыВыпЭпикриза`
3. `ПоказанияКОперации`
4. factual outcomes

MRI feature labels:

- `gold_placenta_previa`
- `gold_anterior_placenta`
- `gold_retroplacental_vessels`
- `gold_lacunae`
- `gold_uterine_wall_thinning`
- `gold_uterine_hernia_or_bulging`
- `gold_preoperative_bleeding`
- `gold_highest_suspected_extent`
- `gold_percreta_suspicion`
- `gold_bladder_serosa_suspicion`

These may be adjudicated from MRI fields.

Missing information must remain empty.

Do not convert missing information into absent.

## Allowed Values

PAS and feature labels:

- `gold_invasion_type`: `none` / `accreta` / `increta` / `percreta`
- `gold_invasion_confidence`: `absent` / `possible` / `probable` / `definite` / `unclear`
- `gold_bladder_involvement`: `absent` / `possible` / `probable` / `present`
- `gold_parametrium_involvement`: `absent` / `possible` / `probable` / `present`
- `gold_posterior_wall_involvement`: `absent` / `possible` / `probable` / `present`
- `gold_placenta_previa`: `absent` / `possible` / `probable` / `present`
- `gold_anterior_placenta`: `absent` / `possible` / `probable` / `present`
- `gold_retroplacental_vessels`: `absent` / `possible` / `probable` / `present`
- `gold_lacunae`: `absent` / `possible` / `probable` / `present`
- `gold_uterine_wall_thinning`: `absent` / `possible` / `probable` / `present`
- `gold_uterine_hernia_or_bulging`: `absent` / `possible` / `probable` / `present`
- `gold_preoperative_bleeding`: `absent` / `possible` / `probable` / `present`
- `gold_highest_suspected_extent`: `none` / `accreta` / `increta` / `percreta`
- `gold_percreta_suspicion`: `absent` / `possible` / `probable` / `present`
- `gold_bladder_serosa_suspicion`: `absent` / `possible` / `probable` / `present`
- `gold_vascular_intervention`: `absent` / `possible` / `probable` / `present`

Outcome and adjudication labels:

- `gold_blood_loss_ml`: integer or empty
- `gold_massive_blood_loss`: `true` / `false` / empty
- `gold_blood_loss_class`: `0-500` / `500-1000` / `1000-1500` / `1500-2000` / `2000+`
- `gold_readiness_level`: `1` / `2` / `3` / `4`
- `gold_risk_group`: `low` / `medium` / `high` / `critical`
- `gold_confidence`: `high` / `medium` / `low`
- `gold_rationale`: short rationale with source fields and source phrases

Leave a gold field empty when there is insufficient information. Do not set
`absent` only because a finding is not mentioned.

## PAS Type Rules

Set `gold_invasion_type = percreta` when there is clear evidence of at least one
of the following:

- invasion beyond the uterus;
- serosal involvement with extrauterine extension;
- PAS-related bladder involvement;
- PAS-related parametrium involvement;
- bladder resection due to placental invasion;
- cystotomy or bladder suturing in the setting of placental invasion;
- `placenta percreta` in operative diagnosis or discharge summary.

Set `gold_invasion_type = increta` when there is evidence of deep myometrial
invasion without convincing extrauterine spread:

- deep myometrial invasion;
- `placenta increta`;
- myometrium is severely thinned or not visualized in areas;
- disrupted placenta/myometrium interface;
- no convincing extension beyond the uterus.

Set `gold_invasion_type = accreta` when there is evidence of superficial PAS or
non-specific PAS:

- `placenta accreta`;
- dense placental adherence;
- superficial invasion;
- general "врастание плаценты" without specified depth.

Set `gold_invasion_type = none` only when PAS/invasion is explicitly absent or
the clinical record reliably shows no PAS.

If multiple PAS types are present, use the most severe confirmed type:

```text
percreta > increta > accreta > none
```

Do not upgrade `gold_invasion_type` to `percreta` solely because MRI suggests:

- cannot exclude percreta;
- possible percreta;
- suspicion of percreta.

In such cases:

- keep confirmed PAS type in `gold_invasion_type`;
- set `gold_highest_suspected_extent = percreta`;
- set `gold_percreta_suspicion = possible`.

Percreta requires operative or discharge confirmation, or clear evidence of
extrauterine invasion.

Store only the PAS type in `gold_invasion_type`. Do not put uncertainty,
phrases, or free text in that field. Put certainty in
`gold_invasion_confidence`.

## Bladder Rules

Set `gold_bladder_involvement = present` only with explicit evidence that
bladder involvement is related to PAS or placental invasion:

- direct bladder invasion by placenta;
- placental ingrowth into bladder;
- bladder resection because of placental invasion;
- bladder suturing because of placental invasion;
- cystotomy because of placental invasion;
- bladder injury explicitly caused by invasive placenta;
- placenta percreta with bladder involvement.

Do NOT automatically treat the following as `present`:

- cystotomy;
- bladder suturing;
- bladder injury.

First determine whether they were caused by PAS.

If the relation to PAS is unclear:

- `probable` if strongly suggestive;
- `possible` if uncertain.

Set `probable` when:

- bladder is clearly involved or adherent;
- placental process affects bladder;
- invasion is strongly suspected but not directly confirmed.

Set `possible` when:

- bladder involvement cannot be excluded;
- vesicouterine space poorly differentiated;
- serosal involvement uncertain;
- bladder deformation without confirmed invasion.

Set `absent` only when bladder involvement is explicitly denied.

Do not set `present` only by phrases like vesicouterine space not
differentiated, bladder wall deformation, bladder adjacency, or cannot exclude
bladder involvement.

## Other Feature Rules

For other imaging/clinical features, use the same certainty scale:

- `present`: clear positive statement;
- `probable`: high-probability statement such as "corresponds to" or "most likely";
- `possible`: uncertainty such as "cannot exclude", "possible", "suspicion";
- `absent`: explicit negative statement;
- empty: insufficient information.

Do not infer absence from silence. Keep uncertain statements separate from
definite findings.

## Suspicion Fields

`gold_highest_suspected_extent` captures the worst suspected PAS extent when
there is uncertainty about more severe disease. It must not be less severe than
`gold_invasion_type`.

Use `gold_percreta_suspicion` and `gold_bladder_serosa_suspicion` for explicit
or implied uncertain percreta/bladder/serosal concern. Use `present` when the
same involvement is confirmed, and `possible`/`probable` when it is suspected.

## Outcome Rules

Blood loss:

- Use `КровопотеряОперация`.
- If `КровопотеряОперация` is empty, use `КровопотеряРоды`.
- If both are empty or not interpretable, leave blood-loss fields empty.

Blood-loss classes:

```text
0-500: <=500 ml
500-1000: >500 and <=1000 ml
1000-1500: >1000 and <=1500 ml
1500-2000: >1500 and <=2000 ml
2000+: >2000 ml
```

Set `gold_massive_blood_loss = true` when blood loss is `>1500 ml`.
Set it to `false` when blood loss is `<=1500 ml`.

Readiness:

```text
1: blood loss <=1000 ml, no bladder present, no vascular intervention present
2: blood loss >1000 and <=1500 ml, no bladder present, no vascular intervention present
3: blood loss >1500 and <=3000 ml, or bladder present, or vascular intervention present
4: blood loss >3000 ml, or extremely severe maternal outcome
```

Risk group:

```text
1 -> low
2 -> medium
3 -> high
4 -> critical
```

Do not use neonatal resuscitation alone as an extremely severe maternal outcome.

## Vascular Intervention Rules

Set `gold_vascular_intervention = present` when an actual vascular intervention
was performed, for example:

- uterine artery embolization;
- internal iliac artery ligation;
- endovascular balloon occlusion of major pelvic/aortic vessels;
- catheterization/embolization of uterine arteries.

Set `possible` or `probable` only when planned/likely intervention is documented
but performance is unclear. Set `absent` when the note explicitly says the team
declined, avoided, or did not perform the intervention.

Do not count intrauterine balloon tamponade as vascular intervention.

## Gold Confidence

Use:

- `high`: operative/discharge source directly supports the main label and
  outcome fields are internally consistent;
- `medium`: source evidence is present but indirect, incomplete, or relies mainly
  on MRI;
- `low`: sparse, conflicting, or weak evidence.

## Rationale

`gold_rationale` must be short and traceable. Include source field names and
short source phrases. Prefer phrasing like:

```text
Ход Вмешательства: "..."; ДиагнозыВыпЭпикриза: "..."; КровопотеряОперация: 1800
```

Do not include model predictions, scoring explanations, or invented findings in
`gold_rationale`.

## Mandatory Self-Review

Before saving each case:

- if `gold_bladder_involvement = present`, `gold_rationale` must contain
  explicit PAS-related bladder invasion, PAS-related resection, PAS-related
  suturing, PAS-related cystotomy, or PAS-related bladder injury;
- if bladder injury/cystotomy/suturing is present but PAS relation is unclear,
  do not use `present`;
- if `gold_bladder_involvement = present`, `gold_readiness_level` cannot be `1`
  or `2`;
- if `gold_invasion_type = percreta`, `gold_rationale` must contain extrauterine
  extension, serosal invasion, PAS-related bladder involvement, PAS-related
  parametrium involvement, PAS-related bladder surgery, or `placenta percreta`;
- MRI suspicion alone cannot create `gold_invasion_type = percreta`;
- if `gold_blood_loss_ml > 1500`, then `gold_massive_blood_loss` must be `true`;
- if `gold_blood_loss_ml <= 1500`, then `gold_massive_blood_loss` must be
  `false`;
- `gold_blood_loss_class` must match `gold_blood_loss_ml`;
- `gold_highest_suspected_extent` must not be less severe than
  `gold_invasion_type`;
- `gold_risk_group` must match `gold_readiness_level`.

If any contradiction exists, correct the case before writing.

## Iterative Annotation Protocol

Gold labeling must be performed one case at a time.

For every case:

1. Read source fields.
2. Determine operative truth.
3. Create draft gold labels.
4. Run self-review.
5. Correct contradictions.
6. Save labels.
7. Continue to next case.

Do not batch-label cases without self-review.

Every saved case must satisfy all Mandatory Self-Review rules.
