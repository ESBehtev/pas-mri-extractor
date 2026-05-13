"""
Pydantic-схемы результата.

Нужны для валидации JSON от LLM и единообразной структуры результата.
"""

from typing import Literal

from pydantic import BaseModel


InvasionType = Literal["none", "accreta", "increta", "percreta"]
InvasionConfidence = Literal["absent", "possible", "probable", "definite", "unclear"]
FeatureStatus = Literal["absent", "possible", "probable", "present"]
RiskGroup = Literal["low", "moderate", "high"]


class MRIFeatures(BaseModel):
    invasion_type: InvasionType = "none"
    invasion_confidence: InvasionConfidence = "absent"

    bladder_involvement: FeatureStatus = "absent"
    parametrium_involvement: FeatureStatus = "absent"
    posterior_wall_involvement: FeatureStatus = "absent"

    placenta_previa: FeatureStatus = "absent"
    anterior_placenta: FeatureStatus = "absent"

    retroplacental_vessels: FeatureStatus = "absent"
    lacunae: FeatureStatus = "absent"
    uterine_wall_thinning: FeatureStatus = "absent"
    uterine_hernia_or_bulging: FeatureStatus = "absent"

    preoperative_bleeding: FeatureStatus = "absent"

    previous_cs_count: int | None = None
    gestational_week: int | None = None

    short_explanation: str = ""


class MRIExtractionResult(BaseModel):
    features: MRIFeatures
    clinical_summary: str = ""
    clinical_rationale: str = ""


class ScoreResult(BaseModel):
    clinical_score: int
    risk_group: RiskGroup
    red_flag: int
    score_reasons: str


class PredictedRisks(BaseModel):
    massive_blood_loss_over_1500_ml_percent: int
    estimated_blood_loss_ml_range: str
    vascular_intervention_percent: int
    bladder_involvement_percent: int
    risk_summary_text: str


class Recommendation(BaseModel):
    readiness_level: str
    readiness_text: str


class FullMRIResult(MRIExtractionResult):
    score: ScoreResult | None = None
    predicted_risks: PredictedRisks | None = None
    recommendation: Recommendation | None = None
    computed_rationale: str | None = None