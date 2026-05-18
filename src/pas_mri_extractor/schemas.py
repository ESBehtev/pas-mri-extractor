"""
Pydantic-схемы результата.

LLM возвращает только:
- case_info
- extracted_features
- evidence

Скоринг и рекомендации добавляются кодом после адаптации результата.
"""

from typing import Literal

from pydantic import BaseModel, Field


InvasionType = Literal["none", "accreta", "increta", "percreta"]
InvasionConfidence = Literal["absent", "possible", "probable", "definite", "unclear"]
FeatureStatus = Literal["absent", "possible", "probable", "present"]
RiskGroup = Literal["low", "moderate", "high"]


class CaseInfo(BaseModel):
    gestational_week: int | None = None
    previous_cs_count: int | None = None


class InvasionFeatures(BaseModel):
    type: InvasionType = "none"
    confidence: InvasionConfidence = "absent"


class AnatomyFeatures(BaseModel):
    bladder_involvement: FeatureStatus = "absent"
    parametrium_involvement: FeatureStatus = "absent"
    posterior_wall_involvement: FeatureStatus = "absent"


class PlacentaLocationFeatures(BaseModel):
    placenta_previa: FeatureStatus = "absent"
    anterior_placenta: FeatureStatus = "absent"


class MRISignsFeatures(BaseModel):
    retroplacental_vessels: FeatureStatus = "absent"
    lacunae: FeatureStatus = "absent"
    uterine_wall_thinning: FeatureStatus = "absent"
    uterine_hernia_or_bulging: FeatureStatus = "absent"


class ClinicalContextFeatures(BaseModel):
    preoperative_bleeding: FeatureStatus = "absent"


class ExtractedFeatures(BaseModel):
    invasion: InvasionFeatures = Field(default_factory=InvasionFeatures)
    anatomy: AnatomyFeatures = Field(default_factory=AnatomyFeatures)
    placenta_location: PlacentaLocationFeatures = Field(
        default_factory=PlacentaLocationFeatures
    )
    mri_signs: MRISignsFeatures = Field(default_factory=MRISignsFeatures)
    clinical_context: ClinicalContextFeatures = Field(
        default_factory=ClinicalContextFeatures
    )


class Evidence(BaseModel):
    positive_findings: list[str] = Field(default_factory=list)
    uncertain_findings: list[str] = Field(default_factory=list)
    negative_findings: list[str] = Field(default_factory=list)


class MRIExtractionResult(BaseModel):
    case_info: CaseInfo = Field(default_factory=CaseInfo)
    extracted_features: ExtractedFeatures = Field(default_factory=ExtractedFeatures)
    evidence: Evidence = Field(default_factory=Evidence)


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