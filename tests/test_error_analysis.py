import unittest

from scripts.error_analysis import (
    analyze_case,
    analyze_records,
    extract_rule_features,
    markdown_report,
    top_patterns,
)


def make_record(
    case_id,
    actual_blood_loss,
    predicted_blood_loss,
    actual_readiness="2",
    predicted_readiness="3",
    invasion_type="increta",
    bladder="absent",
    placenta_previa="present",
    anterior_placenta="present",
    vessels="present",
    lacunae="present",
    percreta_suspicion="possible",
):
    return {
        "case_id": case_id,
        "status": "success",
        "actual": {
            "blood_loss_ml": actual_blood_loss,
            "readiness_level": actual_readiness,
        },
        "llm_risk": {
            "risk_assessment": {
                "estimated_blood_loss_ml": predicted_blood_loss,
            },
            "readiness": {
                "level": predicted_readiness,
            },
        },
        "rule_based": {
            "result": {
                "case_info": {
                    "previous_cs_count": 2,
                },
                "extracted_features": {
                    "invasion": {
                        "type": invasion_type,
                    },
                    "anatomy": {
                        "bladder_involvement": bladder,
                    },
                    "placenta_location": {
                        "placenta_previa": placenta_previa,
                        "anterior_placenta": anterior_placenta,
                    },
                    "mri_signs": {
                        "retroplacental_vessels": vessels,
                        "lacunae": lacunae,
                        "uterine_wall_thinning": "absent",
                        "uterine_hernia_or_bulging": "absent",
                    },
                },
                "suspicion": {
                    "percreta_suspicion": percreta_suspicion,
                },
            },
        },
    }


class ErrorAnalysisTest(unittest.TestCase):
    def test_analyze_case_computes_blood_loss_and_readiness_errors(self) -> None:
        record = make_record("case-1", 1000, 1800, "2", "3")

        result = analyze_case(record)

        self.assertEqual(result["blood_loss_error"], 800)
        self.assertEqual(result["abs_blood_loss_error"], 800)
        self.assertEqual(result["readiness_error"], 1)
        self.assertEqual(result["prediction_group"], "bad_predictions")

    def test_extract_rule_features_reads_nested_rule_based_result(self) -> None:
        record = make_record("case-1", 1000, 1800)

        features = extract_rule_features(record)

        self.assertEqual(features["placenta_previa"], "present")
        self.assertEqual(features["retroplacental_vessels"], "present")
        self.assertEqual(features["lacunae"], "present")
        self.assertEqual(features["percreta_suspicion"], "possible")
        self.assertEqual(features["previous_cs_count"], 2)
        self.assertEqual(features["invasion.type"], "increta")

    def test_analyze_records_summarizes_features_and_patterns(self) -> None:
        records = [
            make_record("over-1", 1000, 1900, lacunae="present"),
            make_record("over-2", 1200, 2100, lacunae="present"),
            make_record(
                "neutral-1",
                1500,
                1600,
                "2",
                "2",
                lacunae="absent",
                percreta_suspicion="absent",
            ),
            make_record(
                "under-1",
                2600,
                1700,
                "3",
                "2",
                bladder="possible",
                lacunae="absent",
                percreta_suspicion="absent",
            ),
        ]

        analysis = analyze_records(records)

        self.assertEqual(analysis["summary"]["n_cases"], 4)
        self.assertEqual(analysis["summary"]["n_bad_predictions"], 3)
        self.assertEqual(analysis["summary"]["n_good_predictions"], 1)
        self.assertEqual(analysis["summary"]["blood_loss_mae_ml"], 700)
        self.assertEqual(
            analysis["discriminative_analysis"]["group_counts"]["overestimation"],
            2,
        )
        self.assertEqual(
            analysis["discriminative_analysis"]["group_counts"]["neutral"],
            1,
        )
        self.assertEqual(
            analysis["discriminative_analysis"]["group_counts"]["underestimation"],
            1,
        )
        self.assertEqual(analysis["severe_cases"]["n_cases"], 1)
        self.assertEqual(analysis["severe_cases"]["mean_prediction"], 1700)
        self.assertEqual(analysis["low_risk_cases"]["n_cases"], 1)
        self.assertEqual(analysis["low_risk_cases"]["mean_prediction"], 1900)

        features = {
            item["feature"]: item for item in analysis["feature_summary"]
        }
        self.assertIn("placenta_previa=present", features)
        self.assertIn("retroplacental_vessels=present", features)

        discriminative = {
            item["feature"]: item
            for item in analysis["discriminative_analysis"]["features"]
        }
        self.assertTrue(
            discriminative["placenta_previa=present"]["ignored_as_driver"]
        )
        over_drivers = [
            item["feature"]
            for item in analysis["discriminative_analysis"][
                "top_enriched_overestimation_drivers"
            ]
        ]
        self.assertIn("lacunae=present", over_drivers)
        self.assertNotIn("placenta_previa=present", over_drivers)

        over_patterns = top_patterns(analysis["cases"], "over", limit=50)
        self.assertTrue(
            any(
                "placenta_previa=present" in item["pattern"]
                and "retroplacental_vessels=present" in item["pattern"]
                for item in over_patterns
            )
        )

    def test_markdown_report_contains_requested_sections(self) -> None:
        analysis = analyze_records(
            [
                make_record("over-1", 1000, 1900),
                make_record("under-1", 2400, 1500, "3", "2"),
            ]
        )

        report = markdown_report(analysis)

        self.assertIn("# Error analysis", report)
        self.assertIn("Blood loss MAE", report)
        self.assertIn("Most common overestimation patterns", report)
        self.assertIn("Most common underestimation patterns", report)
        self.assertIn("Top enriched overestimation drivers", report)
        self.assertIn("Top enriched underestimation drivers", report)
        self.assertIn("Severe cases", report)
        self.assertIn("Low-risk cases", report)
        self.assertIn("Recommendations", report)


if __name__ == "__main__":
    unittest.main()
