import unittest

from arc_prize.compare_ttt_runs import compare


def report(score: int, accuracy: float, solved: list[str], *, transforms=None, extra=0):
    return {
        "checkpoint": "/checkpoints/best.pt",
        "ttt": {
            "augmentation_transforms": transforms or ["identity"],
            "extra_examples_loaded": extra,
            "extra_examples_rejected": 0,
            "max_examples_per_task": None,
        },
        "ttt_metrics": {
            "score": score,
            "cell_accuracy": accuracy,
        },
        "predictions": {
            task_id: [
                {
                    "ttt_exact": task_id in solved,
                    "ttt_candidate_examples": 96,
                    "ttt_examples": 48,
                    "adaptation_pool_size": 4,
                }
            ]
            for task_id in ("a", "b", "c")
        },
    }


class CompareTTTRunsTest(unittest.TestCase):
    def test_metrics_and_task_differences(self) -> None:
        result = compare(
            {
                "baseline": report(1, 0.8, ["a"]),
                "augmentation": report(
                    2,
                    0.85,
                    ["a", "b"],
                    transforms=["identity", "transpose"],
                ),
                "cheat": report(2, 0.9, ["b", "c"], extra=20),
            }
        )
        self.assertEqual(result["runs"]["augmentation"]["score_delta_vs_baseline"], 1)
        self.assertEqual(result["runs"]["cheat"]["extra_examples_loaded"], 20)
        self.assertEqual(
            result["runs"]["baseline"]["adaptation"]["derived_ttt_items_total"],
            144,
        )
        self.assertEqual(
            result["runs"]["baseline"]["adaptation"][
                "derived_ttt_candidates_before_cap_total"
            ],
            288,
        )
        self.assertEqual(result["task_differences"]["augmentation_fixed_vs_baseline"], ["b"])
        self.assertEqual(result["task_differences"]["cheat_regressed_vs_baseline"], ["a"])

    def test_rejects_different_checkpoints(self) -> None:
        baseline = report(0, 0.0, [])
        cheat = report(0, 0.0, [])
        cheat["checkpoint"] = "/checkpoints/other.pt"
        with self.assertRaisesRegex(ValueError, "same checkpoint"):
            compare(
                {
                    "baseline": baseline,
                    "augmentation": report(0, 0.0, []),
                    "cheat": cheat,
                }
            )

    def test_uses_refined_stage_when_all_runs_refine(self) -> None:
        reports = {
            name: report(1, 0.8, ["a"])
            for name in ("baseline", "augmentation", "cheat")
        }
        reports["augmentation"]["refinement_rounds"] = 2
        reports["baseline"]["refinement_rounds"] = 2
        reports["cheat"]["refinement_rounds"] = 2
        for run in reports.values():
            run["refined_metrics"] = {"score": 2, "cell_accuracy": 0.9}
            run["predictions"]["b"][0]["refined_exact"] = True
        result = compare(reports)
        self.assertEqual(result["runs"]["augmentation"]["final_stage"], "refined")
        self.assertEqual(result["runs"]["augmentation"]["final_metrics"]["score"], 2)


if __name__ == "__main__":
    unittest.main()
