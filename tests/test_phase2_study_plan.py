"""No-GPU tests of the execution-blocked paired study-plan contract."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase2_study_plan", ROOT / "scripts/phase2_study_plan.py")
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


class StudyPlanTests(unittest.TestCase):
    def test_pilot_pair_and_unsafe_execution_blocked(self):
        p = planner.make_plan(planner.DEFAULT_STUDY, "pilot")
        self.assertEqual(len(p["jobs"]), 2)
        self.assertEqual([x["algorithm"] for x in p["jobs"]], ["grpo", "ppo-mc-v1"])
        self.assertFalse(p["execution_allowed"])
        self.assertEqual({x["run_seed"] for x in p["jobs"]}, {9001})
        self.assertEqual({x["training_steps"] for x in p["jobs"]}, {20})
        self.assertTrue(p["open_qualification_gates"])
        self.assertEqual(p["legacy_baseline"]["correct"], 26)

    def test_campaign_count_groups_and_stage_namespace_uniqueness(self):
        p = planner.make_plan(planner.DEFAULT_STUDY, "campaign")
        self.assertEqual(len(p["jobs"]), 10)
        self.assertEqual({j["run_seed"] for j in p["jobs"]}, set(range(1001, 1006)))
        stages = [s for j in p["jobs"] for s in j["stages"]]
        self.assertEqual(len(stages), 30)
        self.assertEqual(len({s["run_id"] for s in stages}), 30)
        self.assertEqual(len({s["wandb"]["id"] for s in stages}), 30)
        self.assertEqual(len({s["output"] for s in stages}), 30)
        self.assertEqual(len({v for s in stages for v in s["logs"].values()}), 90)
        self.assertEqual({s["wandb"]["group"] for s in stages}, {p["wandb_group"]})
        self.assertEqual({j["host_memory_mib"] for j in p["jobs"]}, {196608})
        self.assertEqual(p["evaluation_panel"], "NEW_UNTOUCHED_256_REQUIRED")
        self.assertFalse(p["automatically_start_campaign_after_pilot"])

    def test_streams_are_paired_but_not_identical_across_algorithms(self):
        p = planner.make_plan("sample-lab", "pilot")
        a, b = p["jobs"]
        sa, sb = a["proposed_seed_streams_not_wired"], b["proposed_seed_streams_not_wired"]
        self.assertEqual(sa["paired_question_order"], sb["paired_question_order"])
        self.assertNotEqual(sa["python"], sb["python"])
        self.assertIsNone(sa["critic_init"])
        self.assertIsNotNone(sb["critic_init"])

    def test_validate_rejects_duplicate_wandb_and_mislabelled_launchable(self):
        p = planner.make_plan("sample-lab", "pilot")
        p["jobs"][1]["stages"][0]["wandb"]["id"] = p["jobs"][0]["stages"][0]["wandb"]["id"]
        with self.assertRaisesRegex(ValueError, "W&B run id reused"):
            planner.validate_plan(p)
        p = planner.make_plan("sample-lab", "pilot")
        p["execution_allowed"] = True
        with self.assertRaisesRegex(ValueError, "never authorize execution"):
            planner.validate_plan(p)

    def test_validate_rejects_incomplete_pair_and_wrong_resource(self):
        p = planner.make_plan("sample-lab", "pilot")
        p["jobs"][0]["algorithm"] = "ppo-mc-v1"
        with self.assertRaisesRegex(ValueError, "incomplete GRPO/PPO pair"):
            planner.validate_plan(p)
        p = planner.make_plan("sample-lab", "pilot")
        p["jobs"][0]["host_memory_mib"] = 131072
        with self.assertRaisesRegex(ValueError, "resource mismatch"):
            planner.validate_plan(p)

    def test_validate_rejects_stage_and_log_path_collisions(self):
        p = planner.make_plan("sample-lab", "pilot")
        p["jobs"][1]["stages"][0]["run_id"] = p["jobs"][0]["stages"][0]["run_id"]
        with self.assertRaisesRegex(ValueError, "stage name reused"):
            planner.validate_plan(p)
        p = planner.make_plan("sample-lab", "pilot")
        p["jobs"][1]["stages"][0]["logs"]["trainer"] = p["jobs"][0]["stages"][0]["logs"]["trainer"]
        with self.assertRaisesRegex(ValueError, "log path reused"):
            planner.validate_plan(p)

    def test_invalid_names_and_attempts(self):
        for name in ("", "a_underscore", "Uppercase", "x" * 30):
            with self.assertRaises(ValueError):
                planner.make_plan(name, "pilot")
        with self.assertRaises(ValueError):
            planner.make_plan("lab", "pilot", 0)
        with self.assertRaises(ValueError):
            planner.make_plan("lab", "pilot", 100)

    def test_cli_no_overwrite_and_no_cloud_import(self):
        source = (ROOT / "scripts/phase2_study_plan.py").read_text()
        self.assertNotIn("import modal", source)
        self.assertNotIn("subprocess.Popen", source)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"pilot.json"
            p = planner.make_plan("lab", "pilot")
            with output.open("x") as f:
                f.write("existing")
            with self.assertRaises(FileExistsError):
                with output.open("x") as f:
                    f.write("replacement")
            self.assertEqual(output.read_text(), "existing")
            self.assertEqual(len(p["jobs"]), 2)


if __name__ == "__main__":
    unittest.main()
