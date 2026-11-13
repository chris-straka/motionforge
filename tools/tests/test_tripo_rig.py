import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tripo_rig  # noqa: E402


def t(seed, score, fails=0, passed=False, contract=True, rigged=True):
    return {"seed": seed, "score": score, "fails": ["f"] * fails, "pass": passed, "contract": contract, "rigged": rigged}


class PickBest(unittest.TestCase):
    def test_a_pass_beats_a_higher_score(self):
        best = tripo_rig.pick_best([t(1, 90.0, fails=1), t(2, 76.0, passed=True)])
        self.assertEqual(best["seed"], 2)

    def test_fewer_failing_findings_beat_a_higher_score(self):
        best = tripo_rig.pick_best([t(1, 61.5, fails=4), t(2, 65.8, fails=6), t(3, 52.3, fails=7)])
        self.assertEqual(best["seed"], 1)

    def test_a_clean_contract_beats_a_broken_one(self):
        best = tripo_rig.pick_best([t(1, 80.0, fails=2, contract=False), t(2, 70.0, fails=2)])
        self.assertEqual(best["seed"], 2)

    def test_ties_go_to_the_lower_seed_and_failed_rigs_are_out(self):
        best = tripo_rig.pick_best([t(3, 70.0, fails=2), t(2, 70.0, fails=2), t(1, 99.0, rigged=False)])
        self.assertEqual(best["seed"], 2)
        self.assertIsNone(tripo_rig.pick_best([t(1, 0.0, rigged=False)]))


class StepFailures(unittest.TestCase):
    def test_a_failed_animate_fails_the_try(self):
        try_ = {"pass": True, "fails": [], "animate": {"code": 3, "secs": 1.0, "error": "no hips bone found"}}
        tripo_rig.apply_step_failures(try_)
        self.assertFalse(try_["pass"])
        self.assertEqual(try_["fails"], ["animate failed: no hips bone found"])

    def test_a_good_animate_or_none_changes_nothing(self):
        for try_ in ({"pass": True, "fails": []}, {"pass": True, "fails": [], "animate": {"code": 0, "secs": 1.0}}):
            tripo_rig.apply_step_failures(try_)
            self.assertTrue(try_["pass"])
            self.assertEqual(try_["fails"], [])

    def test_fresh_removes_a_stale_output(self):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "try_s1.glb")
        open(p, "w").write("old")
        tripo_rig.fresh(p)
        self.assertFalse(os.path.exists(p))
        tripo_rig.fresh(p)  # absent: no error


if __name__ == "__main__":
    unittest.main()
