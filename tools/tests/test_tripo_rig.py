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


if __name__ == "__main__":
    unittest.main()
