import unittest
from experiment.matched_grading import grade

class GradingTests(unittest.TestCase):
    def test_rational_equivalence_and_untrusted_reward_bits(self):
        result=grade([{'case_id':'a','answer':'1/2'},{'case_id':'b','answer':'3'}],
                     [{'case_id':'a','final_answer':'.5','reward':0},{'case_id':'b','final_answer':'4','reward':1}])
        self.assertEqual(result['correct'],1);self.assertEqual(result['denominator'],2)
    def test_units_malformed_extraction_and_missing_cases_require_review(self):
        result=grade([{'case_id':'a','answer':'3','unit':'m'}],[{'case_id':'a','final_answer':'3'}])
        self.assertEqual(result['adjudication_required'],1)
        with self.assertRaises(ValueError):grade([{'case_id':'a','answer':'1'}],[])
