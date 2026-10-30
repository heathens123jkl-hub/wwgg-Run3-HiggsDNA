import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'production/wwgg'))
from denominator_audit import snapshot,compare


class Tests(unittest.TestCase):
    def test_float32_serialization_is_not_float64_sum(self):
        values=np.array([.1,.2,.3],dtype=np.float32)
        expected=snapshot(values)
        self.assertNotEqual(float(expected['serialized']),expected['reference_fsum'])
        self.assertEqual(compare(str(np.sum(values)),expected)['status'],'PASS')

    def test_reduction_order_does_not_replace_workflow_contract(self):
        values=np.array([1e8,1.,-1e8],dtype=np.float32)
        expected=snapshot(values)
        self.assertEqual(expected['reference_fsum'],1.)
        self.assertEqual(compare(str(np.sum(values)),expected)['status'],'PASS')
        self.assertEqual(compare('1.0',expected)['status'],'FAIL')

    def test_filtered_denominator_still_fails(self):
        values=np.array([2.,-1.,3.],dtype=np.float32)
        self.assertEqual(compare(str(np.sum(values[values>0])),snapshot(values))['status'],'FAIL')

    def test_empty_and_zero_and_float64(self):
        for values in (np.array([],dtype=np.float32),np.array([1.,-1.]),np.array([.1,.2],dtype=np.float64)):
            self.assertEqual(compare(str(np.sum(values)),snapshot(values))['status'],'PASS')

    def test_nonfinite_inputs_and_outputs(self):
        for x in (np.nan,np.inf):
            with self.assertRaises(ValueError):snapshot(np.array([x]))
            self.assertEqual(compare(str(x),snapshot(np.array([1.])))['status'],'FAIL')


if __name__=='__main__':unittest.main(verbosity=2)
