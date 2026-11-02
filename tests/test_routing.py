"""Check the task-specific inference contract without loading models."""
import unittest
import numpy as np
from training.passord import route_pass_predictions


class RoutingContract(unittest.TestCase):
    def test_ties_choose_cnn_and_other_tasks_are_unchanged(self):
        p3=np.array([[.8,.1,.1],[.1,.2,.7]])
        cnn=np.array([[.7,.1,.1,.05,.05],[.6,.1,.1,.1,.1]])
        rf=np.array([[.1,.7,.1,.05,.05],[.05,.8,.05,.05,.05]])
        progress=np.array([.2,.8])
        out=route_pass_predictions(p3,cnn,progress,rf)
        np.testing.assert_array_equal(out['pred_cls3'],[0,2])
        np.testing.assert_array_equal(out['pred_cls5'],[0,1])
        np.testing.assert_array_equal(out['pred_progress'],progress)
        np.testing.assert_array_equal(out['five_class_uses_cnn'],[True,False])
        out['pred_progress'][0]=0
        self.assertEqual(progress[0],.2)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            route_pass_predictions([[1,0,0]],[[1,1,0,0,0]],[.2],[[1,0,0,0,0]])

    def test_misaligned_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            route_pass_predictions([[1,0,0]],[[1,0,0,0,0]],[.2,.3],[[1,0,0,0,0]])

if __name__=='__main__':
    unittest.main()
