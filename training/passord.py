"""Frozen PASS-Ord routing. No training, calibration or parameter selection."""
from __future__ import annotations
import numpy as np

def route_pass_predictions(cnn_p3, cnn_p5, cnn_progress, rf_p5):
    """Inputs are already averaged within passes/across CNN seeds, in identical order.

    Class columns use zero-based labels. CNN wins maximum-score ties.
    Returned scores are not jointly calibrated probabilities of correctness.
    """
    p3,p5,progress,rf=map(lambda x:np.asarray(x,dtype=float),
                         (cnn_p3,cnn_p5,cnn_progress,rf_p5))
    if progress.ndim!=1 or p3.shape!=(len(progress),3) or p5.shape!=(len(progress),5) or rf.shape!=p5.shape:
        raise ValueError('Expected (n,3), (n,5), (n,), (n,5) in matching pass order')
    for p in (p3,p5,rf):
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
            raise ValueError('Class inputs must be finite normalized probabilities')
    if not np.isfinite(progress).all(): raise ValueError('Nonfinite progress')
    choose_cnn=p5.max(1)>=rf.max(1)
    return {'pred_cls3':p3.argmax(1),
            'pred_cls5':np.where(choose_cnn,p5.argmax(1),rf.argmax(1)),
            'pred_progress':progress.copy(),'five_class_uses_cnn':choose_cnn}
