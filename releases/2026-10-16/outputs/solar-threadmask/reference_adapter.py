"""NumPy adapter for unchanged public v6 scoring-function parity checks only."""
import types
import numpy as np


class TensorAdapter(np.ndarray):
    def sum(self,*args,dim=None,**kwargs):
        if dim is not None:kwargs['axis']=dim
        return super().sum(*args,**kwargs)


class Table:
    def __init__(self,matrices):self.matrices=matrices
    def itertuples(self,index=False):
        for m in self.matrices:
            yield types.SimpleNamespace(iou_matrix=np.asarray(m,dtype=np.float32).view(TensorAdapter),n_gt=m.shape[0],n_pred=m.shape[1])
