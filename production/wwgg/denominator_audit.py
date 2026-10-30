"""Check the exact existing WWgg generator normalization contract.

The workflow uses numpy.sum in the input dtype, then str(numpy_scalar).
Keep an independent float64 sum for diagnostics, not as a different oracle.
"""
import hashlib
import math
import numpy as np


def snapshot(weights):
    values=np.asarray(weights)
    if values.ndim!=1 or values.dtype.kind!='f' or not np.isfinite(values).all():
        raise ValueError('Invalid input generator weights')
    total=np.sum(values)
    if not np.isfinite(total):raise ValueError('Nonfinite generator sum')
    return {'rows':len(values),'dtype':str(values.dtype),
            'sha256':hashlib.sha256(values.tobytes()).hexdigest(),
            'numpy_sum':float(total),'serialized':str(total),
            'reference_fsum':math.fsum(float(x) for x in values),
            'sum_abs':math.fsum(abs(float(x)) for x in values)}


def compare(saved, expected):
    actual=float(saved)
    serialized_reference=float(expected['serialized'])
    result={'input':expected,'saved':str(saved),
            'difference_from_serialized_numpy':actual-serialized_reference,
            'difference_from_fsum':actual-expected['reference_fsum'],
            'status':'PASS' if math.isfinite(actual) and actual==serialized_reference else 'FAIL'}
    return result
