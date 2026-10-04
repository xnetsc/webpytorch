import gc
import weakref

import numpy as np

from webtorch._core import Tensor, cat


def test_inference_concat_does_not_retain_older_cache_rows():
    first = Tensor(np.ones((1, 1, 2), np.float32))
    old_ref = weakref.ref(first)
    cache = cat((first, Tensor(np.zeros((1, 1, 2), np.float32))), axis=1)
    assert cache._prev == ()
    del first
    gc.collect()
    assert old_ref() is None


def test_gradient_concat_keeps_ordered_parents():
    first = Tensor(np.ones((1, 1, 2), np.float32), requires_grad=True)
    second = Tensor(np.zeros((1, 1, 2), np.float32))
    cache = cat((first, second), axis=1)
    assert cache.requires_grad
    assert cache._prev == (first, second)
