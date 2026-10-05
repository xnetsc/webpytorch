import numpy as np

from webtorch import _core as C


def test_only_a_weight_stored_at_half_width_is_held_at_half_width(monkeypatch):
    """Half-width storage must not change what the model computes with: a float16 weight is
    computed at float16 either way, a float32 one would be silently narrowed."""
    monkeypatch.setattr(C, "half_weight_ok", lambda K, N: True)
    monkeypatch.setattr(C, "pack_half_weight", lambda w: ("packed", np.asarray(w).dtype))
    w = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
    assert C.half_weight(w.astype(np.float16)) == ("packed", np.dtype(np.float16))
    assert C.half_weight(w) is None                      # float32 stays float32
    assert C.half_weight(w.astype(np.float64)) is None
    assert C.half_weight(w.astype(np.float16).reshape(-1)) is None   # not a matrix


def test_a_backend_that_cannot_hold_it_is_asked_first(monkeypatch):
    monkeypatch.setattr(C, "half_weight_ok", lambda K, N: False)

    def refuse(_w):
        raise AssertionError("packed something the backend had already said it cannot hold")
    monkeypatch.setattr(C, "pack_half_weight", refuse)
    assert C.half_weight(np.zeros((8, 16), np.float16)) is None


def test_k4_texels_hold_four_consecutive_k_of_one_output_column():
    N, K = 6, 12                                         # a checkpoint's (out, in)
    w = np.arange(N * K, dtype=np.float32).reshape(N, K)
    t = C._k4_texels(w)
    assert t.shape == (K // 4, N, 4) and t.flags.c_contiguous
    wt_ = w.T                                            # (K, N), as the maths wants it
    for k4 in range(K // 4):
        for j in range(N):
            assert np.array_equal(t[k4, j], wt_[4 * k4:4 * k4 + 4, j])
    # And a dot of a packed activation row against a texel column is the matmul.
    x = np.arange(3 * K, dtype=np.float32).reshape(3, K)
    xp = x.reshape(3, K // 4, 4)
    got = np.einsum("ikc,kjc->ij", xp, t)
    assert np.array_equal(got, x @ wt_)
