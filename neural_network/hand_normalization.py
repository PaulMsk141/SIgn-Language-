"""Faithful port of jazz's HandNormalization3d (NexJazzExtensions/Hands).

Given the 21 metric world landmarks, jazz produces a canonical `normalized3dNodes`
pose via six steps:

  1. rotate the palm normal to face -z (plus a yaw/snap so wrist->middle is upright)
  2. straighten each non-thumb finger in the XY plane (PCA line projection)
  3. enforce a planar PIP/DIP bend ratio per finger
  4. rescale all bones to fixed anatomical length ratios
  5. equalize the MCP azimuths (palm spread) about the wrist
  6. undo the step-1 rotation

Steps 2-5 are individually toggleable (jazz's HandNormalization3dOptions defaults
enable all four); steps 1 and 6 always run and cancel out when 2-5 are all off.
Input is wrist-centered first, matching jazz's ExtractInto.
"""

from __future__ import annotations

import math

import numpy as np

# MediaPipe / jazz joint indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

_PALM_IDX = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
_FINGERS_NO_THUMB = [
    [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
]
_MCP_INDICES = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
_BONES = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
]
_DEPTH = np.array([0.0, 0.0, -1.0])


def _build_weights_norm() -> np.ndarray:
    thumb = [0.35, 0.40, 0.35, 0.25]
    phal = [0.45, 0.35, 0.20]
    index = [1.2, *phal]
    middle = [1.1, *phal]
    ring = [1.1, *phal]
    pinky = [1.4, *phal]
    scales = [1.1, 0.95, 1.05, 1.02, 0.75]
    w = []
    for finger, s in zip((thumb, index, middle, ring, pinky), scales):
        w += [max(0.0, v * s) for v in finger]
    w = np.array(w, dtype=np.float64)
    return w / w.sum()


_WEIGHTS_NORM = _build_weights_norm()


# --- small vector helpers -------------------------------------------------
def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n >= eps else np.zeros_like(v)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with R@a = b (Rodrigues, 180-degree fallback)."""
    a, b = _unit(a), _unit(b)
    cr = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(cr))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        ax = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0.0, 1, 0])
        v = _unit(np.cross(a, ax))
        K = _skew(v)
        return np.eye(3) + 2.0 * (K @ K)
    K = _skew(cr / s)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _rodrigues(axis_unit: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis_unit
    c, s, C = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _apply(M: np.ndarray, pts: np.ndarray, center: np.ndarray) -> np.ndarray:
    return (pts - center) @ M.T + center


# --- Step 1: rotate palm normal to depth, yaw/snap upright ----------------
def _step1(P: np.ndarray):
    v1 = P[INDEX_MCP] - P[WRIST]
    v2 = P[PINKY_MCP] - P[INDEX_MCP]
    n_orig = np.cross(v1, v2)
    nrm = np.linalg.norm(n_orig)
    if nrm < 1e-8:
        v1 = P[PINKY_MCP] - P[WRIST]
        v2 = P[INDEX_MCP] - P[WRIST]
        n_orig = np.cross(v1, v2)
        nrm = np.linalg.norm(n_orig)
        if nrm < 1e-8:
            return P.copy(), np.eye(3), np.eye(3), P.mean(axis=0)
    n_orig = n_orig / nrm
    if np.dot(n_orig, _DEPTH) < 0:
        n_orig = -n_orig

    palm_center = P[_PALM_IDX].mean(axis=0)
    R = _rot_a_to_b(n_orig, _DEPTH)
    Prot = _apply(R, P, palm_center)

    # Post-rotation facing guard
    cr = np.cross(Prot[INDEX_MCP] - Prot[WRIST], Prot[PINKY_MCP] - Prot[INDEX_MCP])
    nrn = np.linalg.norm(cr)
    if nrn > 1e-12:
        cr = cr / nrn
        if np.dot(cr, _DEPTH) < 0:
            ax = np.array([1.0, 0, 0]) if abs(_DEPTH[0]) < 0.9 else np.array([0.0, 1, 0])
            axis = _unit(np.cross(_DEPTH, ax))
            Rfix = _rodrigues(axis, math.pi)
            Prot = _apply(Rfix, Prot, palm_center)
            R = Rfix @ R

    # Yaw + snap so wrist->middle is vertical in-plane
    y_plane = np.array([0.0, 1, 0]) - np.dot(np.array([0.0, 1, 0]), _DEPTH) * _DEPTH
    if np.linalg.norm(y_plane) < 1e-8:
        fb = np.array([1.0, 0, 0])
        y_plane = fb - np.dot(fb, _DEPTH) * _DEPTH
    y_plane = _unit(y_plane)
    x_plane = _unit(np.cross(y_plane, _DEPTH))

    w2m = Prot[MIDDLE_MCP] - Prot[WRIST]
    w2m_proj = w2m - np.dot(w2m, _DEPTH) * _DEPTH
    wlen = np.linalg.norm(w2m_proj)
    if wlen > 1e-10:
        w2m_u = w2m_proj / wlen
        cos_t = float(np.clip(np.dot(w2m_u, y_plane), -1, 1))
        sin_t = float(np.clip(np.dot(w2m_u, x_plane), -1, 1))
        theta = math.atan2(sin_t, cos_t)
        Ryaw = _rodrigues(_DEPTH, -theta)
        Prot = _apply(Ryaw, Prot, palm_center)
        R = Ryaw @ R

        w2m = Prot[MIDDLE_MCP] - Prot[WRIST]
        y_src = w2m - np.dot(w2m, _DEPTH) * _DEPTH
        ysn = np.linalg.norm(y_src)
        if ysn > 1e-12:
            y_src = y_src / ysn
            x_src = _unit(np.cross(y_src, _DEPTH))
            B_tgt = np.column_stack([x_plane, y_plane, _DEPTH])
            B_srcT = np.array([x_src, y_src, _DEPTH])
            Rsnap = B_tgt @ B_srcT
            Prot = _apply(Rsnap, Prot, palm_center)
            R = Rsnap @ R

            w2m_f = Prot[MIDDLE_MCP] - Prot[WRIST]
            w2m_f_proj = w2m_f - np.dot(w2m_f, _DEPTH) * _DEPTH
            if np.dot(w2m_f_proj, y_plane) < 0:
                Rpi = _rodrigues(_DEPTH, math.pi)
                Prot = _apply(Rpi, Prot, palm_center)
                R = Rpi @ R

    return Prot, R, R.T, palm_center


# --- Step 2: straighten fingers in XY -------------------------------------
def _fit_line_dir_xy(pts_xy: np.ndarray) -> np.ndarray:
    mean = pts_xy.mean(axis=0)
    d = pts_xy - mean
    cov = d.T @ d
    w, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
    return vecs[:, -1]                       # principal direction


def _step2(P: np.ndarray) -> np.ndarray:
    Q = P.copy()
    for idxs in _FINGERS_NO_THUMB:
        origin_xy = P[idxs[0], :2]
        d = _fit_line_dir_xy(P[idxs, :2])
        tipv = P[idxs[3], :2] - P[idxs[0], :2]
        if np.dot(d, tipv) < 0:
            d = -d
        nd = np.linalg.norm(d)
        if nd < 1e-8:
            continue
        u = d / nd
        for j in idxs:
            t = u[0] * (P[j, 0] - origin_xy[0]) + u[1] * (P[j, 1] - origin_xy[1])
            Q[j, 0] = origin_xy[0] + t * u[0]
            Q[j, 1] = origin_xy[1] + t * u[1]
            Q[j, 2] = P[j, 2]
    return Q


# --- Step 3: planar PIP/DIP bend ratio ------------------------------------
def _nrm2(u):
    return math.hypot(u[0], u[1])


def _unit2(u):
    l = _nrm2(u)
    return (u[0] / l, u[1] / l) if l > 1e-9 else (u[0], u[1])


def _wrap_pi(x):
    y = (x + math.pi) % (2 * math.pi)
    if y < 0:
        y += 2 * math.pi
    return y - math.pi


def _signed_angle2(u, v):
    ua, ub = _unit2(u), _unit2(v)
    c = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
    s = ua[0] * ub[1] - ua[1] * ub[0]
    return math.atan2(s, c)


def _rot_dir2(d, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (c * d[0] - s * d[1], s * d[0] + c * d[1])


def _ang_dist(a, b):
    return abs(_wrap_pi(a - b))


def _build_and_measure(a, g, vMP, pip2, L_PD, L_DT):
    uMP = _unit2(vMP)
    uPDn = _rot_dir2(uMP, a)
    uDTn = _rot_dir2(uPDn, g)
    vPDn = (uPDn[0] * L_PD, uPDn[1] * L_PD)
    vDTn = (uDTn[0] * L_DT, uDTn[1] * L_DT)
    dip2 = (pip2[0] + vPDn[0], pip2[1] + vPDn[1])
    tip2 = (dip2[0] + vDTn[0], dip2[1] + vDTn[1])
    alpha = _signed_angle2(vMP, vPDn)
    gamma = _signed_angle2(uPDn, uDTn)
    wedge = _signed_angle2(vMP, (vPDn[0] + vDTn[0], vPDn[1] + vDTn[1]))
    return dip2, tip2, alpha, gamma, wedge


def _step3(P: np.ndarray) -> np.ndarray:
    Q = P.copy()
    k = 1.25
    eps = 1e-9
    tol = math.radians(1e-3)
    for idxs in _FINGERS_NO_THUMB:
        mcp, pip, dip, tip = idxs
        o = P[mcp]
        # planar 2D coords are (z - o.z, y - o.y); mcp is origin (0, 0)
        pip2 = (P[pip, 2] - o[2], P[pip, 1] - o[1])
        dip2 = (P[dip, 2] - o[2], P[dip, 1] - o[1])
        tip2 = (P[tip, 2] - o[2], P[tip, 1] - o[1])
        vMP = pip2
        vPD = (dip2[0] - pip2[0], dip2[1] - pip2[1])
        vDT = (tip2[0] - dip2[0], tip2[1] - dip2[1])
        vPT = (vPD[0] + vDT[0], vPD[1] + vDT[1])
        L_PD, L_DT, L_MP = _nrm2(vPD), _nrm2(vDT), _nrm2(vMP)
        if min(L_MP, L_PD, L_DT) < 1e-6:
            continue

        alpha_b = _signed_angle2(vMP, vPD)
        gamma_b = _signed_angle2(vPD, vDT)
        wedge_b = _signed_angle2(vMP, vPT)
        same_sign = alpha_b * gamma_b >= 0
        ratio_ok = (abs(gamma_b) <= eps) or ((alpha_b / gamma_b) >= k - 1e-12)
        if same_sign and ratio_ok:
            a0, g0 = alpha_b, gamma_b
        else:
            g_base = wedge_b / (k + 1.0)
            a_base = k * g_base
            dev0 = _ang_dist(a_base, alpha_b) + _ang_dist(g_base, gamma_b)
            dev1 = _ang_dist(-a_base, alpha_b) + _ang_dist(-g_base, gamma_b)
            a0, g0 = (a_base, g_base) if dev0 <= dev1 else (-a_base, -g_base)

        sum_curr = a0 + g0
        if abs(sum_curr) > 1e-8:
            denom = sum_curr
        else:
            denom = 1e-8 if wedge_b >= 0 else -1e-8
        s = max(0.25, min(4.0, wedge_b / denom))
        a_s, g_s = a0 * s, g0 * s

        dip2n, tip2n, alpha_m, gamma_m, wedge_m = _build_and_measure(a_s, g_s, vMP, pip2, L_PD, L_DT)

        err = wedge_b - wedge_m
        if abs(err) > tol:
            ds = 1e-3
            _, _, _, _, wedge_plus = _build_and_measure(a0 * (s + ds), g0 * (s + ds), vMP, pip2, L_PD, L_DT)
            deriv = (wedge_plus - wedge_m) / ds
            if abs(deriv) > 1e-6:
                s_new = max(0.1, min(10.0, s + err / deriv))
                a_s, g_s = a0 * s_new, g0 * s_new
                dip2n, tip2n, alpha_m, gamma_m, wedge_m = _build_and_measure(a_s, g_s, vMP, pip2, L_PD, L_DT)

        if alpha_m * gamma_m < 0:
            a_s, g_s = -a_s, -g_s
            dip2n, tip2n, alpha_m, gamma_m, wedge_m = _build_and_measure(a_s, g_s, vMP, pip2, L_PD, L_DT)

        if not ((abs(gamma_m) <= eps) or ((alpha_m / gamma_m) >= k - 1e-12)):
            g_base = wedge_b / (k + 1.0)
            a_base = k * g_base
            dip2n, tip2n, alpha_m, gamma_m, wedge_m = _build_and_measure(a_base, g_base, vMP, pip2, L_PD, L_DT)

        # back to 3D, preserving original x
        Q[dip] = [P[dip, 0], o[1] + dip2n[1], o[2] + dip2n[0]]
        Q[tip] = [P[tip, 0], o[1] + tip2n[1], o[2] + tip2n[0]]
    return Q


# --- Step 4: enforce anatomical bone-length ratios ------------------------
def _step4(P: np.ndarray) -> np.ndarray:
    total_len = sum(np.linalg.norm(P[j] - P[i]) for i, j in _BONES)
    targets = _WEIGHTS_NORM * total_len
    Q = P.copy()
    Q[WRIST] = P[WRIST]
    for b, (i, j) in enumerate(_BONES):
        d = P[j] - P[i]
        n = np.linalg.norm(d)
        u = d / n if n >= 1e-8 else np.array([1.0, 0, 0])
        Q[j] = Q[i] + u * targets[b]
    return Q


# --- Step 5: equalize MCP azimuths ----------------------------------------
def _unwrap(angles):
    out = list(angles)
    for i in range(1, len(out)):
        while out[i] - out[i - 1] > math.pi:
            out[i] -= 2 * math.pi
        while out[i] - out[i - 1] < -math.pi:
            out[i] += 2 * math.pi
    return out


def _step5(P: np.ndarray) -> np.ndarray:
    Q = P.copy()
    w = P[WRIST, :2]
    radii, thetas = [], []
    for i in _MCP_INDICES:
        dx, dy = P[i, 0] - w[0], P[i, 1] - w[1]
        radii.append(math.hypot(dx, dy))
        thetas.append(math.atan2(dy, dx))
    if any(r < 1e-9 for r in radii):
        return Q

    th = _unwrap(thetas)
    if th[3] < th[0]:
        th = [t + math.pi for t in _unwrap([t + math.pi for t in thetas])]
        th = [t - math.pi for t in th]
    span = th[3] - th[0]
    if span < 1e-9:
        return Q

    delta = span / 3.0
    targets = [th[0] + m * delta for m in range(4)]
    for t in range(4):
        r = radii[t]
        tx = w[0] + r * math.cos(targets[t])
        ty = w[1] + r * math.sin(targets[t])
        d0 = tx - P[_MCP_INDICES[t], 0]
        d1 = ty - P[_MCP_INDICES[t], 1]
        for j in _FINGERS_NO_THUMB[t]:
            Q[j, 0] += d0
            Q[j, 1] += d1
    return Q


def _step6(Q: np.ndarray, RInv: np.ndarray, palm_center: np.ndarray) -> np.ndarray:
    return _apply(RInv, Q, palm_center)


def normalize_hand_3d(
    world_kp: np.ndarray,
    align_fingers: bool = True,
    normalize_finger_bend: bool = True,
    normalize_bone_length: bool = True,
    normalize_palm_width: bool = True,
) -> np.ndarray:
    """jazz normalized3dNodes for one hand. ``world_kp`` is [21, 3] world landmarks."""
    P = np.asarray(world_kp, dtype=np.float64).copy()
    P = P - P[WRIST]                       # jazz ExtractInto wrist-centers first
    P, R, RInv, palm_center = _step1(P)
    if align_fingers:
        P = _step2(P)
    if normalize_finger_bend:
        P = _step3(P)
    if normalize_bone_length:
        P = _step4(P)
    if normalize_palm_width:
        P = _step5(P)
    P = _step6(P, RInv, palm_center)
    return P.astype(np.float32)


def _self_check() -> None:
    rng = np.random.default_rng(0)
    kp = rng.standard_normal((21, 3)).astype(np.float32) * 0.05
    # Steps 1 and 6 are inverse rotations: with 2-5 off, output == wrist-centered input.
    off = normalize_hand_3d(kp, False, False, False, False)
    assert np.allclose(off, kp - kp[WRIST], atol=1e-4), "step1/step6 do not round-trip"
    full = normalize_hand_3d(kp)
    assert full.shape == (21, 3) and np.isfinite(full).all()
    # Determinism
    assert np.allclose(full, normalize_hand_3d(kp))
    print("hand_normalization self-check OK: round-trip and full pipeline valid")


if __name__ == "__main__":
    _self_check()
