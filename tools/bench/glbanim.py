"""Minimal glTF/GLB animation reader + forward kinematics (numpy, no Blender).

Only what the bench needs: node TRS, skins, LINEAR/STEP channels (CUBICSPLINE
sampled at its keys' values), world joint positions per frame.
"""

import json
import struct

import numpy as np

CT = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32, 5122: np.int16, 5120: np.int8}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class Glb:
    def __init__(self, path):
        b = open(path, "rb").read()
        jl = struct.unpack("<I", b[12:16])[0]
        self.j = json.loads(b[20:20 + jl])
        o = 20 + jl
        self.bin = b[o + 8:] if len(b) > o else b""
        self.nodes = self.j["nodes"]
        self.parent = [-1] * len(self.nodes)
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.names = [n.get("name", f"node{i}") for i, n in enumerate(self.nodes)]
        self.order = []
        seen = set()

        def visit(i):
            if i in seen:
                return
            if self.parent[i] >= 0:
                visit(self.parent[i])
            seen.add(i)
            self.order.append(i)
        for i in range(len(self.nodes)):
            visit(i)

    def acc(self, i):
        a = self.j["accessors"][i]
        bv = self.j["bufferViews"][a["bufferView"]]
        dt = np.dtype(CT[a["componentType"]])
        n = NC[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        st = bv.get("byteStride", 0)
        if st and st != dt.itemsize * n:
            raw = np.frombuffer(self.bin, np.uint8, count=st * a["count"], offset=off).reshape(a["count"], st)
            arr = np.frombuffer(raw[:, :dt.itemsize * n].tobytes(), dt).reshape(a["count"], n)
        else:
            arr = np.frombuffer(self.bin, dt, count=a["count"] * n, offset=off).reshape(a["count"], n)
        arr = arr.astype(np.float64)
        if a.get("normalized"):
            arr = arr / np.iinfo(dt).max
        return arr

    def rest_trs(self):
        T = np.zeros((len(self.nodes), 3))
        R = np.tile([0.0, 0, 0, 1], (len(self.nodes), 1))
        S = np.ones((len(self.nodes), 3))
        for i, n in enumerate(self.nodes):
            if "matrix" in n:
                m = np.array(n["matrix"]).reshape(4, 4).T
                T[i] = m[:3, 3]
                sc = np.linalg.norm(m[:3, :3], axis=0)
                S[i] = sc
                R[i] = mat_to_quat(m[:3, :3] / sc)
            else:
                T[i] = n.get("translation", [0, 0, 0])
                R[i] = n.get("rotation", [0, 0, 0, 1])
                S[i] = n.get("scale", [1, 1, 1])
        return T, R, S

    def anim_names(self):
        return [a.get("name", f"anim{i}") for i, a in enumerate(self.j.get("animations", []))]

    def sample(self, anim, fps=30.0, times=None):
        """-> (times, T[f,n,3], R[f,n,4] xyzw, S) local TRS per frame."""
        a = self.j["animations"][anim] if isinstance(anim, int) else \
            next(x for x in self.j["animations"] if x.get("name") == anim)
        chans = []
        tmax = 0.0
        for c in a["channels"]:
            s = a["samplers"][c["sampler"]]
            t = self.acc(s["input"]).ravel()
            v = self.acc(s["output"])
            if s.get("interpolation") == "CUBICSPLINE":
                v = v.reshape(len(t), 3, -1)[:, 1]
            chans.append((c["target"].get("node"), c["target"]["path"], t, v, s.get("interpolation", "LINEAR")))
            tmax = max(tmax, t[-1])
        if times is None:
            n = max(1, int(round(tmax * fps))) + 1
            times = np.linspace(0, tmax, n)
        times = np.asarray(times, dtype=np.float64)
        if len(times) == 0:
            times = np.array([0.0])
        T0, R0, S0 = self.rest_trs()
        F = len(times)
        T = np.repeat(T0[None], F, 0)
        R = np.repeat(R0[None], F, 0)
        S = np.repeat(S0[None], F, 0)
        for node, path, t, v, interp in chans:
            if node is None or path not in ("translation", "rotation", "scale"):
                continue
            k = np.clip(np.searchsorted(t, times, side="right") - 1, 0, len(t) - 1)
            k1 = np.minimum(k + 1, len(t) - 1)
            dt = np.where(t[k1] > t[k], t[k1] - t[k], 1.0)
            u = np.clip((times - t[k]) / dt, 0, 1) if interp != "STEP" else np.zeros(F)
            if path == "rotation":
                R[:, node] = qslerp(v[k], v[k1], u)
            else:
                val = v[k] * (1 - u)[:, None] + v[k1] * u[:, None]
                (T if path == "translation" else S)[:, node] = val
        return times, T, R, S

    def world(self, T, R, S):
        """Local TRS per frame -> world 4x4 per frame and node."""
        F, N = T.shape[:2]
        L = np.zeros((F, N, 4, 4))
        L[..., :3, :3] = quat_to_mat(R) * S[..., None, :]
        L[..., :3, 3] = T
        L[..., 3, 3] = 1
        W = np.zeros_like(L)
        for i in self.order:
            p = self.parent[i]
            W[:, i] = L[:, i] if p < 0 else W[:, p] @ L[:, i]
        return W


def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qslerp(a, b, u):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    d = (a * b).sum(-1)
    b = np.where(d[:, None] < 0, -b, b)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    small = s < 1e-6
    wa = np.where(small, 1 - u, np.sin((1 - u) * th) / np.where(small, 1, s))
    wb = np.where(small, u, np.sin(u * th) / np.where(small, 1, s))
    q = a * wa[:, None] + b * wb[:, None]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_mat(q):
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q
