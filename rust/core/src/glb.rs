//! Minimal glTF 2.0 binary (GLB) reader/writer for rigged characters
//! (no dependencies, like the rest of the core).
//!
//! Scope: one embedded BIN buffer (what Blender, Tripo and rigforge
//! write), dense accessors, node TRS or matrix transforms, skins and
//! animations. Unknown JSON is carried through untouched, so a GLB
//! that only gets renamed bones keeps its materials and textures.
//! Corrupt input is an error, never a panic.

use crate::json::{self, Json};
use crate::math::{Mat3, Quat, Vec3};

const MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

pub const FLOAT: u64 = 5126;
pub const UNSIGNED_BYTE: u64 = 5121;
pub const UNSIGNED_SHORT: u64 = 5123;
pub const UNSIGNED_INT: u64 = 5125;
pub const BYTE: u64 = 5120;
pub const SHORT: u64 = 5122;

/// A parsed GLB: the JSON tree plus the BIN chunk bytes.
#[derive(Clone, Debug)]
pub struct Document {
    pub json: Json,
    pub bin: Vec<u8>,
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32, String> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| "truncated GLB".to_string())
}

impl Document {
    pub fn parse(bytes: &[u8]) -> Result<Document, String> {
        if u32_at(bytes, 0)? != MAGIC {
            return Err("not a GLB (bad magic)".into());
        }
        if u32_at(bytes, 4)? != 2 {
            return Err("unsupported GLB container version".into());
        }
        let total = u32_at(bytes, 8)? as usize;
        if total > bytes.len() {
            return Err("truncated GLB".into());
        }
        let mut at = 12;
        let mut json_text = None;
        let mut bin = Vec::new();
        while at + 8 <= total {
            let len = u32_at(bytes, at)? as usize;
            let kind = u32_at(bytes, at + 4)?;
            let start = at + 8;
            let end = start.checked_add(len).ok_or("bad chunk length")?;
            if end > total {
                return Err("truncated GLB chunk".into());
            }
            match kind {
                CHUNK_JSON if json_text.is_none() => {
                    let text = std::str::from_utf8(&bytes[start..end])
                        .map_err(|_| "GLB JSON chunk is not UTF-8".to_string())?;
                    json_text = Some(text.trim_end_matches([' ', '\0']).to_string());
                }
                CHUNK_BIN if bin.is_empty() => bin = bytes[start..end].to_vec(),
                _ => {}
            }
            at = end;
        }
        let text = json_text.ok_or("GLB has no JSON chunk")?;
        let json = json::parse(&text).map_err(|e| format!("GLB JSON: {}", e))?;
        if let Some(buffers) = json.get("buffers").and_then(Json::as_arr) {
            if buffers.len() > 1 || buffers.iter().any(|b| b.get("uri").is_some()) {
                return Err("only GLBs with one embedded buffer are supported".into());
            }
        }
        Ok(Document { json, bin })
    }

    pub fn read(path: &str) -> Result<Document, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
        Document::parse(&bytes).map_err(|e| format!("{}: {}", path, e))
    }

    /// Serialize (JSON padded with spaces, BIN with zeros, both to 4 bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut doc = self.json.clone();
        let mut bin = self.bin.clone();
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        if !bin.is_empty() {
            let buffer = Json::obj(vec![("byteLength", Json::num(bin.len() as f64))]);
            doc.set("buffers", Json::Arr(vec![buffer]));
        } else {
            doc.remove("buffers");
        }
        let mut text = json::emit(&doc)?.into_bytes();
        while text.len() % 4 != 0 {
            text.push(b' ');
        }
        let mut out = Vec::with_capacity(28 + text.len() + bin.len());
        let total = 12 + 8 + text.len() + if bin.is_empty() { 0 } else { 8 + bin.len() };
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
        out.extend_from_slice(&text);
        if !bin.is_empty() {
            out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
            out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
            out.extend_from_slice(&bin);
        }
        Ok(out)
    }

    pub fn write(&self, path: &str) -> Result<(), String> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {}", path, e))
    }

    /// Drop accessors nothing references (old animation samplers after a
    /// clip swap) and the buffer bytes no kept accessor or image uses, so
    /// rewritten files do not carry dead data. Files with extensions are
    /// left alone (an extension may reference accessors we do not know).
    pub fn compact(&mut self) {
        if self
            .json
            .get("extensionsUsed")
            .and_then(Json::as_arr)
            .is_some_and(|a| !a.is_empty())
        {
            return;
        }
        let n_acc = self.array("accessors").len();
        let mut used = vec![false; n_acc];
        let mark = |v: Option<&Json>, used: &mut Vec<bool>| {
            if let Some(i) = v.and_then(Json::as_usize) {
                if i < used.len() {
                    used[i] = true;
                }
            }
        };
        for mesh in self.array("meshes") {
            for prim in mesh.get("primitives").and_then(Json::as_arr).unwrap_or(&[]) {
                for (_, v) in prim.get("attributes").and_then(Json::as_obj).unwrap_or(&[]) {
                    mark(Some(v), &mut used);
                }
                mark(prim.get("indices"), &mut used);
                for t in prim.get("targets").and_then(Json::as_arr).unwrap_or(&[]) {
                    for (_, v) in t.as_obj().unwrap_or(&[]) {
                        mark(Some(v), &mut used);
                    }
                }
            }
        }
        for skin in self.array("skins") {
            mark(skin.get("inverseBindMatrices"), &mut used);
        }
        for anim in self.array("animations") {
            for sm in anim.get("samplers").and_then(Json::as_arr).unwrap_or(&[]) {
                mark(sm.get("input"), &mut used);
                mark(sm.get("output"), &mut used);
            }
        }
        let mut acc_map = vec![usize::MAX; n_acc];
        let mut k = 0;
        for i in 0..n_acc {
            if used[i] {
                acc_map[i] = k;
                k += 1;
            }
        }
        let remap_acc = |v: &mut Json, key: &str, map: &[usize]| {
            if let Some(i) = v.get(key).and_then(Json::as_usize) {
                if let Some(&m) = map.get(i) {
                    v.set(key, Json::num(m as f64));
                }
            }
        };
        if let Some(meshes) = self.json.get_mut("meshes").and_then(Json::as_arr_mut) {
            for mesh in meshes.iter_mut() {
                if let Some(prims) = mesh.get_mut("primitives").and_then(Json::as_arr_mut) {
                    for prim in prims.iter_mut() {
                        if let Some(attrs) = prim.get_mut("attributes") {
                            let keys: Vec<String> = attrs
                                .as_obj()
                                .unwrap_or(&[])
                                .iter()
                                .map(|(k, _)| k.clone())
                                .collect();
                            for key in keys {
                                remap_acc(attrs, &key, &acc_map);
                            }
                        }
                        remap_acc(prim, "indices", &acc_map);
                        if let Some(ts) = prim.get_mut("targets").and_then(Json::as_arr_mut) {
                            for t in ts.iter_mut() {
                                let keys: Vec<String> = t
                                    .as_obj()
                                    .unwrap_or(&[])
                                    .iter()
                                    .map(|(k, _)| k.clone())
                                    .collect();
                                for key in keys {
                                    remap_acc(t, &key, &acc_map);
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(skins) = self.json.get_mut("skins").and_then(Json::as_arr_mut) {
            for skin in skins.iter_mut() {
                remap_acc(skin, "inverseBindMatrices", &acc_map);
            }
        }
        if let Some(anims) = self.json.get_mut("animations").and_then(Json::as_arr_mut) {
            for anim in anims.iter_mut() {
                if let Some(sms) = anim.get_mut("samplers").and_then(Json::as_arr_mut) {
                    for sm in sms.iter_mut() {
                        remap_acc(sm, "input", &acc_map);
                        remap_acc(sm, "output", &acc_map);
                    }
                }
            }
        }
        let accessors: Vec<Json> = self
            .array("accessors")
            .iter()
            .enumerate()
            .filter(|(i, _)| used[*i])
            .map(|(_, a)| a.clone())
            .collect();
        // Buffer views still in use.
        let n_view = self.array("bufferViews").len();
        let mut vused = vec![false; n_view];
        let mut vmark = |v: Option<&Json>| {
            if let Some(i) = v.and_then(Json::as_usize) {
                if i < vused.len() {
                    vused[i] = true;
                }
            }
        };
        for a in &accessors {
            vmark(a.get("bufferView"));
            if let Some(sp) = a.get("sparse") {
                vmark(sp.get("indices").and_then(|x| x.get("bufferView")));
                vmark(sp.get("values").and_then(|x| x.get("bufferView")));
            }
        }
        for img in self.array("images") {
            vmark(img.get("bufferView"));
        }
        let mut view_map = vec![usize::MAX; n_view];
        let mut views = Vec::new();
        let mut bin = Vec::new();
        for (i, v) in self.array("bufferViews").iter().enumerate() {
            if !vused[i] {
                continue;
            }
            let off = v.get("byteOffset").and_then(Json::as_usize).unwrap_or(0);
            let len = v.get("byteLength").and_then(Json::as_usize).unwrap_or(0);
            let Some(bytes) = self.bin.get(off..off + len) else {
                return; // malformed: leave the file as it is
            };
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
            let mut nv = v.clone();
            nv.set("byteOffset", Json::num(bin.len() as f64));
            bin.extend_from_slice(bytes);
            view_map[i] = views.len();
            views.push(nv);
        }
        let mut accessors = accessors;
        for a in accessors.iter_mut() {
            remap_acc(a, "bufferView", &view_map);
            if let Some(sp) = a.get_mut("sparse") {
                if let Some(ix) = sp.get_mut("indices") {
                    remap_acc(ix, "bufferView", &view_map);
                }
                if let Some(vs) = sp.get_mut("values") {
                    remap_acc(vs, "bufferView", &view_map);
                }
            }
        }
        if let Some(imgs) = self.json.get_mut("images").and_then(Json::as_arr_mut) {
            for img in imgs.iter_mut() {
                remap_acc(img, "bufferView", &view_map);
            }
        }
        self.json.set("accessors", Json::Arr(accessors));
        self.json.set("bufferViews", Json::Arr(views));
        self.bin = bin;
    }

    pub fn array(&self, key: &str) -> &[Json] {
        self.json.get(key).and_then(Json::as_arr).unwrap_or(&[])
    }

    /// Mutable top-level array, created when absent.
    pub fn array_mut(&mut self, key: &str) -> &mut Vec<Json> {
        if self.json.get(key).and_then(Json::as_arr).is_none() {
            self.json.set(key, Json::Arr(Vec::new()));
        }
        self.json
            .get_mut(key)
            .and_then(Json::as_arr_mut)
            .expect("array just set")
    }

    /// Read accessor `index` as `count` rows of `width` f64 values
    /// (normalized integers per spec when the accessor says so).
    pub fn read_accessor(&self, index: usize) -> Result<Vec<Vec<f64>>, String> {
        let acc = self
            .array("accessors")
            .get(index)
            .ok_or_else(|| format!("accessor {} missing", index))?;
        if acc.get("sparse").is_some() {
            return Err(format!("accessor {} is sparse (unsupported)", index));
        }
        let comp = acc
            .get("componentType")
            .and_then(Json::as_usize)
            .ok_or("accessor without componentType")? as u64;
        let count = acc
            .get("count")
            .and_then(Json::as_usize)
            .ok_or("accessor without count")?;
        let kind = acc
            .get("type")
            .and_then(Json::as_str)
            .ok_or("accessor without type")?;
        let width = match kind {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            other => return Err(format!("unsupported accessor type {}", other)),
        };
        let normalized = acc
            .get("normalized")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let size = match comp {
            FLOAT | UNSIGNED_INT => 4,
            UNSIGNED_SHORT | SHORT => 2,
            UNSIGNED_BYTE | BYTE => 1,
            _ => return Err(format!("unsupported componentType {}", comp)),
        };
        let Some(view_index) = acc.get("bufferView").and_then(Json::as_usize) else {
            return Ok(vec![vec![0.0; width]; count]);
        };
        let view = self
            .array("bufferViews")
            .get(view_index)
            .ok_or("bufferView missing")?;
        let view_off = view.get("byteOffset").and_then(Json::as_usize).unwrap_or(0);
        let view_len = view
            .get("byteLength")
            .and_then(Json::as_usize)
            .ok_or("bufferView without byteLength")?;
        let acc_off = acc.get("byteOffset").and_then(Json::as_usize).unwrap_or(0);
        let packed = size * width;
        // Matrix columns of small components are 4-byte aligned; this
        // reader only meets MAT4 floats, which are already packed.
        let stride = view
            .get("byteStride")
            .and_then(Json::as_usize)
            .unwrap_or(packed);
        let start = view_off
            .checked_add(acc_off)
            .ok_or("accessor offset overflow")?;
        if count > 0 {
            let need = acc_off + stride * (count - 1) + packed;
            if need > view_len || view_off + view_len > self.bin.len() {
                return Err(format!("accessor {} exceeds its buffer", index));
            }
        }
        let mut rows = Vec::with_capacity(count);
        for i in 0..count {
            let base = start + i * stride;
            let mut row = Vec::with_capacity(width);
            for c in 0..width {
                let at = base + c * size;
                let b = &self.bin[at..at + size];
                let v = match comp {
                    FLOAT => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                    UNSIGNED_INT => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                    UNSIGNED_SHORT => {
                        let u = u16::from_le_bytes([b[0], b[1]]) as f64;
                        if normalized {
                            u / 65535.0
                        } else {
                            u
                        }
                    }
                    SHORT => {
                        let s = i16::from_le_bytes([b[0], b[1]]) as f64;
                        if normalized {
                            (s / 32767.0).max(-1.0)
                        } else {
                            s
                        }
                    }
                    UNSIGNED_BYTE => {
                        let u = b[0] as f64;
                        if normalized {
                            u / 255.0
                        } else {
                            u
                        }
                    }
                    _ => {
                        let s = b[0] as i8 as f64;
                        if normalized {
                            (s / 127.0).max(-1.0)
                        } else {
                            s
                        }
                    }
                };
                if !v.is_finite() {
                    return Err(format!("accessor {} holds a non-finite value", index));
                }
                row.push(v);
            }
            rows.push(row);
        }
        Ok(rows)
    }

    fn push_view(&mut self, bytes: &[u8], target: Option<u64>) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        let mut view = vec![
            ("buffer", Json::num(0.0)),
            ("byteOffset", Json::num(offset as f64)),
            ("byteLength", Json::num(bytes.len() as f64)),
        ];
        if let Some(t) = target {
            view.push(("target", Json::num(t as f64)));
        }
        let views = self.array_mut("bufferViews");
        views.push(Json::obj(view));
        views.len() - 1
    }

    /// Append a FLOAT accessor; `min_max` adds the bounds glTF requires
    /// for animation inputs and positions.
    pub fn push_float_accessor(&mut self, rows: &[Vec<f64>], kind: &str, min_max: bool) -> usize {
        let width = rows.first().map(|r| r.len()).unwrap_or(1);
        let mut bytes = Vec::with_capacity(rows.len() * width * 4);
        for row in rows {
            for v in row {
                bytes.extend_from_slice(&(*v as f32).to_le_bytes());
            }
        }
        let view = self.push_view(&bytes, None);
        let mut acc = vec![
            ("bufferView", Json::num(view as f64)),
            ("componentType", Json::num(FLOAT as f64)),
            ("count", Json::num(rows.len() as f64)),
            ("type", Json::str(kind)),
        ];
        if min_max && !rows.is_empty() {
            let mut lo = vec![f64::INFINITY; width];
            let mut hi = vec![f64::NEG_INFINITY; width];
            for row in rows {
                for (c, v) in row.iter().enumerate() {
                    let v = *v as f32 as f64;
                    lo[c] = lo[c].min(v);
                    hi[c] = hi[c].max(v);
                }
            }
            acc.push(("min", Json::Arr(lo.into_iter().map(Json::num).collect())));
            acc.push(("max", Json::Arr(hi.into_iter().map(Json::num).collect())));
        }
        let accessors = self.array_mut("accessors");
        accessors.push(Json::obj(acc));
        accessors.len() - 1
    }

    /// Append a VEC4 joint-index accessor (u8 when every index fits).
    pub fn push_joints_accessor(&mut self, rows: &[[u16; 4]]) -> usize {
        let wide = rows.iter().any(|r| r.iter().any(|&j| j > 255));
        let mut bytes = Vec::new();
        for row in rows {
            for &j in row {
                if wide {
                    bytes.extend_from_slice(&j.to_le_bytes());
                } else {
                    bytes.push(j as u8);
                }
            }
        }
        let view = self.push_view(&bytes, Some(34962));
        let comp = if wide { UNSIGNED_SHORT } else { UNSIGNED_BYTE };
        let accessors = self.array_mut("accessors");
        accessors.push(Json::obj(vec![
            ("bufferView", Json::num(view as f64)),
            ("componentType", Json::num(comp as f64)),
            ("count", Json::num(rows.len() as f64)),
            ("type", Json::str("VEC4")),
        ]));
        accessors.len() - 1
    }
}

/// Affine transform: general 3x3 linear part (row-major) + translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub m: [[f64; 3]; 3],
    pub t: Vec3,
}

impl Affine {
    pub const IDENTITY: Affine = Affine {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    };

    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Affine {
        let rm = r.normalized().to_mat3().rows;
        let sc = [s.x, s.y, s.z];
        let mut m = [[0.0; 3]; 3];
        for (row, rm_row) in m.iter_mut().zip(rm.iter()) {
            for c in 0..3 {
                row[c] = rm_row[c] * sc[c];
            }
        }
        Affine { m, t }
    }

    /// glTF column-major 4x4 `matrix` array.
    pub fn from_gltf_matrix(a: &[f64]) -> Affine {
        Affine {
            m: [[a[0], a[4], a[8]], [a[1], a[5], a[9]], [a[2], a[6], a[10]]],
            t: Vec3::new(a[12], a[13], a[14]),
        }
    }

    pub fn to_gltf_matrix(&self) -> Vec<f64> {
        let m = self.m;
        vec![
            m[0][0], m[1][0], m[2][0], 0.0, m[0][1], m[1][1], m[2][1], 0.0, m[0][2], m[1][2],
            m[2][2], 0.0, self.t.x, self.t.y, self.t.z, 1.0,
        ]
    }

    pub fn mul(&self, o: &Affine) -> Affine {
        let mut m = [[0.0; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| self.m[r][k] * o.m[k][c]).sum();
            }
        }
        Affine {
            m,
            t: self.apply(o.t),
        }
    }

    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.apply_linear(p) + self.t
    }

    pub fn apply_linear(&self, v: Vec3) -> Vec3 {
        let m = self.m;
        Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }

    pub fn det(&self) -> f64 {
        let m = self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    pub fn inverse(&self) -> Option<Affine> {
        let d = self.det();
        if d.abs() < 1e-300 || !d.is_finite() {
            return None;
        }
        let m = self.m;
        let inv = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d,
            ],
        ];
        let lin = Affine {
            m: inv,
            t: Vec3::ZERO,
        };
        let t = lin.apply_linear(self.t);
        Some(Affine {
            m: inv,
            t: Vec3::new(-t.x, -t.y, -t.z),
        })
    }

    /// Split into translation, rotation, scale (no shear assumed; a
    /// mirrored transform puts the flip into a negative x scale).
    pub fn decompose(&self) -> (Vec3, Quat, Vec3) {
        let col = |c: usize| Vec3::new(self.m[0][c], self.m[1][c], self.m[2][c]);
        let (x, y, z) = (col(0), col(1), col(2));
        let mut s = Vec3::new(x.length(), y.length(), z.length());
        if self.det() < 0.0 {
            s.x = -s.x;
        }
        let safe = |v: Vec3, l: f64| {
            if l.abs() > 1e-300 {
                v.scale(1.0 / l)
            } else {
                v
            }
        };
        let r = Mat3::from_cols(safe(x, s.x), safe(y, s.y), safe(z, s.z));
        (self.t, Quat::from_mat3(r), s)
    }

    /// Pure rotation part (columns normalized).
    pub fn rotation(&self) -> Quat {
        self.decompose().1
    }
}

/// Local transform of a node (TRS fields or `matrix`).
pub fn node_local(node: &Json) -> Affine {
    if let Some(m) = node.get("matrix").and_then(Json::as_arr) {
        let vals: Vec<f64> = m.iter().filter_map(Json::as_f64).collect();
        if vals.len() == 16 {
            return Affine::from_gltf_matrix(&vals);
        }
    }
    let (t, r, s) = node_trs(node);
    Affine::from_trs(t, r, s)
}

/// TRS of a node; a `matrix` node is decomposed.
pub fn node_trs(node: &Json) -> (Vec3, Quat, Vec3) {
    if node.get("matrix").is_some() {
        return node_local_matrix_only(node).decompose();
    }
    let v = |key: &str, n: usize, default: &[f64]| -> Vec<f64> {
        node.get(key)
            .and_then(Json::as_arr)
            .map(|a| a.iter().filter_map(Json::as_f64).collect::<Vec<_>>())
            .filter(|a| a.len() == n)
            .unwrap_or_else(|| default.to_vec())
    };
    let t = v("translation", 3, &[0.0, 0.0, 0.0]);
    let r = v("rotation", 4, &[0.0, 0.0, 0.0, 1.0]);
    let s = v("scale", 3, &[1.0, 1.0, 1.0]);
    (
        Vec3::new(t[0], t[1], t[2]),
        Quat::new(r[3], r[0], r[1], r[2]).normalized(),
        Vec3::new(s[0], s[1], s[2]),
    )
}

fn node_local_matrix_only(node: &Json) -> Affine {
    let vals: Vec<f64> = node
        .get("matrix")
        .and_then(Json::as_arr)
        .map(|m| m.iter().filter_map(Json::as_f64).collect())
        .unwrap_or_default();
    if vals.len() == 16 {
        Affine::from_gltf_matrix(&vals)
    } else {
        Affine::IDENTITY
    }
}

/// Write TRS onto a node (dropping any `matrix`), omitting defaults.
pub fn set_node_trs(node: &mut Json, t: Vec3, r: Quat, s: Vec3) {
    node.remove("matrix");
    let num = |v: f64| Json::num(clean(v));
    if t.length_sq() > 0.0 {
        node.set("translation", Json::Arr(vec![num(t.x), num(t.y), num(t.z)]));
    } else {
        node.remove("translation");
    }
    let r = canonical_quat(r);
    if r.approx_eq(Quat::IDENTITY, 0.0) {
        node.remove("rotation");
    } else {
        node.set(
            "rotation",
            Json::Arr(vec![num(r.x), num(r.y), num(r.z), num(r.w)]),
        );
    }
    if (s.x - 1.0).abs() > 0.0 || (s.y - 1.0).abs() > 0.0 || (s.z - 1.0).abs() > 0.0 {
        node.set("scale", Json::Arr(vec![num(s.x), num(s.y), num(s.z)]));
    } else {
        node.remove("scale");
    }
}

/// Positive-w representative of a rotation (q and -q are the same).
pub fn canonical_quat(q: Quat) -> Quat {
    let q = q.normalized();
    if q.w < 0.0 {
        q.neg()
    } else {
        q
    }
}

/// Snap float noise (|v| < 1e-12) to zero so outputs stay stable.
pub fn clean(v: f64) -> f64 {
    if v.abs() < 1e-12 {
        0.0
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_roundtrip_keeps_json_and_bin() {
        let mut doc = Document {
            json: json::parse(r#"{"asset":{"version":"2.0"},"nodes":[{"name":"a"}]}"#).unwrap(),
            bin: Vec::new(),
        };
        let acc =
            doc.push_float_accessor(&[vec![1.0, 2.0, 3.0], vec![-1.0, 0.5, 9.0]], "VEC3", true);
        let bytes = doc.to_bytes().unwrap();
        assert_eq!(bytes.len() % 4, 0);
        let back = Document::parse(&bytes).unwrap();
        assert_eq!(
            back.read_accessor(acc).unwrap(),
            vec![vec![1.0, 2.0, 3.0], vec![-1.0, 0.5, 9.0]]
        );
        assert_eq!(
            back.array("nodes")[0].get("name").unwrap().as_str(),
            Some("a")
        );
        let a = &back.array("accessors")[acc];
        assert_eq!(
            a.get("max").unwrap().as_arr().unwrap()[2].as_f64(),
            Some(9.0)
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(Document::parse(b"nope").is_err());
        assert!(Document::parse(&[0x67, 0x6C, 0x54, 0x46, 2, 0, 0, 0, 255, 0, 0, 0]).is_err());
    }

    #[test]
    fn affine_inverse_and_decompose() {
        let q = Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.9);
        let a = Affine::from_trs(Vec3::new(1.0, 2.0, 3.0), q, Vec3::new(0.01, 0.01, 0.01));
        let inv = a.inverse().unwrap();
        let p = Vec3::new(0.4, -2.0, 7.0);
        assert!(inv.apply(a.apply(p)).approx_eq(p, 1e-9));
        let (t, r, s) = a.decompose();
        assert!(t.approx_eq(Vec3::new(1.0, 2.0, 3.0), 1e-12));
        assert!(r.approx_eq(q, 1e-9));
        assert!((s.x - 0.01).abs() < 1e-12);
        let m = Affine::from_gltf_matrix(&a.to_gltf_matrix());
        assert!(m.apply(p).approx_eq(a.apply(p), 1e-12));
    }
}
