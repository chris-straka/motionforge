//! Minimal JSON parser and deterministic emitter (no dependencies).
//!
//! The parser builds a [`Json`] tree; typed extraction lives with each
//! format (see `clip.rs`). The emitter helpers write canonical JSON:
//! fixed key order (chosen by the caller), floats shortest-round-trip
//! with `-0.0` normalized to `0.0`, strings escaped. Non-finite floats
//! are a hard error on both sides.

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(pairs) => Some(pairs),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Json> {
        match self {
            Json::Obj(pairs) => pairs.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Set `key` on an object (replacing in place, else appending).
    /// No-op on non-objects.
    pub fn set(&mut self, key: &str, value: Json) {
        if let Json::Obj(pairs) = self {
            match pairs.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value,
                None => pairs.push((key.to_string(), value)),
            }
        }
    }

    /// Remove `key` from an object, returning its value.
    pub fn remove(&mut self, key: &str) -> Option<Json> {
        match self {
            Json::Obj(pairs) => {
                let i = pairs.iter().position(|(k, _)| k == key)?;
                Some(pairs.remove(i).1)
            }
            _ => None,
        }
    }

    pub fn as_arr_mut(&mut self) -> Option<&mut Vec<Json>> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(v) if *v >= 0.0 && v.fract() == 0.0 && *v <= 9.0e15 => Some(*v as usize),
            _ => None,
        }
    }

    pub fn num(v: f64) -> Json {
        Json::Num(v)
    }

    pub fn str(s: &str) -> Json {
        Json::Str(s.to_string())
    }

    pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "bool",
            Json::Num(_) => "number",
            Json::Str(_) => "string",
            Json::Arr(_) => "array",
            Json::Obj(_) => "object",
        }
    }
}

/// Maximum array/object nesting. Real files nest ~5 deep; the cap turns
/// hostile input (`[[[[...`) into an error instead of a stack overflow.
const MAX_DEPTH: usize = 512;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Parser<'a> {
        Parser {
            bytes: text.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
            depth: 0,
        }
    }

    fn err(&self, msg: &str) -> String {
        format!("line {} col {}: {}", self.line, self.col, msg)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.bytes.get(self.pos).copied()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.bump();
        }
    }

    fn expect(&mut self, b: u8, what: &str) -> Result<(), String> {
        self.skip_ws();
        match self.bump() {
            Some(x) if x == b => Ok(()),
            _ => Err(self.err(&format!("expected {}", what))),
        }
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{' | b'[') => {
                if self.depth >= MAX_DEPTH {
                    return Err(self.err("nesting too deep"));
                }
                self.depth += 1;
                let v = if self.peek() == Some(b'{') {
                    self.parse_object()
                } else {
                    self.parse_array()
                };
                self.depth -= 1;
                v
            }
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b't') => self.parse_literal("true", Json::Bool(true)),
            Some(b'f') => self.parse_literal("false", Json::Bool(false)),
            Some(b'n') => self.parse_literal("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(_) => Err(self.err("unexpected character")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn parse_literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        for expected in word.bytes() {
            match self.bump() {
                Some(x) if x == expected => {}
                _ => return Err(self.err(&format!("expected `{}`", word))),
            }
        }
        Ok(value)
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.expect(b'{', "'{'")?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.bump();
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key"));
            }
            let key = self.parse_string()?;
            self.expect(b':', "':'")?;
            let value = self.parse_value()?;
            pairs.push((key, value));
            self.skip_ws();
            match self.bump() {
                Some(b',') => {}
                Some(b'}') => return Ok(Json::Obj(pairs)),
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.expect(b'[', "'['")?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.bump() {
                Some(b',') => {}
                Some(b']') => return Ok(Json::Arr(items)),
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.bump();
        let mut out = String::new();
        loop {
            match self.bump() {
                Some(b'"') => return Ok(out),
                Some(b'\\') => match self.bump() {
                    Some(b'"') => out.push('"'),
                    Some(b'\\') => out.push('\\'),
                    Some(b'/') => out.push('/'),
                    Some(b'b') => out.push('\u{0008}'),
                    Some(b'f') => out.push('\u{000C}'),
                    Some(b'n') => out.push('\n'),
                    Some(b'r') => out.push('\r'),
                    Some(b't') => out.push('\t'),
                    Some(b'u') => {
                        let mut code: u32 = 0;
                        for _ in 0..4 {
                            let h = self.bump().ok_or_else(|| self.err("bad \\u escape"))?;
                            let d = (h as char)
                                .to_digit(16)
                                .ok_or_else(|| self.err("bad \\u escape"))?;
                            code = code * 16 + d;
                        }
                        // Surrogate pair.
                        if (0xD800..0xDC00).contains(&code) {
                            if self.peek() == Some(b'\\') {
                                self.bump();
                                if self.peek() == Some(b'u') {
                                    self.bump();
                                    let mut lo: u32 = 0;
                                    for _ in 0..4 {
                                        let h = self
                                            .bump()
                                            .ok_or_else(|| self.err("bad \\u escape"))?;
                                        let d = (h as char)
                                            .to_digit(16)
                                            .ok_or_else(|| self.err("bad \\u escape"))?;
                                        lo = lo * 16 + d;
                                    }
                                    if (0xDC00..0xE000).contains(&lo) {
                                        code = 0x10000 + ((code - 0xD800) << 10) + (lo - 0xDC00);
                                    } else {
                                        return Err(self.err("bad surrogate pair"));
                                    }
                                } else {
                                    return Err(self.err("bad surrogate pair"));
                                }
                            }
                        }
                        out.push(char::from_u32(code).ok_or_else(|| self.err("bad codepoint"))?);
                    }
                    _ => return Err(self.err("bad escape")),
                },
                Some(b) if b < 0x20 => return Err(self.err("control character in string")),
                Some(b) => {
                    // UTF-8 multi-byte passthrough: decode just this char
                    // (width from the lead byte; validating the whole
                    // remaining input here made long strings quadratic).
                    let start = self.pos - 1;
                    let width = match b {
                        0x00..=0x7f => 1,
                        0xc0..=0xdf => 2,
                        0xe0..=0xef => 3,
                        _ => 4,
                    };
                    let end = (start + width).min(self.bytes.len());
                    let s = std::str::from_utf8(&self.bytes[start..end])
                        .map_err(|_| self.err("invalid utf-8"))?;
                    let ch = s.chars().next().ok_or_else(|| self.err("invalid utf-8"))?;
                    for _ in 1..ch.len_utf8() {
                        self.bump();
                    }
                    out.push(ch);
                }
                None => return Err(self.err("unterminated string")),
            }
        }
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        match self.peek() {
            Some(b'0') => {
                self.bump();
            }
            Some(c) if c.is_ascii_digit() => {
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.bump();
                }
            }
            _ => return Err(self.err("bad number")),
        }
        if self.peek() == Some(b'.') {
            self.bump();
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.err("bad number"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.bump();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump();
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.err("bad number"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("bad number"))?;
        let v: f64 = text.parse().map_err(|_| self.err("number out of range"))?;
        if !v.is_finite() {
            return Err(self.err("non-finite number"));
        }
        Ok(Json::Num(v))
    }
}

/// Parse a whole JSON document (trailing garbage is an error).
pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser::new(text);
    let v = p.parse_value()?;
    p.skip_ws();
    if p.peek().is_some() {
        return Err(p.err("trailing characters"));
    }
    Ok(v)
}

/// Append a JSON string literal with escaping.
pub fn push_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Append a finite float, shortest round-trip, `-0.0` -> `0.0`.
pub fn push_num(out: &mut String, v: f64) -> Result<(), String> {
    if !v.is_finite() {
        return Err("non-finite float in output".to_string());
    }
    let v = if v == 0.0 { 0.0 } else { v };
    out.push_str(&format!("{}", v));
    Ok(())
}

/// Emit a [`Json`] tree compactly (object key order preserved).
pub fn emit(value: &Json) -> Result<String, String> {
    let mut out = String::new();
    emit_into(&mut out, value, None, 0)?;
    Ok(out)
}

/// Emit a [`Json`] tree with two-space indentation and a final newline.
pub fn emit_pretty(value: &Json) -> Result<String, String> {
    let mut out = String::new();
    emit_into(&mut out, value, Some(2), 0)?;
    out.push('\n');
    Ok(out)
}

fn newline(out: &mut String, indent: Option<usize>, depth: usize) {
    if let Some(n) = indent {
        out.push('\n');
        for _ in 0..n * depth {
            out.push(' ');
        }
    }
}

fn emit_into(
    out: &mut String,
    value: &Json,
    indent: Option<usize>,
    depth: usize,
) -> Result<(), String> {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(v) => push_num(out, *v)?,
        Json::Str(s) => push_str(out, s),
        Json::Arr(items) => {
            out.push('[');
            let scalar = items
                .iter()
                .all(|v| !matches!(v, Json::Arr(_) | Json::Obj(_)));
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_some() && scalar {
                        out.push(' ');
                    }
                }
                if !scalar {
                    newline(out, indent, depth + 1);
                }
                emit_into(out, item, indent, depth + 1)?;
            }
            if !scalar && !items.is_empty() {
                newline(out, indent, depth);
            }
            out.push(']');
        }
        Json::Obj(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, indent, depth + 1);
                push_str(out, k);
                out.push(':');
                if indent.is_some() {
                    out.push(' ');
                }
                emit_into(out, v, indent, depth + 1)?;
            }
            if !pairs.is_empty() {
                newline(out, indent, depth);
            }
            out.push('}');
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_nesting_is_an_error_not_a_stack_overflow() {
        let ok = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse(&ok).is_ok());
        let deep = format!("{}{}", "[".repeat(200_000), "]".repeat(200_000));
        assert!(parse(&deep).unwrap_err().contains("nesting too deep"));
        let deep_obj = format!("{}1{}", "{\"a\":".repeat(100_000), "}".repeat(100_000));
        assert!(parse(&deep_obj).unwrap_err().contains("nesting too deep"));
    }

    #[test]
    fn long_non_ascii_strings_parse_in_linear_time() {
        // Mixed 2-, 3- and 4-byte chars; quadratic before (~minutes).
        let body = "é€😀a".repeat(100_000);
        let v = parse(&format!("{{\"s\": \"{}\"}}", body)).unwrap();
        assert_eq!(v.get("s").unwrap().as_str().unwrap(), body);
    }

    #[test]
    fn roundtrip_nested() {
        let doc = r#"{"a": [1, -2.5, 1e3, true, false, null], "b": {"c": "x\ny\"q\""}}"#;
        let v = parse(doc).unwrap();
        assert_eq!(v.get("a").unwrap().as_arr().unwrap().len(), 6);
        assert_eq!(
            v.get("b").unwrap().get("c").unwrap().as_str().unwrap(),
            "x\ny\"q\""
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse(r#"{"a": 1} trailing"#).is_err());
        assert!(parse(r#"{"a": }"#).is_err());
        assert!(parse(r#"{"a": NaN}"#).is_err());
        assert!(parse("\"unterminated").is_err());
        assert!(parse("[1,]".to_string().leak()).is_err());
    }

    #[test]
    fn emit_roundtrips_and_edits() {
        let doc = r#"{"a": [1, -2.5, 1e3, true, false, null], "b": {"c": "x\ny\"q\""}}"#;
        let mut v = parse(doc).unwrap();
        assert_eq!(parse(&emit(&v).unwrap()).unwrap(), v);
        assert_eq!(parse(&emit_pretty(&v).unwrap()).unwrap(), v);
        v.set("d", Json::num(3.0));
        v.get_mut("b").unwrap().set("c", Json::str("z"));
        assert_eq!(v.remove("a").unwrap().as_arr().unwrap().len(), 6);
        assert_eq!(emit(&v).unwrap(), r#"{"b":{"c":"z"},"d":3}"#);
    }

    #[test]
    fn num_format() {
        let mut s = String::new();
        push_num(&mut s, 0.1 + 0.2).unwrap();
        assert_eq!(s, "0.30000000000000004");
        s.clear();
        push_num(&mut s, -0.0).unwrap();
        assert_eq!(s, "0");
        assert!(push_num(&mut s, f64::INFINITY).is_err());
    }
}
