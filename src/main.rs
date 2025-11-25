use anyhow::{anyhow, Context, Result};
use flate2::read::ZlibDecoder;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Read;
use eframe::{egui, NativeOptions};

// -------------------- Basic PDF value types --------------------

#[derive(Debug, Clone)]
pub enum PdfValue {
    Null,
    Bool(bool),
    Integer(i64),
    Real(f64),
    Name(String),
    String(Vec<u8>),
    Array(Vec<PdfValue>),
    Dictionary(HashMap<String, PdfValue>),
    Stream(PdfStream),
    Ref(PdfRef),
}

#[derive(Debug, Clone)]
pub struct PdfStream {
    pub dict: HashMap<String, PdfValue>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PdfRef {
    pub obj_num: u32,
    pub gen: u16,
}

#[derive(Debug, Clone)]
pub struct PdfPage {
    pub dict: HashMap<String, PdfValue>,
}

#[derive(Debug)]
pub struct PdfFile {
    data: Vec<u8>,
    xref: HashMap<u32, usize>,
    trailer: HashMap<String, PdfValue>,
}

// -------------------- PdfFile impl --------------------

impl PdfFile {
    pub fn open(path: &str) -> Result<Self> {
        let data = fs::read(path).with_context(|| format!("reading {path}"))?;

        // Check header
        if !data.starts_with(b"%PDF-") {
            return Err(anyhow!("Not a PDF: missing %PDF- header"));
        }

        // Find startxref
        let startxref_pos =
            find_startxref(&data).ok_or_else(|| anyhow!("Could not find startxref"))?;
        let xref_offset = parse_startxref(&data[startxref_pos..])?;

        // Parse xref table and trailer
        let (xref, trailer) = parse_xref_and_trailer(&data, xref_offset)?;

        Ok(PdfFile { data, xref, trailer })
    }

    pub fn get_root(&self) -> Result<PdfRef> {
        match self.trailer.get("Root") {
            Some(PdfValue::Ref(r)) => Ok(*r),
            _ => Err(anyhow!("Trailer missing /Root or not a ref")),
        }
    }

    pub fn get_object(&self, r: PdfRef) -> Result<PdfValue> {
        let offset = self
            .xref
            .get(&r.obj_num)
            .ok_or_else(|| anyhow!("No xref entry for object {}", r.obj_num))?;

        parse_indirect_object(&self.data, *offset, r.obj_num, r.gen)
    }

    fn resolve_ref(&self, v: &PdfValue) -> Result<PdfValue> {
        if let PdfValue::Ref(r) = v {
            self.get_object(*r)
        } else {
            Ok(v.clone())
        }
    }

    pub fn load_all_pages(&self) -> Result<Vec<PdfPage>> {
        let root_ref = self.get_root()?;
        let root_obj = self.get_object(root_ref)?;
        let root_dict = match root_obj {
            PdfValue::Dictionary(d) => d,
            _ => return Err(anyhow!("Root is not a dictionary")),
        };

        let pages_ref = match root_dict.get("Pages") {
            Some(PdfValue::Ref(r)) => *r,
            _ => return Err(anyhow!("Catalog missing /Pages ref")),
        };

        let mut pages = Vec::new();
        self.walk_pages_node(pages_ref, &mut pages)?;
        Ok(pages)
    }

    fn walk_pages_node(&self, node_ref: PdfRef, out: &mut Vec<PdfPage>) -> Result<()> {
        let node_val = self.get_object(node_ref)?;
        let node_dict = match node_val {
            PdfValue::Dictionary(d) => d,
            _ => return Err(anyhow!("Pages node is not a dict")),
        };

        let type_name = match node_dict.get("Type") {
            Some(PdfValue::Name(n)) => n.as_str(),
            _ => "",
        };

        if type_name == "Pages" {
            let kids = match node_dict.get("Kids") {
                Some(PdfValue::Array(a)) => a,
                _ => return Err(anyhow!("Pages node missing /Kids array")),
            };

            for kid in kids {
                if let PdfValue::Ref(r) = kid {
                    self.walk_pages_node(*r, out)?;
                }
            }
        } else if type_name == "Page" {
            out.push(PdfPage { dict: node_dict });
        } else {
            eprintln!("Warning unknown page tree node type: {:?}", type_name);
        }
        Ok(())
    }

    pub fn get_page_contents_streams(&self, page: &PdfPage) -> Result<Vec<PdfStream>> {
        let contents = page
            .dict
            .get("Contents")
            .ok_or_else(|| anyhow!("Page missing /Contents"))?;

        let mut streams = Vec::new();

        match contents {
            PdfValue::Ref(r) => {
                let obj = self.get_object(*r)?;
                match obj {
                    PdfValue::Stream(s) => streams.push(s),
                    _ => return Err(anyhow!("Contents ref did not resolve to stream")),
                }
            }
            PdfValue::Array(arr) => {
                for item in arr {
                    if let PdfValue::Ref(r) = item {
                        let obj = self.get_object(*r)?;
                        if let PdfValue::Stream(s) = obj {
                            streams.push(s);
                        }
                    }
                }
            }
            PdfValue::Stream(s) => {
                streams.push(s.clone());
            }
            _ => return Err(anyhow!("Unsupported /Contents type")),
        }

        Ok(streams)
    }
}

// -------------------- Step 1: find startxref --------------------

fn find_startxref(data: &[u8]) -> Option<usize> {
    let needle = b"startxref";
    // look in the last 1 KB of the file
    let start = data.len().saturating_sub(1024);
    let slice = &data[start..];
    if let Some(rel_pos) = find_subsequence(slice, needle) {
        Some(start + rel_pos)
    } else {
        None
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    for i in 0..=(haystack.len() - needle.len()) {
        if &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}

fn parse_startxref(chunk: &[u8]) -> Result<usize> {
    // chunk starts at "startxref"
    let s = std::str::from_utf8(chunk)?;
    let mut lines = s.lines();

    let first = lines
        .next()
        .ok_or_else(|| anyhow!("Empty chunk in parse_startxref"))?;
    if !first.trim_start().starts_with("startxref") {
        return Err(anyhow!("parse_startxref called on wrong location"));
    }

    let offset_line = lines
        .next()
        .ok_or_else(|| anyhow!("Missing offset after startxref"))?;
    let offset: usize = offset_line.trim().parse()?;
    Ok(offset)
}

// -------------------- Step 2: parse xref + trailer --------------------

fn parse_xref_and_trailer(
    data: &[u8],
    xref_offset: usize,
) -> Result<(HashMap<u32, usize>, HashMap<String, PdfValue>)> {
    let mut pos = xref_offset;

    // Expect "xref"
    consume_whitespace(data, &mut pos);
    if !consume_keyword(data, &mut pos, b"xref") {
        return Err(anyhow!(
            "Expected 'xref' at offset {}",
            xref_offset
        ));
    }

    let mut xref: HashMap<u32, usize> = HashMap::new();

    loop {
        consume_whitespace(data, &mut pos);

        // Next should be "trailer" or a subsection header: "<first> <count>"
        if match_keyword(data, pos, b"trailer") {
            break;
        }

        // parse "first_object count"
        let first = parse_unsigned_int(data, &mut pos)?;
        consume_whitespace(data, &mut pos);
        let count = parse_unsigned_int(data, &mut pos)?;

        // each entry is "offset generation n/f"
        for i in 0..count {
            consume_whitespace(data, &mut pos);
            let offset = parse_unsigned_int(data, &mut pos)?;
            consume_whitespace(data, &mut pos);
            let _generation = parse_unsigned_int(data, &mut pos)?;
            consume_whitespace(data, &mut pos);
            let in_use_flag =
                data.get(pos)
                    .ok_or_else(|| anyhow!("Unexpected EOF in xref entry"))?;
            pos += 1; // skip 'n' or 'f'

            // line end
            if let Some(b'\r') = data.get(pos) {
                pos += 1;
            }
            if let Some(b'\n') = data.get(pos) {
                pos += 1;
            }

            if *in_use_flag == b'n' {
                let obj_num = first + i;
                xref.insert(obj_num as u32, offset);
            }
        }
    }

    // Now parse "trailer" dictionary
    consume_whitespace(data, &mut pos);
    if !consume_keyword(data, &mut pos, b"trailer") {
        return Err(anyhow!("Expected 'trailer' after xref"));
    }
    consume_whitespace(data, &mut pos);

    let (trailer_dict, _) = parse_dictionary(data, pos)?;
    Ok((xref, trailer_dict))
}

// -------------------- Low-level helpers --------------------

fn consume_whitespace(data: &[u8], pos: &mut usize) {
    while *pos < data.len() {
        let b = data[*pos];
        if b == b' '
            || b == b'\t'
            || b == b'\n'
            || b == b'\r'
            || b == b'\x0C'
            || b == b'\x00'
        {
            *pos += 1;
        } else {
            break;
        }
    }
}

fn match_keyword(data: &[u8], pos: usize, kw: &[u8]) -> bool {
    data.len() >= pos + kw.len() && &data[pos..pos + kw.len()] == kw
}

fn consume_keyword(data: &[u8], pos: &mut usize, kw: &[u8]) -> bool {
    if match_keyword(data, *pos, kw) {
        *pos += kw.len();
        true
    } else {
        false
    }
}

fn parse_unsigned_int(data: &[u8], pos: &mut usize) -> Result<usize> {
    consume_whitespace(data, pos);
    let start = *pos;
    while *pos < data.len() && (data[*pos] as char).is_ascii_digit() {
        *pos += 1;
    }
    if *pos == start {
        return Err(anyhow!("Expected integer at position {start}"));
    }
    let s = std::str::from_utf8(&data[start..*pos])?;
    Ok(s.parse()?)
}

// -------------------- Step 3: parse an indirect object --------------------

fn parse_indirect_object(
    data: &[u8],
    offset: usize,
    obj_num: u32,
    gen: u16,
) -> Result<PdfValue> {
    let mut pos = offset;
    consume_whitespace(data, &mut pos);

    // Expect "<obj_num> <gen> obj"
    let parsed_obj = parse_unsigned_int(data, &mut pos)? as u32;
    consume_whitespace(data, &mut pos);
    let parsed_gen = parse_unsigned_int(data, &mut pos)? as u16;
    consume_whitespace(data, &mut pos);
    if !consume_keyword(data, &mut pos, b"obj") {
        return Err(anyhow!("Expected 'obj' keyword at offset {offset}"));
    }
    if parsed_obj != obj_num || parsed_gen != gen {
        eprintln!(
            "Warning: xref says {} {} but object header says {} {}",
            obj_num, gen, parsed_obj, parsed_gen
        );
    }

    consume_whitespace(data, &mut pos);
    let (value, new_pos) = parse_value(data, pos)?;
    pos = new_pos;
    consume_whitespace(data, &mut pos);

    // Handle stream
    if let PdfValue::Dictionary(dict) = &value {
        if consume_keyword(data, &mut pos, b"stream") {
            // optional EOL after 'stream'
            if let Some(b'\r') = data.get(pos) {
                pos += 1;
            }
            if let Some(b'\n') = data.get(pos) {
                pos += 1;
            }

            // length comes from /Length
            let length = match dict.get("Length") {
                Some(PdfValue::Integer(n)) => *n as usize,
                Some(PdfValue::Ref(r)) => {
                    return Err(anyhow!(
                        "Stream /Length is an indirect ref {:?}, not handled yet",
                        r
                    ));
                }
                _ => return Err(anyhow!("Stream missing /Length or not integer")),
            };

            if pos + length > data.len() {
                return Err(anyhow!("Stream length goes past end of file"));
            }

            let raw_stream = data[pos..pos + length].to_vec();
            pos += length;

            // after stream data: "endstream"
            let mut end_pos = pos;
            consume_whitespace(data, &mut end_pos);
            if !consume_keyword(data, &mut end_pos, b"endstream") {
                return Err(anyhow!("Missing endstream after stream data"));
            }
            pos = end_pos;

            let stream = PdfStream {
                dict: dict.clone(),
                data: raw_stream,
            };

            // After endstream there should be "endobj"
            consume_whitespace(data, &mut pos);
            if !consume_keyword(data, &mut pos, b"endobj") {
                return Err(anyhow!("Missing endobj after stream object"));
            }

            Ok(PdfValue::Stream(stream))
        } else {
            // Dictionary object without stream
            consume_whitespace(data, &mut pos);
            if !consume_keyword(data, &mut pos, b"endobj") {
                return Err(anyhow!("Missing endobj after dictionary object"));
            }
            Ok(PdfValue::Dictionary(dict.clone()))
        }
    } else {
        // Non-dict object
        consume_whitespace(data, &mut pos);
        if !consume_keyword(data, &mut pos, b"endobj") {
            return Err(anyhow!("Missing endobj after object"));
        }
        Ok(value)
    }
}

// -------------------- Generic value parser --------------------

fn parse_value(data: &[u8], mut pos: usize) -> Result<(PdfValue, usize)> {
    consume_whitespace(data, &mut pos);
    if pos >= data.len() {
        return Err(anyhow!("Unexpected EOF in parse_value"));
    }

    let b = data[pos];

    let v = match b {
        b'n' if match_keyword(data, pos, b"null") => {
            pos += 4;
            PdfValue::Null
        }
        b't' if match_keyword(data, pos, b"true") => {
            pos += 4;
            PdfValue::Bool(true)
        }
        b'f' if match_keyword(data, pos, b"false") => {
            pos += 5;
            PdfValue::Bool(false)
        }
        b'/' => {
            pos += 1;
            let start = pos;
            while pos < data.len() {
                let c = data[pos];
                if c == b' '
                    || c == b'\t'
                    || c == b'\n'
                    || c == b'\r'
                    || c == b'<'
                    || c == b'>'
                    || c == b'['
                    || c == b']'
                    || c == b'/'
                    || c == b'('
                    || c == b')'
                {
                    break;
                }
                pos += 1;
            }
            let name = String::from_utf8_lossy(&data[start..pos]).to_string();
            PdfValue::Name(name)
        }
        b'(' => {
            pos += 1;
            let start = pos;
            while pos < data.len() && data[pos] != b')' {
                pos += 1;
            }
            let s = data[start..pos].to_vec();
            if pos < data.len() && data[pos] == b')' {
                pos += 1;
            }
            PdfValue::String(s)
        }
        b'[' => {
            pos += 1;
            let mut items = Vec::new();
            loop {
                consume_whitespace(data, &mut pos);
                if pos >= data.len() {
                    return Err(anyhow!("EOF in array"));
                }
                if data[pos] == b']' {
                    pos += 1;
                    break;
                }
                let (val, new_pos) = parse_value(data, pos)?;
                pos = new_pos;
                items.push(val);
            }
            PdfValue::Array(items)
        }
        b'<' => {
            // dictionary or hex string
            if pos + 1 < data.len() && data[pos + 1] == b'<' {
                let (dict, new_pos) = parse_dictionary(data, pos)?;
                pos = new_pos;
                PdfValue::Dictionary(dict)
            } else {
                pos += 1;
                let start = pos;
                while pos < data.len() && data[pos] != b'>' {
                    pos += 1;
                }
                let hex_bytes = &data[start..pos];
                if pos < data.len() && data[pos] == b'>' {
                    pos += 1;
                }
                let s = hex_to_bytes(hex_bytes)?;
                PdfValue::String(s)
            }
        }
        b'-' | b'+' | b'0'..=b'9' => {
            let (num_str, new_pos) = parse_number_token(data, pos)?;
            pos = new_pos;
            if num_str.contains('.') {
                let val: f64 = num_str.parse()?;
                PdfValue::Real(val)
            } else {
                let val: i64 = num_str.parse()?;
                PdfValue::Integer(val)
            }
        }
        _ => {
            return Err(anyhow!(
                "Unexpected byte {} ('{}') at position {}",
                b,
                b as char,
                pos
            ));
        }
    };

    // Detect "<int> <int> R" pattern = indirect reference
    if let PdfValue::Integer(n1) = v {
        let mut look = pos;
        consume_whitespace(data, &mut look);

        if let Ok(n2) = parse_signed_int_peek(data, &mut look) {
            consume_whitespace(data, &mut look);
            if look < data.len() && data[look] == b'R' {
                let gen = n2 as u16;
                let obj = n1 as u32;
                let new_pos = look + 1;
                return Ok((PdfValue::Ref(PdfRef { obj_num: obj, gen }), new_pos));
            }
        }
        Ok((PdfValue::Integer(n1), pos))
    } else {
        Ok((v, pos))
    }
}

fn parse_dictionary(
    data: &[u8],
    mut pos: usize,
) -> Result<(HashMap<String, PdfValue>, usize)> {
    if !match_keyword(data, pos, b"<<") {
        return Err(anyhow!("Expected '<<' for dictionary"));
    }
    pos += 2;
    let mut dict = HashMap::new();

    loop {
        consume_whitespace(data, &mut pos);
        if match_keyword(data, pos, b">>") {
            pos += 2;
            break;
        }
        if pos >= data.len() {
            return Err(anyhow!("EOF in dictionary"));
        }
        let (key_v, new_pos) = parse_value(data, pos)?;
        pos = new_pos;
        let key = match key_v {
            PdfValue::Name(n) => n,
            _ => return Err(anyhow!("Dictionary key must be a name")),
        };
        let (val, new_pos) = parse_value(data, pos)?;
        pos = new_pos;
        dict.insert(key, val);
    }
    Ok((dict, pos))
}

fn parse_number_token(data: &[u8], mut pos: usize) -> Result<(String, usize)> {
    let start = pos;
    if data[pos] == b'+' || data[pos] == b'-' {
        pos += 1;
    }
    while pos < data.len() {
        let c = data[pos] as char;
        if c.is_ascii_digit() || c == '.' {
            pos += 1;
        } else {
            break;
        }
    }
    let s = std::str::from_utf8(&data[start..pos])?.to_string();
    Ok((s, pos))
}

fn parse_signed_int_peek(data: &[u8], pos: &mut usize) -> Result<i64> {
    let start = *pos;
    if *pos < data.len() && (data[*pos] == b'+' || data[*pos] == b'-') {
        *pos += 1;
    }
    while *pos < data.len() && (data[*pos] as char).is_ascii_digit() {
        *pos += 1;
    }
    if *pos == start {
        return Err(anyhow!("Expected signed int"));
    }
    let s = std::str::from_utf8(&data[start..*pos])?;
    Ok(s.parse()?)
}

fn hex_to_bytes(hex: &[u8]) -> Result<Vec<u8>> {
    let mut cleaned = Vec::new();
    for &b in hex {
        if (b as char).is_ascii_hexdigit() {
            cleaned.push(b);
        }
    }
    if cleaned.len() % 2 != 0 {
        cleaned.push(b'0');
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < cleaned.len() {
        let byte = u8::from_str_radix(&String::from_utf8_lossy(&cleaned[i..i + 2]), 16)?;
        out.push(byte);
        i += 2;
    }
    Ok(out)
}

// -------------------- Caesar decode helper (for your test PDF) --------------------

fn caesar_decode_alpha(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if ('A'..='Z').contains(&ch) {
            let offset = (ch as u8 - b'A' + 26 - 3) % 26;
            out.push((b'A' + offset) as char);
        } else {
            out.push(ch);
        }
    }
    out
}

// -------------------- Stream decoding --------------------

fn decode_stream(stream: &PdfStream) -> Result<Vec<u8>> {
    // Only handle no filter or simple /FlateDecode for now
    let filter = stream.dict.get("Filter");
    let has_flate = match filter {
        None => false,
        Some(PdfValue::Name(name)) => name == "FlateDecode",
        Some(PdfValue::Array(arr)) => {
            arr.iter().any(|v| matches!(v, PdfValue::Name(n) if n == "FlateDecode"))
        }
        _ => false,
    };

    if has_flate {
        let mut decoder = ZlibDecoder::new(&stream.data[..]);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out)?;
        Ok(out)
    } else {
        // no filter we know about, just return raw
        Ok(stream.data.clone())
    }
}

// -------------------- Resources helper --------------------

fn get_page_resources(pdf: &PdfFile, page: &PdfPage) -> Result<HashMap<String, PdfValue>> {
    if let Some(res_val) = page.dict.get("Resources") {
        let resolved = pdf.resolve_ref(res_val)?;
        match resolved {
            PdfValue::Dictionary(d) => Ok(d),
            _ => Err(anyhow!("Page /Resources is not a dictionary")),
        }
    } else {
        // Some PDFs inherit resources from parents; we ignore that for now
        Ok(HashMap::new())
    }
}

// -------------------- Text extraction --------------------

fn extract_text_from_page(pdf: &PdfFile, page: &PdfPage) -> Result<String> {
    let streams = pdf.get_page_contents_streams(page)?;
    let resources = get_page_resources(pdf, page)?;
    let mut full_text = String::new();

    for s in streams {
        let decoded = decode_stream(&s)?;
        extract_text_from_bytes(pdf, &decoded, &resources, &mut full_text)?;
    }

    Ok(full_text)
}

fn extract_text_from_bytes(
    pdf: &PdfFile,
    data: &[u8],
    resources: &HashMap<String, PdfValue>,
    full_text: &mut String,
) -> Result<()> {
    let mut i = 0;
    let mut stack: Vec<String> = Vec::new(); // for string operands
    let mut in_text_object = false;
    let mut last_name: Option<String> = None; // e.g., /X0 before Do

    while i < data.len() {
        // Skip whitespace
        while i < data.len()
            && (data[i] == b' ' || data[i] == b'\t' || data[i] == b'\r' || data[i] == b'\n')
        {
            i += 1;
        }
        if i >= data.len() {
            break;
        }

        let c = data[i];

        // literal string: (Hello)
        if c == b'(' {
            i += 1;
            let start = i;
            // naive: read until ')'
            while i < data.len() && data[i] != b')' {
                i += 1;
            }
            let s_bytes = &data[start..i];
            if i < data.len() && data[i] == b')' {
                i += 1;
            }
            let s = String::from_utf8_lossy(s_bytes).to_string();
            stack.push(s);
            continue;
        }

        // operator or name/hex token: read until whitespace
        let start = i;
        while i < data.len()
            && data[i] != b' '
            && data[i] != b'\t'
            && data[i] != b'\r'
            && data[i] != b'\n'
        {
            i += 1;
        }
        let token = &data[start..i];
        let tok_str = String::from_utf8_lossy(token).to_string();

        // Hex string in content stream: <4869...>
        if tok_str.starts_with('<') && tok_str.ends_with('>') && tok_str.len() > 2 {
            let inner = &tok_str[1..tok_str.len() - 1];
            let bytes = hex_to_bytes(inner.as_bytes())?;
            let s = String::from_utf8_lossy(&bytes).to_string();
            stack.push(s);
            continue;
        }

        // Names like /F1, /X0, etc.
        if tok_str.starts_with('/') {
            last_name = Some(tok_str[1..].to_string());
            continue;
        }

        match tok_str.as_str() {
            "BT" => {
                in_text_object = true;
            }
            "ET" => {
                in_text_object = false;
                full_text.push('\n');
            }
            "Tj" => {
                if in_text_object {
                    if let Some(text) = stack.pop() {
                        let decoded = caesar_decode_alpha(&text);
                        full_text.push_str(&decoded);
                    }
                }
            }
            "TJ" => {
                if in_text_object {
                    if let Some(text) = stack.pop() {
                        let decoded = caesar_decode_alpha(&text);
                        full_text.push_str(&decoded);
                    }
                }
            }
            "Do" => {
                // 'Do' draws an XObject. Often forms (sub-streams) contain text.
                if let Some(name) = &last_name {
                    if let Some(PdfValue::Dictionary(xobjs)) = resources.get("XObject") {
                        if let Some(PdfValue::Ref(r)) = xobjs.get(name) {
                            let obj = pdf.get_object(*r)?;
                            if let PdfValue::Stream(x_stream) = obj {
                                let decoded = decode_stream(&x_stream)?;
                                // recurse into the XObject content
                                extract_text_from_bytes(pdf, &decoded, resources, full_text)?;
                            }
                        }
                    }
                }
            }
            _ => {
                // ignore for now (Td/Tm/etc)
            }
        }
    }

    Ok(())
}

// -------------------- Simple GUI app --------------------

struct PdfApp {
    pages: Vec<String>,
    current_page: usize,
}

impl eframe::App for PdfApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Prev").clicked() && self.current_page > 0 {
                    self.current_page -= 1;
                }
                if ui.button("Next").clicked()
                    && self.current_page + 1 < self.pages.len()
                {
                    self.current_page += 1;
                }
                ui.label(format!(
                    "Page {}/{}",
                    self.current_page + 1,
                    self.pages.len().max(1)
                ));
            });

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                if self.pages.is_empty() {
                    ui.label("(no pages)");
                } else {
                    ui.monospace(&self.pages[self.current_page]);
                }
            });
        });
    }
}

// -------------------- Main --------------------

fn main() -> Result<()> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(anyhow!("Usage: pdf-reader [--gui] <file.pdf>"));
    }

    let use_gui = if args[0] == "--gui" {
        args.remove(0);
        true
    } else {
        false
    };

    if args.is_empty() {
        return Err(anyhow!("Usage: pdf-reader [--gui] <file.pdf>"));
    }
    let path = args.remove(0);

    let pdf = PdfFile::open(&path)?;
    println!("PDF opened successfully.\n");

    println!("Trailer keys:");
    for key in pdf.trailer.keys() {
        println!("  /{}", key);
    }

    let root = pdf.get_root()?;
    println!("\nRoot object: {:?}\n", root);

    let root_obj = pdf.get_object(root)?;
    println!("Root object value: {:#?}\n", root_obj);

    let pages = pdf.load_all_pages()?;
    println!("Found {} page(s)\n", pages.len());

    if use_gui {
        let mut page_texts = Vec::new();
        for page in &pages {
            match extract_text_from_page(&pdf, page) {
                Ok(text) => page_texts.push(text),
                Err(e) => page_texts.push(format!("Error extracting text: {e}")),
            }
        }

        let app = PdfApp {
            pages: page_texts,
            current_page: 0,
        };

        let native_options = NativeOptions::default();
        eframe::run_native(
            &format!("PDF Reader - {}", path),
            native_options,
            Box::new(|_cc| Box::new(app)),
        )
        .map_err(|e| anyhow!("GUI error: {e}"))?;
    } else {
        for (i, page) in pages.iter().enumerate() {
            println!("===== Page {} =====", i + 1);
            match extract_text_from_page(&pdf, page) {
                Ok(text) => {
                    if text.trim().is_empty() {
                        println!("(no text found)");
                    } else {
                        println!("{text}");
                    }
                }
                Err(e) => {
                    println!("Error extracting text: {e}");
                }
            }
            println!();
        }
    }

    Ok(())
}
