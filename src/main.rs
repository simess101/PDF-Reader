use anyhow::{anyhow, Context, Result};
use flate2::read::ZlibDecoder;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Read;

use eframe::{egui, NativeOptions};
use image::DynamicImage;

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

        if !data.starts_with(b"%PDF-") {
            return Err(anyhow!("Not a PDF: missing %PDF- header"));
        }

        let startxref_pos =
            find_startxref(&data).ok_or_else(|| anyhow!("Could not find startxref"))?;
        let xref_offset = parse_startxref(&data[startxref_pos..])?;

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
    let start = data.len().saturating_sub(1024);
    let slice = &data[start..];
    find_subsequence(slice, needle).map(|rel| start + rel)
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

    consume_whitespace(data, &mut pos);
    if !consume_keyword(data, &mut pos, b"xref") {
        return Err(anyhow!("Expected 'xref' at offset {}", xref_offset));
    }

    let mut xref: HashMap<u32, usize> = HashMap::new();

    loop {
        consume_whitespace(data, &mut pos);

        if match_keyword(data, pos, b"trailer") {
            break;
        }

        let first = parse_unsigned_int(data, &mut pos)?;
        consume_whitespace(data, &mut pos);
        let count = parse_unsigned_int(data, &mut pos)?;

        for i in 0..count {
            consume_whitespace(data, &mut pos);
            let offset = parse_unsigned_int(data, &mut pos)?;
            consume_whitespace(data, &mut pos);
            let _generation = parse_unsigned_int(data, &mut pos)?;
            consume_whitespace(data, &mut pos);
            let in_use_flag =
                data.get(pos)
                    .ok_or_else(|| anyhow!("Unexpected EOF in xref entry"))?;
            pos += 1;

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
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0C | 0x00) {
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

    if let PdfValue::Dictionary(dict) = &value {
        if consume_keyword(data, &mut pos, b"stream") {
            if let Some(b'\r') = data.get(pos) {
                pos += 1;
            }
            if let Some(b'\n') = data.get(pos) {
                pos += 1;
            }

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

            consume_whitespace(data, &mut pos);
            if !consume_keyword(data, &mut pos, b"endobj") {
                return Err(anyhow!("Missing endobj after stream object"));
            }

            Ok(PdfValue::Stream(stream))
        } else {
            consume_whitespace(data, &mut pos);
            if !consume_keyword(data, &mut pos, b"endobj") {
                return Err(anyhow!("Missing endobj after dictionary object"));
            }
            Ok(PdfValue::Dictionary(dict.clone()))
        }
    } else {
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
                if matches!(
                    c,
                    b' ' | b'\t' | b'\n' | b'\r' | b'<' | b'>' | b'[' | b']' | b'/'
                        | b'(' | b')'
                ) {
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

// -------------------- Stream decoding --------------------

fn decode_stream(stream: &PdfStream) -> Result<Vec<u8>> {
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
        Ok(stream.data.clone())
    }
}

// -------------------- Resources + ToUnicode --------------------

fn get_page_resources(pdf: &PdfFile, page: &PdfPage) -> Result<HashMap<String, PdfValue>> {
    if let Some(res_val) = page.dict.get("Resources") {
        let resolved = pdf.resolve_ref(res_val)?;
        match resolved {
            PdfValue::Dictionary(d) => Ok(d),
            _ => Err(anyhow!("Page /Resources is not a dictionary")),
        }
    } else {
        Ok(HashMap::new())
    }
}

// Map: font resource name (e.g., "F1") -> (code -> char)
type FontCMap = HashMap<u16, char>;
type FontMap = HashMap<String, FontCMap>;

fn parse_tounicode_cmap(data: &[u8]) -> Result<FontCMap> {
    let s = String::from_utf8_lossy(data);
    let mut map: FontCMap = HashMap::new();

    let mut tokens = s.split_whitespace();
    let mut mode: Option<&str> = None;

    while let Some(tok) = tokens.next() {
        match tok {
            "beginbfchar" => mode = Some("bfchar"),
            "endbfchar" => mode = None,
            "beginbfrange" => mode = Some("bfrange"),
            "endbfrange" => mode = None,
            _ => {
                if mode == Some("bfchar") {
                    if tok.starts_with('<') && tok.ends_with('>') {
                        let src_hex = tok.trim_matches(&['<', '>'][..]);
                        if let Some(dst_tok) = tokens.next() {
                            let dst_hex = dst_tok.trim_matches(&['<', '>'][..]);
                            if let (Ok(src), Ok(dst)) = (
                                u16::from_str_radix(src_hex, 16),
                                u16::from_str_radix(dst_hex, 16),
                            ) {
                                if let Some(ch) = std::char::from_u32(dst as u32) {
                                    map.insert(src, ch);
                                }
                            }
                        }
                    }
                } else if mode == Some("bfrange") {
                    if tok.starts_with('<') && tok.ends_with('>') {
                        let start_hex = tok.trim_matches(&['<', '>'][..]);
                        let end_tok = tokens
                            .next()
                            .ok_or_else(|| anyhow!("bfrange missing end code"))?;
                        let dst_tok = tokens
                            .next()
                            .ok_or_else(|| anyhow!("bfrange missing dst start"))?;

                        let end_hex = end_tok.trim_matches(&['<', '>'][..]);
                        let dst_hex = dst_tok.trim_matches(&['<', '>'][..]);

                        let start = u16::from_str_radix(start_hex, 16)?;
                        let end = u16::from_str_radix(end_hex, 16)?;
                        let mut dst = u16::from_str_radix(dst_hex, 16)?;

                        for code in start..=end {
                            if let Some(ch) = std::char::from_u32(dst as u32) {
                                map.insert(code, ch);
                            }
                            dst = dst.wrapping_add(1);
                        }
                    }
                }
            }
        }
    }

    Ok(map)
}

fn build_font_maps(pdf: &PdfFile, resources: &HashMap<String, PdfValue>) -> FontMap {
    let mut fonts: FontMap = HashMap::new();

    if let Some(PdfValue::Dictionary(font_dict)) = resources.get("Font") {
        for (name, font_val) in font_dict {
            let font_obj = match pdf.resolve_ref(font_val) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let font_dict2 = match font_obj {
                PdfValue::Dictionary(d) => d,
                _ => continue,
            };

            if let Some(tounicode_val) = font_dict2.get("ToUnicode") {
                let tu_obj = match pdf.resolve_ref(tounicode_val) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let PdfValue::Stream(s) = tu_obj {
                    if let Ok(bytes) = decode_stream(&s) {
                        if let Ok(cmap) = parse_tounicode_cmap(&bytes) {
                            fonts.insert(name.clone(), cmap);
                        }
                    }
                }
            }
        }
    }

    fonts
}

fn decode_string_with_font(raw: &[u8], current_font: &Option<String>, fonts: &FontMap) -> String {
    if let Some(font_name) = current_font {
        if let Some(cmap) = fonts.get(font_name) {
            if raw.len() % 2 == 0 && !cmap.is_empty() {
                let mut out = String::new();
                let mut i = 0;
                while i + 1 < raw.len() {
                    let code = ((raw[i] as u16) << 8) | (raw[i + 1] as u16);
                    if let Some(ch) = cmap.get(&code) {
                        out.push(*ch);
                    } else {
                        // fallback: just show something instead of blank
                        out.push('?');
                    }
                    i += 2;
                }
                return out;
            }
        }
    }

    String::from_utf8_lossy(raw).to_string()
}

// -------------------- Text extraction --------------------

fn extract_text_from_page(pdf: &PdfFile, page: &PdfPage) -> Result<String> {
    let streams = pdf.get_page_contents_streams(page)?;
    let resources = get_page_resources(pdf, page)?;
    let font_maps = build_font_maps(pdf, &resources);

    let mut full_text = String::new();
    for s in streams {
        let decoded = decode_stream(&s)?;
        extract_text_from_bytes(&decoded, &font_maps, &mut full_text)?;
    }
    Ok(full_text)
}

fn extract_text_from_bytes(
    data: &[u8],
    fonts: &FontMap,
    full_text: &mut String,
) -> Result<()> {
    let mut i = 0;
    let mut stack: Vec<Vec<u8>> = Vec::new(); // raw string operands
    let mut in_text_object = false;
    let mut last_name: Option<String> = None;
    let mut current_font: Option<String> = None;

    while i < data.len() {
        while i < data.len()
            && matches!(data[i], b' ' | b'\t' | b'\r' | b'\n')
        {
            i += 1;
        }
        if i >= data.len() {
            break;
        }

        let c = data[i];

        if c == b'(' {
            i += 1;
            let start = i;
            while i < data.len() && data[i] != b')' {
                i += 1;
            }
            let s_bytes = data[start..i].to_vec();
            if i < data.len() && data[i] == b')' {
                i += 1;
            }
            stack.push(s_bytes);
            continue;
        }

        // operator / name / hex token
        let start = i;
        while i < data.len()
            && !matches!(data[i], b' ' | b'\t' | b'\r' | b'\n')
        {
            i += 1;
        }
        let token = &data[start..i];
        let tok_str = String::from_utf8_lossy(token).to_string();

        // hex string like <002B0044...>
        if tok_str.starts_with('<') && tok_str.ends_with('>') && tok_str.len() > 2 {
            let inner = &tok_str[1..tok_str.len() - 1];
            let bytes = hex_to_bytes(inner.as_bytes())?;
            stack.push(bytes);
            continue;
        }

        // Names (/F1 etc.)
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
            "Tf" => {
                if let Some(name) = &last_name {
                    current_font = Some(name.clone());
                }
                // ignore font size operand; we don't use it here
            }
            "Tj" | "TJ" => {
                if in_text_object {
                    if let Some(raw) = stack.pop() {
                        let decoded = decode_string_with_font(&raw, &current_font, fonts);
                        full_text.push_str(&decoded);
                    }
                }
            }
            _ => {
                // ignore other operators (Td/Tm/etc.) for now
            }
        }
    }

    Ok(())
}

// -------------------- Image extraction (very simple) --------------------

fn extract_images_from_page(pdf: &PdfFile, page: &PdfPage) -> Result<Vec<DynamicImage>> {
    let resources = get_page_resources(pdf, page)?;
    let mut out = Vec::new();

    if let Some(PdfValue::Dictionary(xobjs)) = resources.get("XObject") {
        for (_name, val) in xobjs {
            let obj = pdf.resolve_ref(val)?;
            if let PdfValue::Stream(stream) = obj {
                if let Some(PdfValue::Name(subtype)) = stream.dict.get("Subtype") {
                    if subtype == "Image" {
                        let data = decode_stream(&stream)?;
                        if let Ok(img) = image::load_from_memory(&data) {
                            out.push(img);
                        }
                    }
                }
            }
        }
    }

    Ok(out)
}

// -------------------- GUI structures --------------------

struct PageData {
    text: String,
    images: Vec<DynamicImage>,
}

struct PdfApp {
    pages: Vec<PageData>,
    current_page: usize,
    // cache textures for the current page only
    tex_page_index: Option<usize>,
    tex_handles: Vec<egui::TextureHandle>,
}

impl eframe::App for PdfApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Ensure textures for current page
        if self.tex_page_index != Some(self.current_page) {
            // drop old textures
            self.tex_handles.clear();
            if let Some(page) = self.pages.get(self.current_page) {
                for (idx, img) in page.images.iter().enumerate() {
                    let rgb = img.to_rgba8();
                    let size = [img.width() as usize, img.height() as usize];
                    let color_image =
                        egui::ColorImage::from_rgba_unmultiplied(size, &rgb);
                    let handle = ctx.load_texture(
                        format!("page{}_img{}", self.current_page, idx),
                        color_image,
                        egui::TextureOptions::default(),
                    );
                    self.tex_handles.push(handle);
                }
            }
            self.tex_page_index = Some(self.current_page);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Prev").clicked() && self.current_page > 0 {
                    self.current_page -= 1;
                    self.tex_page_index = None;
                }
                if ui.button("Next").clicked()
                    && self.current_page + 1 < self.pages.len()
                {
                    self.current_page += 1;
                    self.tex_page_index = None;
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
                    // show page text
                    ui.monospace(&self.pages[self.current_page].text);
            
                    ui.separator();
            
                    // show images for this page
                    let page = &self.pages[self.current_page];
                    if !page.images.is_empty() {
                        ui.heading("Images on this page:");
                        for tex in &self.tex_handles {
                            let size = egui::Vec2::new(tex.size()[0] as f32, tex.size()[1] as f32);
                            ui.image((tex.id(), size));
                            // Removed redundant or undefined variable usage
                            ui.add_space(12.0);
                        }
                    }
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

    let pages_vec = pdf.load_all_pages()?;
    println!("Found {} page(s)\n", pages_vec.len());

    if use_gui {
        let mut page_data = Vec::new();
        for page in &pages_vec {
            let text = match extract_text_from_page(&pdf, page) {
                Ok(t) => t,
                Err(e) => format!("Error extracting text: {e}"),
            };
            let images = extract_images_from_page(&pdf, page).unwrap_or_default();
            page_data.push(PageData { text, images });
        }

        let app = PdfApp {
            pages: page_data,
            current_page: 0,
            tex_page_index: None,
            tex_handles: Vec::new(),
        };

        let native_options = NativeOptions::default();
        eframe::run_native(
            &format!("PDF Reader - {}", path),
            native_options,
            Box::new(|_cc| Box::new(app)),
        )
        .map_err(|e| anyhow!("GUI error: {e}"))?;
    } else {
        for (i, page) in pages_vec.iter().enumerate() {
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
