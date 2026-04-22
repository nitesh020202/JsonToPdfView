import os
import re
import json
import base64
import tempfile
import io
import fitz  # PyMuPDF
import PIL.Image
from google import genai
from google.genai import types as genai_types
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def _make_client():
    key = os.environ.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=key) if key else None

gemini_client = _make_client()
GEMINI_MODEL  = "gemini-2.0-flash"

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


# ── Dynamic template (no hardcoded keys) ─────────────────────────────────────

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "question_template.json")

DEFAULT_TEMPLATE = {
    "questionid": 1,
    "question": "",
    "option1": "", "option2": "", "option3": "", "option4": "",
    "Answer": "", "Explanation": "",
    "course": "CBSE", "subjectname": "Math", "chapter": "",
    "practice": "", "subtopic": "", "medium": "English",
    "difficulty": "", "question_type": "Subjective",
    "previous_year": "", "marks": "", "class": ""
}

_template_cache: dict | None = None

def load_template() -> dict:
    global _template_cache
    if _template_cache is not None:
        return _template_cache
    if os.path.exists(TEMPLATE_PATH):
        try:
            with open(TEMPLATE_PATH) as f:
                _template_cache = json.load(f)
                return _template_cache
        except Exception:
            pass
    _template_cache = dict(DEFAULT_TEMPLATE)
    return _template_cache

def save_template(template: dict):
    global _template_cache
    _template_cache = template
    with open(TEMPLATE_PATH, "w") as f:
        json.dump(template, f, indent=2)

def get_option_keys(template: dict) -> list[str]:
    return [k for k in template if k.lower().startswith("option")]

def get_answer_key(template: dict) -> str:
    for k in template:
        if k.lower() == "answer":
            return k
    return "Answer"

def get_explanation_key(template: dict) -> str:
    for k in template:
        if k.lower() in ("explanation", "solution", "explain"):
            return k
    return "Explanation"

def get_qtype_key(template: dict) -> str:
    for k in template:
        if "type" in k.lower() and "question" in k.lower():
            return k
    return "question_type"


# ── LaTeX / HTML normalisation ────────────────────────────────────────────────

def html_to_latex(text: str) -> str:
    if not text:
        return text
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<b>(.*?)</b>", r"\\textbf{\1}", text, flags=re.DOTALL)
    text = re.sub(r"<i>(.*?)</i>", r"\\textit{\1}", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = text.replace("&nbsp;", " ")
    return text.strip()


def extract_math_expressions(text: str) -> list[dict]:
    exprs = []
    for m in re.finditer(r"\\\[(.*?)\\\]", text, re.DOTALL):
        latex = m.group(1).strip()
        exprs.append({
            "type": "display",
            "latex": latex,
            "latex_base64": base64.b64encode(latex.encode()).decode(),
            "raw": m.group(0),
        })
    for m in re.finditer(r"\\\((.*?)\\\)", text, re.DOTALL):
        latex = m.group(1).strip()
        exprs.append({
            "type": "inline",
            "latex": latex,
            "latex_base64": base64.b64encode(latex.encode()).decode(),
            "raw": m.group(0),
        })
    return exprs


def enrich_field(value: str) -> dict:
    if not value:
        return {"text": value, "latex_cleaned": value, "math_expressions": []}
    cleaned = html_to_latex(value)
    return {
        "text": cleaned,
        "latex_cleaned": cleaned,
        "math_expressions": extract_math_expressions(cleaned),
    }


# ── Question-type normalisation ───────────────────────────────────────────────

def normalize_by_question_type(q: dict, template: dict) -> dict:
    opt_keys  = get_option_keys(template)
    ans_key   = get_answer_key(template)
    qt_key    = get_qtype_key(template)
    qt = (q.get(qt_key) or "Subjective").strip().lower()

    if "mcq" in qt or "multiple" in qt or "choice" in qt:
        q[qt_key] = "MCQ"
        ans = str(q.get(ans_key, "")).strip()
        valid = [str(i) for i in range(1, len(opt_keys) + 1)]
        if ans not in valid:
            for i, ok in enumerate(opt_keys, 1):
                opt_val = str(q.get(ok, "")).strip()
                if opt_val and opt_val.lower() == ans.lower():
                    ans = str(i)
                    break
        q[ans_key] = ans

    elif "true" in qt or "false" in qt:
        q[qt_key] = "True/False"
        for ok in opt_keys:
            q[ok] = ""
        ans = str(q.get(ans_key, "")).strip().lower()
        if ans in ("true", "1", "yes", "correct"):
            q[ans_key] = "1"
        elif ans in ("false", "0", "no", "incorrect", "wrong"):
            q[ans_key] = "0"
        else:
            q[ans_key] = ans

    elif "numeric" in qt:
        q[qt_key] = "Numeric"
        for ok in opt_keys:
            q[ok] = ""
        ans = str(q.get(ans_key, "")).strip()
        try:
            q[ans_key] = str(int(float(ans)))
        except (ValueError, TypeError):
            q[ans_key] = ans

    elif "fill" in qt or "blank" in qt:
        q[qt_key] = "Fill in the Blanks"
        for ok in opt_keys:
            q[ok] = ""

    else:
        q[qt_key] = q.get(qt_key) or "Subjective"
        for ok in opt_keys:
            q[ok] = ""
        q[ans_key] = ""

    return q


# ── Main conversion ───────────────────────────────────────────────────────────

def _img_tag(data_uri: str) -> str:
    return f'<img src="{data_uri}" style="max-width:280px;display:block;margin:6px 0"/>'

def convert_question_json(q: dict, page_images: list[dict] | None = None) -> dict:
    template = load_template()
    expl_key = get_explanation_key(template)

    q.pop("category", None)
    q = normalize_by_question_type(q, template)

    # Resolve diagram placements — embed images into the right field
    if page_images:
        placements = q.pop("diagram_placements", [])
        indices_fallback = q.pop("diagram_image_indices", [])

        # First pass: replace <<DIAGRAM_N>> markers wherever they appear in any field
        for idx, img_info in enumerate(page_images):
            marker = f"<<DIAGRAM_{idx}>>"
            tag    = _img_tag(img_info["data_uri"])
            replaced = False
            for field in list(template.keys()):
                field_val = q.get(field) or ""
                if marker in field_val:
                    q[field] = field_val.replace(marker, tag)
                    replaced = True
            # If marker not found, use diagram_placements field or fallback to expl_key
            if not replaced and placements:
                for p in placements:
                    if p.get("image_index", 0) == idx:
                        field = p.get("field", expl_key)
                        q[field] = (q.get(field) or "") + "\n" + tag
                        replaced = True
                        break
            if not replaced:
                q[expl_key] = (q.get(expl_key) or "") + "\n" + tag

        # Clean up any remaining markers
        for field in template:
            val = q.get(field) or ""
            if "<<DIAGRAM_" in val:
                import re as _re
                q[field] = _re.sub(r"<<DIAGRAM_\d+>>", "", val)

        _ = indices_fallback
    else:
        q.pop("diagram_placements", None)
        q.pop("diagram_image_indices", None)

    # Output only the keys defined in template
    result = {}
    for key in template:
        val = q.get(key)
        if val is None:
            default = template[key]
            result[key] = "" if isinstance(default, str) else default
        elif isinstance(val, str):
            result[key] = val.replace("\n", "<br>")
        else:
            result[key] = val

    return result


# ── PDF image extraction ──────────────────────────────────────────────────────

def pdf_page_to_base64(pdf_path: str, page_num: int) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode()
    doc.close()
    return data


def extract_page_embedded_images(pdf_path: str, page_num: int) -> list[dict]:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            img_data = doc.extract_image(xref)
            raw = img_data["image"]
            ext = img_data.get("ext", "png")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
            b64 = base64.b64encode(raw).decode()
            images.append({"mime_type": mime, "image_base64": b64,
                            "data_uri": f"data:{mime};base64,{b64}"})
        except Exception:
            pass
    doc.close()
    return images


# ── PDF extraction prompt ─────────────────────────────────────────────────────

def build_extraction_prompt(num_diagrams: int) -> str:
    template = load_template()
    opt_keys  = get_option_keys(template)
    ans_key   = get_answer_key(template)
    expl_key  = get_explanation_key(template)
    qt_key    = get_qtype_key(template)

    # Build schema string from template keys
    schema_lines = []
    for k, v in template.items():
        if k == "questionid":
            schema_lines.append(f'  "{k}": <integer>')
        elif k == "question":
            schema_lines.append(f'  "{k}": "<text + LaTeX: \\\\( \\\\) inline, \\\\[ \\\\] display>"')
        elif k == expl_key:
            schema_lines.append(f'  "{k}": "<text + LaTeX>"')
        elif k == qt_key:
            schema_lines.append(f'  "{k}": "Subjective"')
        elif isinstance(v, str):
            schema_lines.append(f'  "{k}": "{v}"')
        else:
            schema_lines.append(f'  "{k}": {json.dumps(v)}')
    schema_lines.append('  "diagram_placements": []')
    schema_str = "{\n" + ",\n".join(schema_lines) + "\n}"

    # MCQ option keys for type rules
    opt_list = ", ".join(opt_keys) if opt_keys else "option1, option2, option3, option4"
    opt_empty = ", ".join(f'"{k}"=""' for k in opt_keys)

    if num_diagrams > 0:
        diagram_note = (
            f'DIAGRAM RULES ({num_diagrams} diagram(s) on this page, indexed 0–{num_diagrams-1}):\n'
            f'- Insert marker <<DIAGRAM_0>>, <<DIAGRAM_1>>, etc. at the EXACT spot where each diagram appears.\n'
            f'- CRITICAL: Write ALL text BEFORE the marker AND ALL text AFTER the marker — never stop mid-solution.\n'
            f'- Example: "{expl_key}": "step1<br>step2<br><<DIAGRAM_0>><br>step3<br>step4<br>step5"\n'
            f'- Also fill "diagram_placements": [{{"image_index": 0, "field": "{expl_key}"}}]'
        )
    else:
        diagram_note = '- "diagram_placements": []'

    return f"""You are a math content extraction expert. Extract ALL questions from this exam page.

Return ONLY a valid JSON array. Each element follows this schema:
{schema_str}

QUESTION TYPE RULES:
- MCQ: {opt_list} MUST all have values. {ans_key} = "1"/"2"/"3"/"4".
- True/False: {opt_empty}. {ans_key} = "1" (True) or "0" (False).
- Numeric: {opt_empty}. {ans_key} = integer string e.g. "42".
- Fill in the Blanks: {opt_empty}. {ans_key} = the answer.
- Subjective: {opt_empty}. {ans_key} = "". {expl_key} = COMPLETE step-by-step solution.

LATEX RULES:
- Inline: \\\\( ... \\\\)  Display: \\\\[ ... \\\\]
- Fractions: \\\\frac{{a}}{{b}}  Roots: \\\\sqrt{{x}}  Powers: x^{{2}}
- CRITICAL: every backslash must be doubled in JSON strings
- Use <br> for line breaks — never use \\n or real newlines inside strings

{diagram_note}
- If no questions found, return []
"""


# ── JSON repair ──────────────────────────────────────────────────────────────

def fix_json_backslashes(s: str) -> str:
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_string:
            if c == '\\' and i + 1 < len(s):
                nc = s[i + 1]
                if nc == '"':
                    result.append('\\'); result.append('"'); i += 2; continue
                elif nc == '\\':
                    result.append('\\'); result.append('\\'); i += 2; continue
                else:
                    result.append('\\'); result.append('\\')
            elif c == '"':
                in_string = False
                result.append(c)
            else:
                result.append(c)
        else:
            if c == '"':
                in_string = True
            result.append(c)
        i += 1
    return ''.join(result)


def parse_gemini_json(raw: str) -> list:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    candidate = match.group(0)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(fix_json_backslashes(candidate))
    except json.JSONDecodeError as e:
        raise e


# ── SSE streaming extraction ──────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _process_page(pdf_path: str, page_num: int, num_pages: int) -> dict:
    import time
    page_images = extract_page_embedded_images(pdf_path, page_num)
    page_b64    = pdf_page_to_base64(pdf_path, page_num)
    prompt      = build_extraction_prompt(len(page_images))

    last_error = ""
    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    genai_types.Part.from_bytes(
                        data=base64.b64decode(page_b64),
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ],
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=8192,
                    temperature=0.1,
                ),
            )
            questions = parse_gemini_json(response.text.strip())
            for q in questions:
                q["_page_images"] = page_images
            return {"page_num": page_num, "questions": questions,
                    "diagrams": len(page_images), "error": ""}
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error (attempt {attempt+1}): {e}"
        except Exception as e:
            last_error = str(e)
            if "429" in last_error or "quota" in last_error.lower():
                time.sleep(3)
    return {"page_num": page_num, "questions": [], "diagrams": len(page_images), "error": last_error}


def stream_pdf_extraction(pdf_path: str):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    yield sse("start", {"total_pages": num_pages})

    max_workers = min(num_pages, 4)
    results_by_page = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_page, pdf_path, p, num_pages): p
            for p in range(num_pages)
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            pn = result["page_num"]
            results_by_page[pn] = result
            completed += 1
            if result["error"] and not result["questions"]:
                yield sse("page_error", {"page": pn + 1, "error": result["error"]})
            else:
                yield sse("page_done", {
                    "page": pn + 1,
                    "total": num_pages,
                    "questions_found": len(result["questions"]),
                    "total_so_far": sum(len(results_by_page[i]["questions"]) for i in results_by_page),
                    "diagrams_found": result["diagrams"],
                })

    all_questions = []
    for pn in sorted(results_by_page):
        all_questions.extend(results_by_page[pn]["questions"])

    yield sse("converting", {"total_questions": len(all_questions)})

    enriched = []
    seen = set()
    for i, q in enumerate(all_questions, 1):
        imgs = q.pop("_page_images", [])
        converted = convert_question_json(q, imgs)
        # Ensure unique sequential questionids
        qid = converted.get("questionid")
        if not qid or qid in seen:
            converted["questionid"] = i
        seen.add(converted["questionid"])
        enriched.append(converted)

    yield sse("done", {"total_questions": len(enriched), "result": enriched})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/convert-json", methods=["POST"])
def api_convert_json():
    data = request.get_json(force=True)
    if isinstance(data, list):
        result = [convert_question_json(dict(q)) for q in data]
    else:
        result = convert_question_json(dict(data))
    return jsonify(result)


@app.route("/api/convert-pdf-stream", methods=["POST"])
def api_convert_pdf_stream():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files accepted"}), 400

    tmp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(tmp_dir, "upload_stream.pdf")
    f.save(pdf_path)

    def generate():
        try:
            yield from stream_pdf_extraction(pdf_path)
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/set-key", methods=["POST"])
def api_set_key():
    data = request.get_json(force=True)
    key = data.get("api_key", "").strip()
    if not key:
        return jsonify({"error": "API key cannot be empty"}), 400
    os.environ["GEMINI_API_KEY"] = key
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    with open(env_path, "w") as f:
        f.write(f"GEMINI_API_KEY={key}\n")
    global gemini_client
    gemini_client = genai.Client(api_key=key)
    return jsonify({"ok": True})


@app.route("/api/get-template", methods=["GET"])
def api_get_template():
    return jsonify(load_template())


@app.route("/api/set-template", methods=["POST"])
def api_set_template():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Template must be a JSON object"}), 400
    if not data:
        return jsonify({"error": "Template cannot be empty"}), 400
    save_template(data)
    return jsonify({"ok": True, "keys": list(data.keys())})


@app.route("/api/decode-latex", methods=["POST"])
def api_decode_latex():
    data = request.get_json(force=True)
    encoded = data.get("latex_base64", "")
    try:
        decoded = base64.b64decode(encoded.encode()).decode()
        return jsonify({"latex": decoded})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
