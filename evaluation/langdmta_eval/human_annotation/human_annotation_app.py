"""Local web-based annotation tool for LangDMTA evaluation.

Setup:
    # Option A: Use the existing langlab conda environment (recommended)
    conda activate langlab

    # Option B: Create a fresh environment
    conda create -n annotation python=3.11 -y
    conda activate annotation
    pip install flask pandas

Usage:
    python human_annotation_app.py
    python human_annotation_app.py --csv annotation_sample_annotator_sheet.csv
    python human_annotation_app.py --csv my_sheet.csv --port 5050

Opens a browser at http://localhost:5000 (or chosen port).

Requirements:
    - Python >= 3.9
    - flask
    - pandas
    - markdown
"""

import argparse
import csv
import html as html_mod
import json
import webbrowser
from pathlib import Path
from threading import Timer

import markdown
import pandas as pd
from flask import Flask, Response, redirect, render_template_string, request, url_for

# ---------------------------------------------------------------------------
# Metric definitions (same categorical labels as the LLM judge)
# ---------------------------------------------------------------------------
METRICS = [
    {
        "key": "human_completeness",
        "label": "Completeness",
        "options": ["INCOMPLETE", "PARTIALLY COMPLETE", "COMPLETE"],
        "help": (
            "How complete is the output in covering all required information for the task? This "
            "metric evaluates recall: of the information that should be in the output (see EVALUATION CONTEXT), how much "
            "is actually present? COMPLETE means all key information, results, and details required "
            "by the task are present. PARTIALLY COMPLETE means the output addresses the main point "
            "but is missing some required information or key details. INCOMPLETE means the output "
            "fails to provide most of the required information. Do NOT penalize for off-topic "
            "content here — that is assessed by Relevancy."
        ),
    },
    {
        "key": "human_relevancy",
        "label": "Relevancy",
        "options": ["NOT RELEVANT", "PARTIALLY RELEVANT", "RELEVANT"],
        "help": (
            "How relevant is the output content to the task and evaluation context? This metric "
            "evaluates precision: of the information present in the output, is it pertinent to "
            "the user's request? RELEVANT means all content directly addresses the task without "
            "irrelevant or off-topic information. PARTIALLY RELEVANT means the output addresses "
            "the task but includes noticeable irrelevant or off-topic content. NOT RELEVANT means "
            "the output is largely off-topic or addresses a different question. Do NOT penalize "
            "for missing information here — that is assessed by Completeness."
        ),
    },
    {
        "key": "human_structural_clarity",
        "label": "Structural Clarity",
        "options": ["UNCLEAR", "PARTIALLY CLEAR", "CLEAR"],
        "help": (
            "How well-structured and easy to parse is the output, independent of whether the "
            "content is correct, relevant, or complete? This metric evaluates presentation only: "
            "formatting, organization, logical flow, and readability. CLEAR means well-organized "
            "with appropriate structure and logical flow. PARTIALLY CLEAR means some structural "
            "issues but still broadly understandable. UNCLEAR means poorly organized or formatted "
            "in a way that significantly hinders comprehension."
        ),
    },
    {
        "key": "human_scope_adherence",
        "label": "Scope Adherence",
        "options": ["BELOW TARGET", "ON TARGET", "ABOVE TARGET"],
        "help": (
            "Does the agent operate within its intended scope and use its capabilities "
            "appropriately? ABOVE TARGET means the agent goes beyond the task scope, makes "
            "unsolicited tool calls, or provides unrequested actions. BELOW TARGET means the "
            "agent fails to use available tools, hallucinates instead of calling tools, refuses "
            "to engage with a reasonable request, or asks for unnecessary clarification. "
            "ON TARGET means the agent uses its tools correctly, addresses exactly what was "
            "asked, and neither overreaches nor under-delivers. This metric differs from Completeness "
            "and Relevancy in that it assesses whether the agent is appropriately ambitious or cautious "
            "given the task, rather than just whether the output contains required information."
        ),
    },
]

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annotation Tool - Case {{ idx + 1 }}/{{ total }}</title>
<style>
  :root { --bg: #f8f9fa; --card: #fff; --blue: #d0e1fd; --green: #c8e6c9;
          --orange: #fde0b0; --grey: #f1f3f4; --accent: #1a73e8; --red: #d93025; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: #202124; line-height: 1.6; }
  .container { max-width: 960px; margin: 0 auto; padding: 16px; }

  /* -- progress -- */
  .progress-bar { background: #b8bcc0; border-radius: 8px; height: 10px; margin-bottom: 8px; }
  .progress-fill { background: #34a853; height: 100%; border-radius: 8px;
                   transition: width .3s; }
  .progress-text { font-size: 13px; color: #5f6368; margin-bottom: 16px; }

  /* -- nav -- */
  .nav { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .nav a, .nav button { display: inline-flex; align-items: center; gap: 4px;
    padding: 8px 16px; border-radius: 6px; font-size: 14px; font-weight: 500;
    text-decoration: none; border: 1px solid #b8bcc0; background: var(--card);
    color: #202124; cursor: pointer; transition: background .15s; }
  .nav a:hover, .nav button:hover { background: #f1f3f4; }
  .nav .btn-save { background: var(--accent); color: #fff; border-color: var(--accent); }
  .nav .btn-save:hover { background: #1557b0; }
  .nav select { padding: 8px 12px; border-radius: 6px; border: 1px solid #b8bcc0;
    font-size: 14px; background: var(--card); }
  .badge { padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge-done { background: #e6f4ea; color: #137333; }
  .badge-pending { background: #fef3e0; color: #b06000; }

  /* -- cards -- */
  .card { background: var(--card); border-radius: 8px; padding: 16px; margin-bottom: 12px;
          border: 1px solid #c8c8c8; }
  .card h4 { font-size: 13px; color: #5f6368; text-transform: uppercase; letter-spacing: .5px;
             margin-bottom: 8px; }
  .card-blue { background: var(--blue); border-color: #a8c7fa; }
  .card-green { background: var(--green); border-color: #a0d4a3; }
  .card-orange { background: var(--orange); border-color: #f5c266; }
  .card-grey { background: #e0e2e4; border-color: #b0b4b8; }
  .card pre { white-space: pre-wrap; word-break: break-word; font-family: inherit;
              font-size: 14px; }
  .card code { background: rgba(0,0,0,.06); padding: 2px 6px; border-radius: 4px;
               font-size: 13px; }
  .md-content { font-size: 14px; line-height: 1.6; }
  .md-content p { margin: 0 0 8px; }
  .md-content table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
  .md-content th, .md-content td { border: 1px solid #b7e1cd; padding: 4px 8px; }
  .md-content th { background: rgba(0,0,0,.04); }
  .md-content pre { background: rgba(0,0,0,.04); padding: 8px; border-radius: 4px;
                    overflow-x: auto; }
  .md-content ul, .md-content ol { margin: 4px 0 8px 20px; }
  .meta { font-size: 12px; color: #80868b; margin-bottom: 12px; }

  /* -- scoring -- */
  .scoring { background: var(--card); border-radius: 8px; padding: 20px;
             border: 1px solid #c8c8c8; margin-bottom: 16px; }
  .scoring h3 { margin-bottom: 16px; font-size: 18px; }
  .metric-row { margin-bottom: 16px; }
  .metric-label { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
  .metric-help { font-size: 12px; color: #5f6368; margin-bottom: 6px; }
  .radio-group { display: flex; gap: 0; }
  .radio-group label { flex: 1; text-align: center; padding: 8px 4px; font-size: 13px;
    border: 1px solid #b8bcc0; cursor: pointer; transition: all .15s;
    background: var(--card); user-select: none; }
  .radio-group label:first-child { border-radius: 6px 0 0 6px; }
  .radio-group label:last-child { border-radius: 0 6px 6px 0; }
  .radio-group label:not(:first-child) { border-left: none; }
  .radio-group input { display: none; }
  .radio-group input:checked + span { /* handled via parent */ }
  .radio-group label:has(input:checked) {
    background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 600; }
  .radio-group label:has(input:checked) + label { border-left-color: var(--accent); }
  .metric-row.invalid .radio-group label { border-color: var(--red); }
  .metric-row.invalid .metric-label { color: var(--red); }

  textarea { width: 100%; padding: 10px; border: 1px solid #b8bcc0; border-radius: 6px;
             font-family: inherit; font-size: 14px; resize: vertical; min-height: 60px; }
  .error-msg { color: var(--red); font-size: 13px; font-weight: 500; margin-bottom: 12px;
               display: none; }
  .error-msg.show { display: block; }

  /* -- keyboard hint -- */
  .hint { font-size: 12px; color: #80868b; margin-top: 12px; text-align: center; }
  kbd { background: #f1f3f4; border: 1px solid #b8bcc0; border-radius: 3px;
        padding: 1px 5px; font-size: 11px; }
</style>
</head>
<body>
<div class="container">

  <!-- Progress -->
  <div class="progress-bar"><div class="progress-fill" style="width:{{ (completed / total * 100) | round(1) }}%"></div></div>
  <div class="progress-text">{{ completed }}/{{ total }} cases annotated</div>

  <!-- Nav -->
  <form method="GET" action="/case" style="display:inline" id="nav-form">
  <div class="nav">
    {% if idx > 0 %}
    <a href="/case?idx={{ idx - 1 }}">&larr; Previous</a>
    {% else %}
    <a style="opacity:.4;pointer-events:none">&larr; Previous</a>
    {% endif %}

    <select name="idx" onchange="document.getElementById('nav-form').submit()">
    {% for i in range(total) %}
      <option value="{{ i }}" {{ 'selected' if i == idx }}>
        Case {{ i + 1 }} (ID: {{ cases[i].annotation_id }}){% if cases[i].is_done %} &#10003;{% endif %}
      </option>
    {% endfor %}
    </select>

    {% if idx < total - 1 %}
    <a href="/case?idx={{ idx + 1 }}">Next &rarr;</a>
    {% else %}
    <a style="opacity:.4;pointer-events:none">Next &rarr;</a>
    {% endif %}

    <span class="badge {{ 'badge-done' if case.is_done else 'badge-pending' }}">
      {{ 'DONE' if case.is_done else 'PENDING' }}
    </span>
  </div>
  </form>

  <!-- Context cards -->
  <div class="meta">Category: <b>{{ case.category }}</b> &middot; Test: <b>{{ case.test_name }}</b> &middot; Session: {{ case.session }}</div>

  <div class="card card-blue">
    <h4>Question</h4>
    <pre>{{ case.question }}</pre>
  </div>

  <div class="card card-green">
    <h4>Agent Output</h4>
    <div class="md-content">{{ case.agent_output_html | safe }}</div>
  </div>

  <div class="card card-orange">
    <h4>Evaluation Context</h4>
    <pre>{{ case.evaluation_context if case.evaluation_context else 'No additional context' }}</pre>
  </div>

  <div class="card card-grey">
    <h4>Tool Calls</h4>
    <p><b>Actual:</b> <code>{{ case.tool_calls }}</code></p>
    <p><b>Expected:</b> <code>{{ case.expected_tool_calls }}</code></p>
    <p><b>Accepted:</b> <code>{{ case.accepted_tool_calls }}</code></p>
  </div>

  <!-- Scoring form -->
  <form method="POST" action="/save" id="score-form">
  <input type="hidden" name="idx" value="{{ idx }}">
  <div class="scoring">
    <h3>{% if read_only %}Annotator Scores (read-only){% else %}Your Scores{% endif %}</h3>
    {% if read_only %}
    <div style="background:#fef3e0;border:1px solid #fdd99b;border-radius:6px;padding:8px 12px;font-size:13px;color:#b06000;margin-bottom:12px;">
      Read-only mode &mdash; browsing only, no changes will be saved.
    </div>
    {% endif %}
    <div class="error-msg" id="error-msg">Please fill in all score fields before saving.</div>

    {% for m in metrics %}
    <div class="metric-row" id="row-{{ m.key }}">
      <div class="metric-label">{{ m.label }}</div>
      <div class="metric-help">{{ m.help }}</div>
      <div class="radio-group">
        {% for opt in m.options %}
        <label>
          <input type="radio" name="{{ m.key }}" value="{{ opt }}"
            {{ 'checked' if case.scores[m.key] == opt }}
            {{ 'disabled' if read_only }}>
          <span>{{ opt }}</span>
        </label>
        {% endfor %}
      </div>
    </div>
    {% endfor %}

    <div class="metric-row">
      <div class="metric-label">Justification <span style="color:#80868b;font-weight:normal;font-size:12px;">(optional)</span></div>
      <div class="metric-help">Optionally provide a brief justification (1-2 sentences). Not required to submit.</div>
      <textarea name="human_justification" placeholder="Focus on anything surprising or where you were uncertain..."
        {{ 'readonly' if read_only }}>{{ case.scores.human_justification }}</textarea>
    </div>

    {% if not read_only %}
    <button type="submit" class="btn-save" style="margin-top:12px;padding:10px 32px;border-radius:6px;font-size:15px;font-weight:600;border:none;cursor:pointer;background:#1a73e8;color:#fff;">
      Save &amp; Next &rarr;
    </button>
    {% endif %}
  </div>
  </form>

  <div class="hint">
    Keyboard: <kbd>Alt</kbd>+<kbd>&larr;</kbd> previous &middot;
    <kbd>Alt</kbd>+<kbd>&rarr;</kbd> next &middot;
    <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save
  </div>
</div>

<script>
// Client-side validation: all radio groups must have a selection
document.getElementById('score-form').addEventListener('submit', function(e) {
  const metricKeys = {{ metric_keys_json | safe }};
  let missing = [];
  metricKeys.forEach(function(key) {
    const checked = document.querySelector('input[name="' + key + '"]:checked');
    const row = document.getElementById('row-' + key);
    if (!checked) {
      missing.push(key);
      row.classList.add('invalid');
    } else {
      row.classList.remove('invalid');
    }
  });
  if (missing.length > 0) {
    e.preventDefault();
    document.getElementById('error-msg').classList.add('show');
    document.getElementById('row-' + missing[0]).scrollIntoView({behavior: 'smooth', block: 'center'});
  }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  if (e.altKey && e.key === 'ArrowLeft') {
    {% if idx > 0 %}window.location = '/case?idx={{ idx - 1 }}';{% endif %}
  } else if (e.altKey && e.key === 'ArrowRight') {
    {% if idx < total - 1 %}window.location = '/case?idx={{ idx + 1 }}';{% endif %}
  } else if (e.ctrlKey && e.key === 'Enter') {
    document.getElementById('score-form').dispatchEvent(new Event('submit', {cancelable: true}));
    if (!document.getElementById('error-msg').classList.contains('show')) {
      document.getElementById('score-form').submit();
    }
  }
});
</script>
</body>
</html>
"""


DONE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annotation Complete</title>
<style>
  :root { --bg: #f8f9fa; --card: #fff; --accent: #1a73e8; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: #202124; line-height: 1.6;
         display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  .done-card { background: var(--card); border-radius: 12px; padding: 48px;
               max-width: 600px; width: 100%; text-align: center;
               border: 1px solid #c8c8c8; box-shadow: 0 2px 8px rgba(0,0,0,.06); }
  .checkmark { font-size: 64px; margin-bottom: 16px; }
  h1 { font-size: 28px; margin-bottom: 8px; color: #137333; }
  .subtitle { font-size: 16px; color: #5f6368; margin-bottom: 24px; }
  .stat { background: #e6f4ea; border-radius: 8px; padding: 16px; margin-bottom: 16px;
          font-size: 18px; font-weight: 600; color: #137333; }
  .path-box { background: #f1f3f4; border-radius: 6px; padding: 12px; margin-bottom: 24px;
              font-family: monospace; font-size: 13px; word-break: break-all; color: #202124;
              text-align: left; }
  .path-label { font-size: 12px; color: #5f6368; text-transform: uppercase;
                letter-spacing: .5px; margin-bottom: 4px; font-family: sans-serif; }
  .actions { display: flex; gap: 12px; justify-content: center; margin-top: 8px; }
  .actions a { padding: 10px 24px; border-radius: 6px; font-size: 14px; font-weight: 500;
               text-decoration: none; border: 1px solid #b8bcc0; color: #202124;
               transition: background .15s; }
  .actions a:hover { background: #f1f3f4; }
  .actions a.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .actions a.primary:hover { background: #1557b0; }
</style>
</head>
<body>
<div class="done-card">
  <div class="checkmark">&#10004;</div>
  <h1>Annotation Complete</h1>
  <p class="subtitle">Thank you! All cases have been annotated.</p>

  <div class="stat">{{ completed }} / {{ total }} cases scored</div>

  <div>
    <div class="path-label">Annotated file saved to</div>
    <div class="path-box">{{ output_path }}</div>
  </div>

  <div>
    <div class="path-label">Original CSV (unchanged)</div>
    <div class="path-box">{{ csv_path }}</div>
  </div>

  <div class="actions">
    <a href="/">Review annotations</a>
  </div>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
class CaseView:
    """Lightweight view object for a single annotation case."""

    def __init__(self, row: pd.Series, metrics: list):
        self.annotation_id = row["annotation_id"]
        self.question = row["question"]
        self.agent_output = row["agent_output"]
        self.agent_output_html = markdown.markdown(
            str(row["agent_output"]), extensions=["tables", "fenced_code"]
        )
        self.evaluation_context = str(row.get("evaluation_context", "") or "").strip()
        self.tool_calls = row["tool_calls"]
        self.expected_tool_calls = row["expected_tool_calls"]
        self.accepted_tool_calls = row["accepted_tool_calls"]
        self.category = row["category"]
        self.test_name = row["test_name"]
        self.session = row["session"]
        self.scores = {
            m["key"]: str(row.get(m["key"], "") or "").strip() for m in metrics
        }
        self.scores["human_justification"] = str(
            row.get("human_justification", "") or ""
        ).strip()
        self.is_done = all(self.scores[m["key"]] != "" for m in metrics)


def load_data(csv_path: str, metrics: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for m in metrics:
        if m["key"] not in df.columns:
            df[m["key"]] = ""
        df[m["key"]] = df[m["key"]].fillna("").astype(str)
    if "human_justification" not in df.columns:
        df["human_justification"] = ""
    df["human_justification"] = df["human_justification"].fillna("").astype(str)
    return df


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def create_app(csv_path: str, read_only: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["READ_ONLY"] = read_only
    metrics = METRICS
    metric_keys = [m["key"] for m in metrics]
    csv_path = str(Path(csv_path).resolve())
    working_path = csv_path.replace(".csv", "_in_progress.csv")
    output_path = csv_path.replace(".csv", "_annotated.csv")

    # In read-only mode, load directly from the specified file (no working copy)
    if read_only:
        df = load_data(csv_path, metrics)
        print(f"READ-ONLY MODE: viewing {csv_path}")
    elif Path(working_path).exists():
        df = load_data(working_path, metrics)
        print(f"Resuming from working copy: {working_path}")
    else:
        df = load_data(csv_path, metrics)

    def _save_df():
        df.to_csv(working_path, index=False)

    def _count_completed():
        scored = df[metric_keys].replace("", pd.NA)
        return int(scored.dropna(how="any").shape[0])

    def _all_cases():
        return [CaseView(df.iloc[i], metrics) for i in range(len(df))]

    @app.route("/")
    def index():
        return redirect(url_for("case_view", idx=0))

    @app.route("/case")
    def case_view():
        idx = int(request.args.get("idx", 0))
        idx = max(0, min(idx, len(df) - 1))
        case = CaseView(df.iloc[idx], metrics)
        cases = _all_cases()
        return render_template_string(
            TEMPLATE,
            idx=idx,
            total=len(df),
            completed=_count_completed(),
            case=case,
            cases=cases,
            metrics=metrics,
            metric_keys_json=json.dumps(metric_keys),
            read_only=app.config["READ_ONLY"],
        )

    @app.route("/save", methods=["POST"])
    def save():
        if app.config["READ_ONLY"]:
            return Response("Read-only mode: saving is disabled.", status=403)

        idx = int(request.form["idx"])
        idx = max(0, min(idx, len(df) - 1))

        for m in metrics:
            val = request.form.get(m["key"], "").strip()
            df.at[idx, m["key"]] = val

        df.at[idx, "human_justification"] = request.form.get(
            "human_justification", ""
        ).strip()

        _save_df()

        # If all cases are done, go to completion page
        if _count_completed() == len(df):
            return redirect(url_for("done"))

        # Advance to next case (or stay on last)
        next_idx = min(idx + 1, len(df) - 1)
        return redirect(url_for("case_view", idx=next_idx))

    @app.route("/done")
    def done():
        completed = _count_completed()
        total = len(df)
        df.to_csv(output_path, index=False)
        return render_template_string(
            DONE_TEMPLATE,
            completed=completed,
            total=total,
            output_path=output_path,
            csv_path=csv_path,
        )

    @app.route("/export")
    def export():
        if app.config["READ_ONLY"]:
            return Response("Read-only mode: export is disabled.", status=403)
        df.to_csv(output_path, index=False)
        completed = _count_completed()
        return (
            f"<h2>Exported</h2>"
            f"<p>Saved to <code>{output_path}</code></p>"
            f"<p>{completed}/{len(df)} cases completed.</p>"
            f'<p><a href="/">Back to annotation</a></p>'
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Local annotation web app")
    parser.add_argument(
        "--csv",
        default="annotation_sample_annotator_sheet.csv",
        help="Path to annotator sheet CSV",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open browser"
    )
    parser.add_argument(
        "--read-only", action="store_true", help="Browse annotations without saving"
    )
    args = parser.parse_args()

    app = create_app(args.csv, read_only=args.read_only)

    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    csv_resolved = str(Path(args.csv).resolve())
    print(f"Annotation app running at http://localhost:{args.port}")
    print(f"Original CSV (read-only): {csv_resolved}")
    print(f"Working copy: {csv_resolved.replace('.csv', '_in_progress.csv')}")
    print(f"Export URL: http://localhost:{args.port}/export")
    print("Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
