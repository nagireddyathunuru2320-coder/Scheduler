import json, os, re
from typing import TypedDict, List, Dict, Any, Optional
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

load_dotenv()
app = Flask(__name__)

def get_api_key() -> Optional[str]:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

def get_llm():
    key = get_api_key()
    if not key:
        return None
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0.2)

def parse_json_safely(raw: Any) -> dict:
    if hasattr(raw, "content"):
        raw = raw.content
    if isinstance(raw, list):
        raw = "".join([c.get("text", "") if isinstance(c, dict) else getattr(c, "text", str(c)) for c in raw])
    text = str(raw or "").strip()
    clean = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE)
    match = re.search(r"(\{.*\})", clean, re.DOTALL)
    try:
        return json.loads(match.group(1) if match else clean)
    except Exception:
        return {"summary": text[:200], "schedule_text": text, "items": [], "conflicts": [], "reasoning": "Generated from request constraints."}

class SchedulerState(TypedDict, total=False):
    user_request: str
    date_context: str
    availability: str
    priority_pref: str
    events: List[Dict[str, Any]]
    constraints: List[str]
    conflicts: List[Dict[str, Any]]
    schedule: str
    items: List[Dict[str, Any]]
    summary: str
    reasoning: str
    clarification: str
    error: Optional[str]

EXTRACT_PROMPT = """You are the Entity & Constraint Extraction Engine for AI Scheduler.
Analyze the user's scheduling request and parameters.
Rules:
1. FIXED EVENTS: Cannot move (classes, exams, meetings, appointments, flights). Extract exact or stated times.
2. FLEXIBLE TASKS: Can move (studying, exercise, project work, reading, personal chores). Extract duration and preferred time windows.
3. DEADLINES: Identify deadlines (e.g. 'submit before Friday') that require preparatory work sessions.
4. RECURRENCE: Understand daily, weekdays, weekends, weekly patterns.
5. TIME WINDOWS: Interpret morning, afternoon, evening, night, early morning, after lunch, before class, after work.
6. CONSTRAINTS: Extract user availability bounds and stated preferences.
7. AMBIGUITY: If critical info is ambiguous, provide a short clarification question.
Return ONLY valid JSON (no markdown fences):
{"events": [{"name": "...", "type": "fixed|flexible", "start": "...", "end": "...", "duration": "...", "priority": "High|Medium|Low", "deadline": "...", "recurrence": "...", "preferred_time": "..."}], "constraints": ["..."], "clarification": "..."}"""

SCHEDULE_PROMPT = """You are the Schedule Generation, Conflict Detection & Optimization Engine for AI Scheduler.
Rules:
1. PRESERVE FIXED EVENTS: Never move or drop fixed events (classes, exams, meetings, appointments).
2. CONFLICT DETECTION: Identify overlaps (e.g., 10:00-11:00 Meeting vs 10:30-12:00 Gym). In 'conflicts', record conflicting_events, exact reason, and suggested alternatives. Never silently drop events.
3. DEADLINES: For tasks with deadlines (e.g. 'Finish project before Friday'), distribute preparatory focus sessions leading up to the deadline instead of merely placing an event at the deadline.
4. FLEXIBLE PLACEMENT: Place flexible tasks into free slots respecting priority (High first), preferred times, and user availability.
5. PACING: Insert realistic 10-15 min breaks between demanding blocks; avoid unrealistic back-to-back schedules.
6. TRANSPARENCY: Provide a concise summary and reasoning explaining scheduling choices.
7. LIMITATION: You are an intelligent schedule planner only; do not claim calendar sync or notification sending.
Return ONLY valid JSON (no markdown fences):
{
  "summary": "1-2 sentence overview",
  "schedule_text": "Structured text schedule with headers, times, tasks, priority, and status",
  "items": [{"day": "...", "time": "HH:MM - HH:MM", "task": "...", "type": "Fixed|Flexible|Break|Deadline", "priority": "High|Medium|Low", "status": "Scheduled|Conflict|Needs clarification", "notes": "..."}],
  "conflicts": [{"conflicting_events": "...", "reason": "...", "suggestion": "..."}],
  "reasoning": "Explanation for key scheduling choices",
  "clarification": "..."
}"""

def format_schedule_items(items: List[Dict[str, Any]]) -> str:
    lines, curr_day = [], ""
    for it in items:
        day = it.get("day", "")
        if day and day != curr_day:
            curr_day = day
            lines.append(f"\n{curr_day.upper()}")
        lines.append(f"{it.get('time', '')} | {it.get('task', '')} [{it.get('type', 'Task')}] - Priority: {it.get('priority', 'Medium')} ({it.get('status', 'Scheduled')})")
    return "\n".join(lines).strip()

def extract_node(state: SchedulerState) -> Dict[str, Any]:
    llm = get_llm()
    if not llm:
        return {"error": "Google Gemini API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY."}
    try:
        user_msg = f"Request: {state['user_request']}\nDate Context: {state.get('date_context')}\nAvailability: {state.get('availability')}\nPriority Strategy: {state.get('priority_pref')}"
        resp = llm.invoke([SystemMessage(content=EXTRACT_PROMPT), HumanMessage(content=user_msg)])
        data = parse_json_safely(resp)
        return {
            "events": data.get("events", []),
            "constraints": data.get("constraints", []),
            "clarification": data.get("clarification", ""),
            "error": None
        }
    except Exception as e:
        return {"events": [], "constraints": [state["user_request"]], "clarification": "", "error": str(e)}

def schedule_node(state: SchedulerState) -> Dict[str, Any]:
    if state.get("error"):
        return {"error": state.get("error")}
    llm = get_llm()
    if not llm:
        return {"error": "Google Gemini API key not found."}
    try:
        context = f"Request: {state['user_request']}\nDate Context: {state.get('date_context')}\nAvailability: {state.get('availability')}\nPriority Strategy: {state.get('priority_pref')}\nExtracted Events: {json.dumps(state.get('events', []))}\nConstraints: {json.dumps(state.get('constraints', []))}"
        resp = llm.invoke([SystemMessage(content=SCHEDULE_PROMPT), HumanMessage(content=context)])
        data = parse_json_safely(resp)
        items = data.get("items", [])
        sched_text = data.get("schedule_text", "")
        if not sched_text and items:
            sched_text = format_schedule_items(items)
        return {
            "summary": data.get("summary", "Schedule organized successfully."),
            "schedule": sched_text,
            "items": items,
            "conflicts": data.get("conflicts", []),
            "reasoning": data.get("reasoning", "Optimized based on fixed commitments and priorities."),
            "clarification": data.get("clarification", state.get("clarification", "")),
            "error": None
        }
    except Exception as e:
        return {"error": str(e)}

workflow = StateGraph(SchedulerState)
workflow.add_node("extract_constraints", extract_node)
workflow.add_node("build_schedule", schedule_node)
workflow.set_entry_point("extract_constraints")
workflow.add_edge("extract_constraints", "build_schedule")
workflow.add_edge("build_schedule", END)
scheduler_graph = workflow.compile()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Scheduler | Intelligent Task & Schedule Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#090d16;--card:#121827;--border:#1e293b;--text:#f1f5f9;--muted:#94a3b8;--primary:#6366f1;--glow:rgba(99,102,241,0.25)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.5;min-height:100vh;padding:24px 16px;background-image:radial-gradient(circle at 50% 0%,rgba(99,102,241,0.12),transparent 55%)}
.container{max-width:960px;margin:0 auto}
header{text-align:center;margin-bottom:28px}
.badge-tag{display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);color:#a5b4fc;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600;margin-bottom:12px}
h1{font-size:2.2rem;font-weight:700;letter-spacing:-0.5px;background:linear-gradient(135deg,#fff 30%,#94a3b8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
p.sub{color:var(--muted);font-size:14px;margin-top:6px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px;margin-bottom:20px;box-shadow:0 8px 30px rgba(0,0,0,0.3)}
.prompt-chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.chip{background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:16px;padding:5px 12px;font-size:12px;cursor:pointer;transition:all .15s}
.chip:hover{background:#334155;color:#fff;border-color:var(--primary)}
label{display:block;font-size:12px;font-weight:600;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
textarea{width:100%;height:105px;background:#0b1120;border:1px solid #334155;border-radius:10px;padding:12px;color:var(--text);font-family:inherit;font-size:14px;resize:vertical}
textarea:focus,input:focus,select:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px var(--glow)}
.grid-3{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:14px}
input,select{width:100%;background:#0b1120;border:1px solid #334155;border-radius:8px;padding:9px 12px;color:var(--text);font-family:inherit;font-size:13px}
.btn-row{display:flex;gap:12px;margin-top:18px}
button.primary{background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;border:none;border-radius:8px;padding:11px 22px;font-weight:600;font-size:14px;cursor:pointer;display:inline-flex;align-items:center;gap:8px}
button.primary:hover{opacity:.92}
button.primary:disabled{opacity:.5;cursor:not-allowed}
button.secondary{background:transparent;color:var(--muted);border:1px solid #334155;border-radius:8px;padding:11px 18px;font-size:13px;cursor:pointer}
button.secondary:hover{background:#1e293b;color:#fff}
.alert{border-radius:10px;padding:14px 16px;margin-bottom:18px;font-size:13px;display:none}
.alert-error{background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);color:#fca5a5}
.alert-warning{background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);color:#fcd34d}
.alert-info{background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.3);color:#67e8f9}
.summary-box{background:linear-gradient(180deg,rgba(99,102,241,0.08),rgba(15,23,42,0.6));border:1px solid rgba(99,102,241,0.25);border-radius:12px;padding:16px;margin-bottom:18px}
.summary-title{font-size:15px;font-weight:600;color:#c7d2fe;margin-bottom:4px}
.reasoning-text{font-size:13px;color:var(--muted);margin-top:8px;border-top:1px solid #1e293b;padding-top:8px}
.timeline{display:flex;flex-direction:column;gap:10px}
.event-card{display:flex;align-items:center;gap:14px;background:#0b1120;border:1px solid #1e293b;border-radius:10px;padding:12px 16px}
.time-col{font-weight:600;font-size:13px;color:#38bdf8;min-width:120px}
.info-col{flex:1}
.task-title{font-weight:600;font-size:14px;color:#f8fafc}
.task-notes{font-size:12px;color:#64748b;margin-top:2px}
.badge-group{display:flex;gap:6px;flex-wrap:wrap}
.badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:12px;text-transform:uppercase}
.b-fixed{background:rgba(168,85,247,0.15);color:#c084fc;border:1px solid rgba(168,85,247,0.3)} .b-flex{background:rgba(59,130,246,0.15);color:#60a5fa;border:1px solid rgba(59,130,246,0.3)} .b-break{background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3)}
.b-high{background:rgba(239,68,68,0.15);color:#f87171} .b-med{background:rgba(245,158,11,0.15);color:#fbbf24} .b-low{background:rgba(100,116,139,0.15);color:#94a3b8} .b-conf{background:rgba(239,68,68,0.2);color:#fca5a5;border:1px solid #ef4444}
.raw-box{background:#0b1120;border:1px solid #1e293b;border-radius:8px;padding:14px;font-family:monospace;font-size:12px;color:#cbd5e1;white-space:pre-wrap;margin-top:14px}
.copy-btn{float:right;background:#1e293b;border:1px solid #334155;color:#cbd5e1;border-radius:6px;padding:4px 8px;font-size:11px;cursor:pointer}
.copy-btn:hover{color:#fff;background:#334155}
footer{text-align:center;font-size:12px;color:#64748b;margin-top:32px}
.spinner{width:14px;height:14px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge-tag">⚡ AI Scheduler • LangGraph Engine</div>
    <h1>Intelligent Schedule Assistant</h1>
    <p class="sub">Natural-language task planning, fixed-event preservation & conflict resolution</p>
  </header>
  <div class="card">
    <div class="prompt-chips">
      <span class="chip" onclick="fillPrompt('I have college from 9 AM to 4 PM tomorrow. Schedule 2 hours of study, 1 hour gym and 2 hours project work.')">College 9-4 + Study & Gym</span>
      <span class="chip" onclick="fillPrompt('I have a meeting at 10 AM and an exam at 3 PM tomorrow. Find time for gym and study.')">Meeting 10am, Exam 3pm, Gym & Study</span>
      <span class="chip" onclick="fillPrompt('Finish my project before Friday. Break it into daily work sessions and plan them.')">Finish Project Before Friday</span>
      <span class="chip" onclick="fillPrompt('Create a balanced daily routine for weekdays including morning workout, deep work, and reading.')">Daily Routine</span>
    </div>
    <label for="prompt">Describe Tasks, Meetings & Deadlines</label>
    <textarea id="prompt" placeholder="e.g., I have a meeting tomorrow from 10:00 to 11:30 AM. Need 2 hours study, 1 hour gym in the evening, and lunch break..."></textarea>
    <div class="grid-3">
      <div><label for="dateContext">Target Date (Optional)</label><input id="dateContext" type="text" placeholder="e.g. Tomorrow, Next week"></div>
      <div><label for="availability">Availability (Optional)</label><input id="availability" type="text" placeholder="e.g. 08:00 - 22:00"></div>
      <div>
        <label for="priority">Priority Strategy</label>
        <select id="priority">
          <option value="Balanced">Balanced Priority</option>
          <option value="High Priority First">High Priority First</option>
          <option value="Deep Work Focus">Deep Work & Study Focus</option>
          <option value="Health & Routine">Health & Recovery Pacing</option>
        </select>
      </div>
    </div>
    <div class="btn-row">
      <button class="primary" id="btnSubmit" onclick="generateSchedule()">
        <span class="spinner" id="spinner"></span><span id="btnText">✨ Generate Schedule</span>
      </button>
      <button class="secondary" onclick="clearForm()">Clear</button>
    </div>
  </div>
  <div id="errAlert" class="alert alert-error"></div>
  <div id="clarifyAlert" class="alert alert-info"></div>
  <div id="conflictAlert" class="alert alert-warning"></div>
  <div id="resultSection" style="display:none">
    <div class="summary-box">
      <div class="summary-title" id="summaryText"></div>
      <div class="reasoning-text" id="reasoningText"></div>
    </div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <h3 style="font-size:16px;font-weight:600">Generated Schedule</h3>
        <button class="copy-btn" onclick="copySchedule()">📋 Copy Text</button>
      </div>
      <div class="timeline" id="timelineList"></div>
      <div class="raw-box" id="rawSchedule"></div>
    </div>
  </div>
  <footer>
    AI Scheduler is an intelligent planning assistant powered by Google Gemini & LangGraph.<br>
    Stateless reasoning engine • Preserves fixed events & resolves conflicts • No external calendar modification.
  </footer>
</div>
<script>
function fillPrompt(txt){ document.getElementById('prompt').value = txt; }
function clearForm(){
  document.getElementById('prompt').value = '';
  document.getElementById('resultSection').style.display = 'none';
  hideAlerts();
}
function hideAlerts(){
  ['errAlert','clarifyAlert','conflictAlert'].forEach(id=>{
    const el = document.getElementById(id); el.style.display='none'; el.innerHTML='';
  });
}
function copySchedule(){
  const txt = document.getElementById('rawSchedule').innerText;
  navigator.clipboard.writeText(txt).then(()=>alert('Schedule copied to clipboard!'));
}
async function generateSchedule(){
  const msg = document.getElementById('prompt').value.trim();
  hideAlerts();
  if(!msg){
    const err = document.getElementById('errAlert');
    err.innerText = 'Please describe your tasks or schedule before submitting.';
    err.style.display = 'block';
    return;
  }
  const btn = document.getElementById('btnSubmit'), spinner = document.getElementById('spinner'), btnText = document.getElementById('btnText');
  btn.disabled = true; spinner.style.display = 'inline-block'; btnText.innerText = 'Analyzing & Optimizing...';
  try {
    const res = await fetch('/api/schedule', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: msg,
        date: document.getElementById('dateContext').value.trim(),
        availability: document.getElementById('availability').value.trim(),
        priority: document.getElementById('priority').value
      })
    });
    const data = await res.json();
    if(!res.ok || data.status === 'error'){
      throw new Error(data.error || 'Failed to generate schedule.');
    }
    renderResults(data);
  } catch(e){
    const err = document.getElementById('errAlert');
    err.innerText = e.message || 'Error communicating with scheduler service.';
    err.style.display = 'block';
  } finally {
    btn.disabled = false; spinner.style.display = 'none'; btnText.innerText = '✨ Generate Schedule';
  }
}
function renderResults(data){
  document.getElementById('resultSection').style.display = 'block';
  document.getElementById('summaryText').innerText = data.summary || 'Schedule generated.';
  document.getElementById('reasoningText').innerText = 'AI Reasoning: ' + (data.reasoning || 'Optimized slots according to priority and constraints.');
  if(data.clarification){
    const cl = document.getElementById('clarifyAlert');
    cl.innerHTML = '<strong>Clarification note:</strong> ' + data.clarification;
    cl.style.display = 'block';
  }
  if(data.conflicts && data.conflicts.length > 0){
    const cf = document.getElementById('conflictAlert');
    let html = '<strong>⚠️ Conflicts Detected:</strong><ul style="margin-top:6px;padding-left:18px">';
    data.conflicts.forEach(c=>{
      html += `<li><strong>${c.conflicting_events || 'Conflict'}:</strong> ${c.reason || ''} <em>(Alternative: ${c.suggestion || 'Adjust time'})</em></li>`;
    });
    html += '</ul>';
    cf.innerHTML = html; cf.style.display = 'block';
  }
  const list = document.getElementById('timelineList');
  list.innerHTML = '';
  const items = data.items || [];
  if(items.length > 0){
    items.forEach(it=>{
      const card = document.createElement('div');
      card.className = 'event-card';
      const typeClass = (it.type || '').toLowerCase().includes('fix') ? 'b-fixed' : (it.type || '').toLowerCase().includes('break') ? 'b-break' : 'b-flex';
      const prioClass = (it.priority || '').toLowerCase() === 'high' ? 'b-high' : (it.priority || '').toLowerCase() === 'low' ? 'b-low' : 'b-med';
      const isConf = (it.status || '').toLowerCase().includes('conflict');
      card.innerHTML = `
        <div class="time-col">${it.time || ''}</div>
        <div class="info-col">
          <div class="task-title">${it.task || ''}</div>
          ${it.notes ? `<div class="task-notes">${it.notes}</div>` : ''}
        </div>
        <div class="badge-group">
          <span class="badge ${typeClass}">${it.type || 'Task'}</span>
          <span class="badge ${prioClass}">${it.priority || 'Medium'}</span>
          <span class="badge ${isConf ? 'b-conf' : ''}">${it.status || 'Scheduled'}</span>
        </div>
      `;
      list.appendChild(card);
    });
  } else {
    list.innerHTML = '<div style="color:#94a3b8;font-size:13px">See formatted schedule below.</div>';
  }
  document.getElementById('rawSchedule').innerText = data.schedule || 'No schedule text returned.';
}
</script>
</body>
</html>"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/api/schedule", methods=["POST"])
def api_schedule():
    if not request.is_json:
        return jsonify({"status": "error", "error": "Request body must be JSON."}), 400
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"status": "error", "error": "Please provide a scheduling request in 'message'."}), 400
    if not get_api_key():
        return jsonify({
            "status": "error",
            "error": "Google Gemini API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY.",
            "schedule": "",
            "conflicts": []
        }), 400
    try:
        init_state = {
            "user_request": message,
            "date_context": str(data.get("date") or "").strip(),
            "availability": str(data.get("availability") or "").strip(),
            "priority_pref": str(data.get("priority") or "Balanced").strip(),
            "events": [], "constraints": [], "conflicts": [], "items": [],
            "schedule": "", "summary": "", "reasoning": "", "clarification": "", "error": None
        }
        res = scheduler_graph.invoke(init_state)
        err = res.get("error")
        if err:
            err_msg = str(err)
            if "API key not valid" in err_msg or "INVALID_ARGUMENT" in err_msg and "key" in err_msg.lower():
                user_msg = "Invalid Gemini API key. Please verify your GOOGLE_API_KEY or GEMINI_API_KEY."
            elif "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "quota" in err_msg.lower():
                user_msg = "Gemini API quota exceeded or rate limit reached. Please wait a moment and try again."
            else:
                user_msg = f"Scheduling optimization error: {err_msg[:120]}"
            return jsonify({"status": "error", "error": user_msg, "schedule": "", "conflicts": []}), 500
        return jsonify({
            "status": "success",
            "schedule": res.get("schedule", ""),
            "conflicts": res.get("conflicts", []),
            "items": res.get("items", []),
            "summary": res.get("summary", ""),
            "reasoning": res.get("reasoning", ""),
            "clarification": res.get("clarification", "")
        })
    except Exception:
        return jsonify({"status": "error", "error": "Internal scheduling service error.", "schedule": "", "conflicts": []}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
