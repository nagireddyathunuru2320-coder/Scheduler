import os
import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent


# =========================
# FASTAPI
# =========================

app = FastAPI()


# =========================
# API KEY & MODELS
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )


# =========================
# SCHEDULE DATA
# =========================

today = datetime.date.today()

schedule_db = {
    "1": {
        "title": "Team Meeting",
        "date": str(today + datetime.timedelta(days=1)),
        "time": "10:00 AM",
        "description": "Project discussion"
    },
    "2": {
        "title": "AI Workshop",
        "date": str(today + datetime.timedelta(days=1)),
        "time": "2:00 PM",
        "description": "Workshop on AI and Machine Learning"
    },
    "3": {
        "title": "Project Review",
        "date": str(today + datetime.timedelta(days=2)),
        "time": "11:00 AM",
        "description": "Review project progress"
    }
}


# =========================
# VECTOR STORE
# =========================

vector_store = None

def init_vector_store():
    global vector_store
    documents = [
        Document(
            page_content=(
                f"Title: {event['title']}\n"
                f"Date: {event['date']}\n"
                f"Time: {event['time']}\n"
                f"Description: {event['description']}"
            )
        )
        for event in schedule_db.values()
    ]

    if not documents:
        documents = [Document(page_content="No events are currently scheduled.")]

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name="personal_scheduler"
    )
    return vector_store


# =========================
# TOOLS
# =========================

@tool
def search_schedule(query: str) -> str:
    """Search the schedule for relevant events."""
    global vector_store
    if vector_store is None:
        init_vector_store()

    if not schedule_db:
        return "No events are currently scheduled."

    results = vector_store.similarity_search(query, k=min(3, len(schedule_db)))
    if not results:
        return "No matching events found."

    return "\n\n".join(doc.page_content for doc in results)


@tool
def add_event(title: str, date: str, time: str, description: str = "") -> str:
    """Add an event to the schedule."""
    if schedule_db:
        event_id = str(max(int(eid) for eid in schedule_db.keys()) + 1)
    else:
        event_id = "1"

    schedule_db[event_id] = {
        "title": title,
        "date": date,
        "time": time,
        "description": description
    }

    init_vector_store()
    return f"Added {title} on {date} at {time}."


@tool
def delete_event(title: str) -> str:
    """Delete an event using its title."""
    for event_id, event in list(schedule_db.items()):
        if event["title"].lower() == title.lower():
            del schedule_db[event_id]
            init_vector_store()
            return f"Deleted {title}."

    return f"Event '{title}' was not found."


tools = [search_schedule, add_event, delete_event]


# =========================
# AGENT SETUP
# =========================

def get_agent():
    system_prompt = f"""You are an intelligent RAG Schedule Agent.
Today's date is {today}.

Your responsibilities:
- Answer questions about the user's schedule.
- Search events using search_schedule.
- Add events using add_event.
- Delete events using delete_event.

Important rules:
- Never invent schedule information.
- Use search_schedule when the user asks about existing events.
- Keep answers clear and short.
- Do not reveal internal reasoning or thinking."""

    return create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=system_prompt
    )


# =========================
# REQUEST MODEL
# =========================

class UserQuery(BaseModel):
    query: str


# =========================
# EXTRACT TEXT
# =========================

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return str(content)


# =========================
# CHAT API
# =========================

@app.post("/chat")
def chat(request: UserQuery):
    try:
        if not os.getenv("GEMINI_API_KEY"):
            return {"response": "Error: GEMINI_API_KEY environment variable is not configured on the server."}

        agent = get_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": request.query}]
        })

        messages = result.get("messages", [])
        if not messages:
            return {"response": "Sorry, I could not generate a response."}

        answer = extract_text(messages[-1].content)
        return {"response": answer or "No response generated."}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}


# =========================
# UI & HEALTH ROUTES
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Scheduler Agent</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; }
        #chat { margin-top: 20px; margin-bottom: 20px; }
        .message { margin: 10px 0; }
        input { width: 400px; padding: 8px; font-size: 16px; }
        button { padding: 8px 15px; font-size: 16px; cursor: pointer; }
    </style>
</head>
<body>
    <h2>📅 RAG Scheduler Agent</h2>
    <p>Ask me about your schedule, add events, or delete events.</p>
    <div id="chat"></div>
    <input id="message" type="text" placeholder="Ask about your schedule...">
    <button onclick="sendMessage()">Send</button>

<script>
async function sendMessage() {
    const input = document.getElementById("message");
    const message = input.value.trim();
    if (!message) return;

    const chat = document.getElementById("chat");
    const userMessage = document.createElement("p");
    userMessage.className = "message";
    userMessage.innerHTML = `<b>You:</b> ${message}`;
    chat.appendChild(userMessage);

    input.value = "";

    const agentMessage = document.createElement("p");
    agentMessage.className = "message";
    agentMessage.innerHTML = `<b>Agent:</b> <i>Thinking...</i>`;
    chat.appendChild(agentMessage);

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: message })
        });
        const data = await response.json();
        agentMessage.innerHTML = `<b>Agent:</b> ${data.response}`;
    } catch (error) {
        agentMessage.innerHTML = `<b>Agent:</b> Connection error. Please try again.`;
    }
}

document.getElementById("message").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});
</script>
</body>
</html>
"""

@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
