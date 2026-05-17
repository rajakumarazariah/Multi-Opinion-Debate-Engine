# ⚖️ Multi-Opinion Debate Engine

A LangGraph-powered reasoning system that generates structured, multi-perspective answers to complex technical and strategic questions. Two LLM "personas" argue independently, a critic evaluates their positions, and a synthesizer produces a final blended answer — all driven by an iterative critique-refine loop.

---

## How It Works

```
Technical Perspective
        ↓
Practical Perspective
        ↓
      Critic  ──── major gaps? ──→ loop back ↑
        ↓ (no gaps / max iterations reached)
    Synthesizer
        ↓
   Final Answer
```

Each run through the graph is one **iteration**. The critic decides whether to loop back for another round (up to a configurable maximum) or proceed to synthesis.

### The four nodes

| Node | Model | Role |
|---|---|---|
| **Technical Perspective** | Llama 3.3 70B (Groq) | Architecture, algorithms, scalability, security — no business talk |
| **Practical Perspective** | Llama 3.3 70B (Groq) | Cost, team skills, timelines, ROI — no low-level technical detail |
| **Critic** | Llama 3.3 70B (Groq, temp 0.3) | Finds contradictions, weak assumptions, and missing angles; emits structured JSON to drive loop control |
| **Synthesizer** | Llama 3.3 70B (Groq, temp 0.2) | Merges both perspectives into a coherent final answer with a confidence score |

---

## Project Structure

```
.
├── debate_engine.py   # Core LangGraph graph, nodes, and runner
├── app.py             # Streamlit web UI
├── .env               # API keys (not committed)
└── debate_result.json # Auto-saved output when run from the CLI
```

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit langgraph langchain-groq langchain-core python-dotenv
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Run the web UI

```bash
streamlit run app.py
```

### 4. Run from the CLI (optional)

```bash
# Default question
python debate_engine.py

# Custom question
python debate_engine.py "Should we rewrite our Python monolith in Rust?"

# Control iteration count via env var
MAX_ITERATIONS=3 python debate_engine.py "Microservices vs monolith for a B2B SaaS?"
```

The CLI saves results to `debate_result.json` automatically.

---

## Configuration

| Option | Where | Default | Description |
|---|---|---|---|
| `max_iterations` | UI slider / CLI env var | 2 | Max critique–refine loops before forcing synthesis |
| `MAX_ITERATIONS` | `MAX_ITERATIONS=N` env var | 2 | CLI override for iteration cap |
| `LANGCHAIN_PROJECT` | `debate_engine.py` | `multi-opinion-debate-engine-groq` | LangSmith project name (optional tracing) |

---

## Output

Every run produces a structured result with these fields:

```json
{
  "query": "...",
  "technical_perspective": "...",
  "practical_perspective": "...",
  "critique": "...",
  "final_answer": "...",
  "confidence_score": 0.87,
  "iteration": 2,
  "reasoning_trace": ["[12:01:05] Technical Perspective generated (1240 chars)", "..."],
  "iteration_history": [
    { "iteration": 1, "has_major_gaps": true, "gap_summary": "Security risks unaddressed" },
    { "iteration": 2, "has_major_gaps": false, "gap_summary": "Perspectives well-balanced" }
  ]
}
```

The Streamlit UI lets you download this as JSON via the **⬇ Download full result** button.

---

## Example Questions

The engine works best on questions with both a technical dimension and a practical/business dimension:

- Should a 50-engineer startup adopt Kubernetes or stick with Heroku/Railway?
- Should we rewrite our Python monolith in Rust for better performance?
- Should we adopt GitHub Copilot across the engineering team?
- Microservices vs monolith for a B2B SaaS reaching 100k users?
- Should we self-host our LLM or use a managed API like OpenAI?

---

## Architecture Notes

- **State management** — `DebateState` is a `TypedDict` that LangGraph passes between nodes. Each node returns only the fields it modifies; LangGraph shallow-merges the update back into the shared state.
- **Loop control** — The critic node emits a `has_major_gaps` boolean parsed from a required JSON block. The `should_loop` routing function reads this alongside the iteration counter to decide between `"loop"` and `"synthesize"`.
- **Temperature discipline** — Generative nodes run at 0.7 for creative breadth; the critic at 0.3 for consistent structured output; the synthesizer at 0.2 for stable, deterministic final answers.
- **Reasoning trace** — Each node appends a timestamped entry to `reasoning_trace`, giving full auditability of what ran and when.
