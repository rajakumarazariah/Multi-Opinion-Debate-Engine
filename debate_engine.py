
import os                         
import json                       
from typing import TypedDict, List, Optional, Annotated
from datetime import datetime      

from dotenv import load_dotenv     

from langgraph.graph import StateGraph, END 
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()   

os.environ.setdefault("LANGCHAIN_PROJECT", "multi-opinion-debate-engine-groq")


# 1. GRAPH STATE  (the shared memory that flows between every node)
class DebateState(TypedDict):
    """
    TypedDict acts as a typed schema for the LangGraph state object.
    Every node receives this dict, may modify some fields, and returns the
    updated dict.  LangGraph merges the returned fields back into the state.
    """

    # Input 
    query: str                          

    #  Perspectives (filled by the two perspective nodes) 
    technical_perspective: str         
    practical_perspective: str         

    #  Critique (filled by the critic node) 
    critique: str                       # Full critique text
    has_major_gaps: bool                # True  → loop back for another round
    gap_summary: str                    # One-liner explaining the gaps found

    #  Synthesis (filled by the synthesizer node) 
    final_answer: str                   # The blended, coherent answer
    confidence_score: float             # 0.0–1.0; how confident the synthesizer is
    reasoning_trace: List[str]          # Step-by-step log of what each model did

    #  Loop control
    iteration: int                      # How many critique→refine cycles have run
    max_iterations: int                 # Hard cap to prevent infinite loops
    iteration_history: List[dict]       # Snapshot of every iteration for audit



# 2. LLM INSTANCES  ← THE ONLY SECTION THAT CHANGED FROM THE ORIGINAL

llama3_70b = ChatGroq(
    model="llama-3.3-70b-versatile",   # Groq model ID (case-sensitive)
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"), # Pulled from .env
)

#  Practical perspective model (llama-3.3-70b-versatile) 
mixtral = ChatGroq(
    model="llama-3.3-70b-versatile",   
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

#  Critic model (llama-3.3-70b-versatile)
# temperature=0.3 → low for precise, consistent gap detection.
llama3_8b_critic = ChatGroq(
    model="llama-3.3-70b-versatile",   
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

#  Synthesizer model (llama-3.3-70b-versatile) ─────────────────────────────
# temperature=0.2 → very low for stable, deterministic final answers.
gemma2_synthesizer = ChatGroq(
    model="llama-3.3-70b-versatile",  
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)


# 3. NODE FUNCTIONS
# Each function signature is:  (state: DebateState) -> dict
# The returned dict contains ONLY the fields this node wants to update.
# LangGraph shallow-merges it back into the full state.

#  3a. Technical Perspective Node (Llama 3.3 70B) 
def technical_perspective_node(state: DebateState) -> dict:
    """
    Llama 3.3 70B is instructed to view the query purely through a technical lens:
    architecture, algorithms, performance trade-offs, scalability, security.
    It is explicitly told NOT to discuss practicality or business concerns so
    the two perspectives stay maximally distinct.
    """
    print("\n[Node] 🔧 Llama 3.3 70B generating TECHNICAL perspective…")

    system_prompt = """You are a senior software architect and technical expert.
Your role is to analyze questions from a purely TECHNICAL perspective.

Focus ONLY on:
- Technical architecture and design patterns
- Algorithms, data structures, and computational complexity
- Performance, scalability, and reliability considerations
- Security vulnerabilities and technical risks
- Implementation challenges and technical trade-offs
- Code quality, testing, and maintainability

DO NOT discuss: business viability, user experience, cost, team dynamics,
or market considerations. Stay strictly technical."""

    # Build the message list that the LLM will see
    messages = [
        SystemMessage(content=system_prompt),           # Sets the persona
        HumanMessage(content=f"Analyze this from a technical perspective:\n\n{state['query']}")
    ]

    # Call Llama 3.3 70B via Groq — .invoke() is synchronous
    response = llama3_70b.invoke(messages)

    # response.content is a plain string with the assistant's reply
    technical_view = response.content

    # Append a human-readable entry to the reasoning trace
    trace_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Llama 3.3 70B Technical Perspective generated ({len(technical_view)} chars)"

    return {
        "technical_perspective": technical_view,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


#  3b. Practical Perspective Node (llama-3.3-70b-versatile) 
def practical_perspective_node(state: DebateState) -> dict:
    """
    llama-3.3-70b-versatile is instructed to view the query from a practical/business angle:
    real-world adoption, cost, team skills, user experience, timelines.
    Mirroring the instruction to avoid technical depth keeps perspectives distinct.
    """
    print("\n[Node] 💼 llama-3.3-70b-versatile generating PRACTICAL perspective…")

    system_prompt = """You are a seasoned product manager and business strategist.
Your role is to analyze questions from a purely PRACTICAL perspective.

Focus ONLY on:
- Real-world implementation challenges and organizational constraints
- Cost, budget, and resource requirements
- Team skills, training needs, and adoption curves
- User experience and customer impact
- Timeline, milestones, and delivery risks
- Business value, ROI, and market fit
- Change management and stakeholder concerns

DO NOT discuss: low-level technical implementation, algorithms, code quality,
or architectural patterns. Stay strictly practical and business-focused."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze this from a practical perspective:\n\n{state['query']}")
    ]

    # Call llama-3.3-70b-versatile via Groq
    response = mixtral.invoke(messages)
    practical_view = response.content

    trace_entry = f"[{datetime.now().strftime('%H:%M:%S')}] llama-3.3-70b-versatile Practical Perspective generated ({len(practical_view)} chars)"

    return {
        "practical_perspective": practical_view,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


#  3c. Critic Node (Llama 3.1 8B Instant) ─────
def critic_node(state: DebateState) -> dict:
    """
    Llama 3.1 8B Instant receives both perspectives and systematically identifies:
      1. Direct contradictions between the two views
      2. Weak or unsubstantiated assumptions in either view
      3. Important angles that both views missed entirely
      4. Whether the gaps are "major" (triggering another loop iteration)

    The critic must return a JSON block so we can parse has_major_gaps
    programmatically without fragile regex.

    Using the fast 8B model here keeps latency low for what is essentially
    a structured extraction task rather than a generative one.
    """
    print(f"\n[Node] 🔍 llama-3.3-70b-versatile analyzing perspectives (iteration {state['iteration'] + 1})…")

    system_prompt = """You are a rigorous debate moderator and critical thinker.
Your job is to critique TWO different perspectives on the same question.

Analyze and identify:
1. CONTRADICTIONS: Direct conflicts between the two perspectives
2. WEAK ASSUMPTIONS: Claims made without sufficient justification
3. MISSING ARGUMENTS: Important angles neither perspective addressed
4. GAPS SEVERITY: Whether the missing pieces are minor or major

You MUST end your response with a JSON block in exactly this format:
```json
{
  "has_major_gaps": true|false,
  "gap_summary": "One sentence describing the most critical gap",
  "contradiction_count": <integer>,
  "weak_assumption_count": <integer>,
  "missing_topics": ["topic1", "topic2"]
}
```
Set has_major_gaps=true only when a critical dimension is entirely absent
(e.g., security completely ignored, or feasibility not addressed at all)."""

    # Compose the full context the critic will reason over
    critic_prompt = f"""ORIGINAL QUESTION:
{state['query']}

━━━ TECHNICAL PERSPECTIVE (Llama 3.3 70B) ━━━
{state['technical_perspective']}

━━━ PRACTICAL PERSPECTIVE (llama-3.3-70b-versatile) ━━━
{state['practical_perspective']}

━━━ PREVIOUS CRITIQUE (if any) ━━━
{state.get('critique', 'None — this is the first iteration.')}

Please provide your detailed critique followed by the required JSON block."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=critic_prompt),
    ]

    # Call the fast 8B model — runs at ~750 tok/s on Groq
    response = llama3_8b_critic.invoke(messages)
    critique_text = response.content

    #  Parse the JSON block embedded in the critique ─────────────────────
    has_major_gaps = False   # Safe default if parsing fails
    gap_summary = "No major gaps detected."

    try:
        # Locate the ```json … ``` fence and extract its content
        json_start = critique_text.find("```json")
        json_end   = critique_text.find("```", json_start + 7)   # closing fence

        if json_start != -1 and json_end != -1:
            # Slice out just the JSON text (skip "```json\n" prefix)
            json_str = critique_text[json_start + 7 : json_end].strip()
            parsed   = json.loads(json_str)

            has_major_gaps = parsed.get("has_major_gaps", False)
            gap_summary    = parsed.get("gap_summary", gap_summary)
    except (json.JSONDecodeError, ValueError) as e:
        # If parsing fails we log it but don't crash; has_major_gaps stays False
        print(f"  ⚠️  Could not parse critic JSON: {e}")

    #  Snapshot the current iteration for the audit history ──────────────
    iteration_snapshot = {
        "iteration": state["iteration"] + 1,
        "timestamp": datetime.now().isoformat(),
        "technical_length": len(state["technical_perspective"]),
        "practical_length": len(state["practical_perspective"]),
        "has_major_gaps": has_major_gaps,
        "gap_summary": gap_summary,
    }

    trace_entry = (
        f"[{datetime.now().strftime('%H:%M:%S')}] llama-3.3-70b-versatile Critic: "
        f"major_gaps={has_major_gaps} | {gap_summary}"
    )

    return {
        "critique": critique_text,
        "has_major_gaps": has_major_gaps,
        "gap_summary": gap_summary,
        "iteration": state["iteration"] + 1,            # Increment the loop counter
        "iteration_history": state.get("iteration_history", []) + [iteration_snapshot],
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


#  3d. Synthesizer Node (llama-3.3-70b-versatile) ──────────
def synthesizer_node(state: DebateState) -> dict:
    """
    llama-3.3-70b-versatile (synthesizer) has access to everything: both perspectives, the critique,
    and the iteration history.  It produces:
      • A single, coherent answer that resolves contradictions
      • A confidence score (0–1) reflecting how well it resolved the debate
      • The final reasoning trace entry

    Low temperature (0.2) keeps the synthesis stable and deterministic.
    """
    print("\n[Node] ✨ llama-3.3-70b-versatile building final answer…")

    system_prompt = """You are a master synthesizer and decision-making expert.
You receive multiple expert perspectives and a critique, then produce ONE
coherent, balanced answer that:

1. Incorporates the strongest points from EACH perspective
2. Resolves identified contradictions with reasoned judgement
3. Acknowledges remaining uncertainties honestly
4. Provides actionable recommendations where appropriate

End your response with a JSON block:
```json
{
  "confidence_score": 0.0-1.0,
  "confidence_reasoning": "Why this confidence level",
  "key_synthesis_points": ["point1", "point2", "point3"]
}
```
confidence_score guide:
  0.9–1.0 → Perspectives strongly aligned; very confident synthesis
  0.7–0.9 → Minor gaps remain; good synthesis possible
  0.5–0.7 → Notable contradictions; synthesis is a compromise
  < 0.5   → Fundamental disagreements; answer is necessarily uncertain"""

    # Build context summary of all iterations if there were multiple loops
    iteration_context = ""
    if state["iteration"] > 1:
        iteration_context = f"\n\nNOTE: This required {state['iteration']} critique iterations.\n"
        for snap in state.get("iteration_history", []):
            iteration_context += f"  Iteration {snap['iteration']}: {snap['gap_summary']}\n"

    synthesis_prompt = f"""ORIGINAL QUESTION:
{state['query']}

━━━ TECHNICAL PERSPECTIVE (Llama 3.3 70B) ━━━
{state['technical_perspective']}

━━━ PRACTICAL PERSPECTIVE (llama-3.3-70b-versatile) ━━━
{state['practical_perspective']}

━━━ CRITIC'S ANALYSIS (llama-3.3-70b-versatile) ━━━
{state['critique']}
{iteration_context}

Please synthesize a final, comprehensive answer."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=synthesis_prompt),
    ]

    # Call llama-3.3-70b-versatile via Groq
    response = gemma2_synthesizer.invoke(messages)
    synthesis_text = response.content

    #  Parse confidence score from the embedded JSON ─────────────────────
    confidence_score = 0.75   # Sensible default if parsing fails

    try:
        json_start = synthesis_text.find("```json")
        json_end   = synthesis_text.find("```", json_start + 7)

        if json_start != -1 and json_end != -1:
            json_str     = synthesis_text[json_start + 7 : json_end].strip()
            parsed       = json.loads(json_str)
            confidence_score = float(parsed.get("confidence_score", 0.75))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  Could not parse synthesizer JSON: {e}")

    trace_entry = (
        f"[{datetime.now().strftime('%H:%M:%S')}] llama-3.3-70b-versatile Synthesis complete | "
        f"confidence={confidence_score:.2f} | iterations={state['iteration']}"
    )

    return {
        "final_answer": synthesis_text,
        "confidence_score": confidence_score,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


# ─────────────────
# 4. ROUTING / CONDITIONAL EDGES
# ─────────────────

def should_loop(state: DebateState) -> str:
    """
    This function is called by LangGraph after the critic node to decide
    which node to visit next.

    Returns a string key that LangGraph maps to a target node via the
    conditional_edges mapping defined below.

    Logic:
      • If the critic found major gaps AND we haven't hit the iteration cap
        → return "loop"  → graph routes back to technical_perspective
      • Otherwise → return "synthesize" → proceed to final answer
    """
    iteration      = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)   # default cap
    has_major_gaps = state.get("has_major_gaps", False)

    if has_major_gaps and iteration < max_iterations:
        print(f"\n[Router] ⚠️  Major gaps detected — looping (iteration {iteration}/{max_iterations})")
        return "loop"
    else:
        reason = "max iterations reached" if iteration >= max_iterations else "no major gaps"
        print(f"\n[Router] ✅ Proceeding to synthesis ({reason})")
        return "synthesize"


# ─────────────────
# 5. BUILD THE LANGGRAPH GRAPH
# ─────────────────

def build_debate_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Nodes are named strings; edges define execution order.
    add_conditional_edges lets a routing function dynamically pick the next node.
    """

    # StateGraph(DebateState) tells LangGraph the shape of the shared state
    graph = StateGraph(DebateState)

    #  Register nodes ──────────────────────
    graph.add_node("technical_perspective", technical_perspective_node)
    graph.add_node("practical_perspective", practical_perspective_node)
    graph.add_node("critic",               critic_node)
    graph.add_node("synthesizer",          synthesizer_node)

    #  Entry point ─────────────────────────
    # LangGraph needs to know which node to call first
    graph.set_entry_point("technical_perspective")

    #  Sequential edges (A always goes to B) ─────────────────────────────
    graph.add_edge("technical_perspective", "practical_perspective")
    graph.add_edge("practical_perspective", "critic")

    #  Conditional edge from critic ────────
    # should_loop() is called; its return value is looked up in the mapping
    # to find the next node name.
    graph.add_conditional_edges(
        "critic",           # source node
        should_loop,        # routing function
        {
            "loop":       "technical_perspective",  # re-run both perspectives
            "synthesize": "synthesizer",             # proceed to final answer
        }
    )

    #  Terminal edge ───────────────────────
    # After synthesis, the graph ends
    graph.add_edge("synthesizer", END)

    # compile() validates the graph and returns a runnable object
    return graph.compile()


# ─────────────────
# 6. RUNNER FUNCTION
# ─────────────────

def run_debate(query: str, max_iterations: int = 3) -> dict:
    """
    High-level entry point.

    Args:
        query:          The user's question or topic to debate.
        max_iterations: Maximum critique→refine loops before forcing synthesis.

    Returns:
        The final DebateState dict after all nodes have executed.
    """

    print("\n" + "═" * 60)
    print("  MULTI-OPINION DEBATE ENGINE  (Groq Edition)")
    print("  Models: llama-3.3-70b-versatile (all nodes)")
    print("═" * 60)
    print(f"  Query: {query[:80]}{'…' if len(query) > 80 else ''}")
    print(f"  Max iterations: {max_iterations}")
    print("═" * 60)

    #  Build the compiled graph ────────────
    app = build_debate_graph()

    #  Initial state ───────────────────────
    # Every field in DebateState must have a value here (or be Optional).
    initial_state: DebateState = {
        "query":                  query,
        "technical_perspective":  "",
        "practical_perspective":  "",
        "critique":               "",
        "has_major_gaps":         False,
        "gap_summary":            "",
        "final_answer":           "",
        "confidence_score":       0.0,
        "reasoning_trace":        [],
        "iteration":              0,
        "max_iterations":         max_iterations,
        "iteration_history":      [],
    }

    #  Execute the graph ───────────────────
    # .invoke() runs the graph synchronously and returns the final state
    final_state = app.invoke(initial_state)

    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# 7. PRETTY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_results(state: dict) -> dict:
    """
    Formats the final state into a readable console report and returns a
    structured dict for the Streamlit frontend (app.py) to consume directly.

    The console output is identical to the original.  The only addition is the
    return value — previously None, now a plain dict — so the frontend can call
    print_results(result) and immediately render what it gets back.

    Returns:
        dict with keys: query, technical_perspective, practical_perspective,
        critique, final_answer, confidence_score, iteration,
        reasoning_trace, iteration_history.
    """

    divider = "─" * 60

    print(f"\n{'═' * 60}")
    print("  DEBATE ENGINE — FINAL REPORT")
    print(f"{'═' * 60}")

    print(f"\n📋 QUERY:\n{state['query']}\n")

    print(divider)
    print("🔧 TECHNICAL PERSPECTIVE (gemini):")
    print(divider)
    print(state["technical_perspective"])

    print(f"\n{divider}")
    print("💼 PRACTICAL PERSPECTIVE (Groq):")
    print(divider)
    print(state["practical_perspective"])

    print(f"\n{divider}")
    print("🔍 CRITIC'S ANALYSIS:")
    print(divider)
    print(state["critique"])

    print(f"\n{divider}")
    print("✨ SYNTHESIZED FINAL ANSWER:")
    print(divider)
    print(state["final_answer"])

    print(f"\n{divider}")
    print("📊 METADATA:")
    print(divider)
    print(f"  Confidence Score : {state['confidence_score']:.2f} / 1.00")
    print(f"  Total Iterations : {state['iteration']}")
    print(f"  Trace Steps      : {len(state['reasoning_trace'])}")

    if state["iteration_history"]:
        print(f"\n  Iteration History:")
        for snap in state["iteration_history"]:
            flag = "⚠️ " if snap["has_major_gaps"] else "✅"
            print(f"    {flag} Iteration {snap['iteration']}: {snap['gap_summary']}")

    print(f"\n  Reasoning Trace:")
    for step in state["reasoning_trace"]:
        print(f"    → {step}")

    print(f"\n{'═' * 60}\n")

    # ── Structured return value consumed by app.py ────────────────────────
    return {
        "query":                 state["query"],
        "technical_perspective": state["technical_perspective"],
        "practical_perspective": state["practical_perspective"],
        "critique":              state["critique"],
        "final_answer":          state["final_answer"],
        "confidence_score":      state["confidence_score"],
        "iteration":             state["iteration"],
        "reasoning_trace":       state["reasoning_trace"],
        "iteration_history":     state.get("iteration_history", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Accept an optional query from the command line ────────────────────
    # Standalone:   python debate_engine.py "Should we adopt Rust?"
    # Frontend:     app.py imports run_debate / print_results directly;
    #               this block is never reached in that case.
    if len(sys.argv) > 1:
        QUERY = " ".join(sys.argv[1:])
    else:
        QUERY = """
        Should a mid-sized startup (50 engineers) adopt Kubernetes for container
        orchestration, or stick with managed PaaS solutions like Heroku or Railway?
        """.strip()

    # MAX_ITERATIONS can be overridden via env var:
    #   MAX_ITERATIONS=1 python debate_engine.py "your question"
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "2"))

    result = run_debate(query=QUERY, max_iterations=MAX_ITERATIONS)

    print_results(result)

    output_path = "debate_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full result saved to: {output_path}")