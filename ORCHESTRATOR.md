# Support Orchestrator Architecture

## Overview

The Support Orchestrator is a LangGraph-based multi-agent system that provides youth well-being support through behavioral analysis and gentle interventions. It employs a **three-phase architecture** (Planner-Actor-Critic) with strict ethical guardrails to ensure privacy, transparency, and age-appropriate care.

---

## Core Principles

### 🔒 Privacy & Ethics First

- **No content surveillance** - Only behavioral signals (screen time, sleep, routine)
- **No diagnosis or labels** - Strength-based language only
- **Full transparency** - Clear explanations of what is/isn't accessed
- **User consent** - Granular control over escalation paths

### 🎯 Design Philosophy

- **Restraint by default** - Choosing to do nothing (HOLD) is often the right decision
- **Least intrusive intervention** - Start gentle, escalate only when necessary
- **Contextual awareness** - Considers history, patterns, and previous outcomes
- **Themeable interactions** - Disguises self-care as engaging activities (pet care, adventures, etc.)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Behavioral Metrics                 │
│  • Screen time (Apple Screen Time / Google Family Link)     │
│  • Sleep patterns, device pickups, app usage breakdown      │
│  • Physical activity, routine consistency                    │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 1: PLANNER                            │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │ Signal Interpreter    │  →   │   Planner Agent      │    │
│  │ - Detects patterns    │      │ - Decides action     │    │
│  │ - Compares baseline   │      │ - Considers history  │    │
│  │ - Generates findings  │      │ - Prioritizes HOLD   │    │
│  └──────────────────────┘      └──────────────────────┘    │
│                                         │                    │
│                           Outputs: HOLD / NUDGE_CHILD /     │
│                                   BRIEF_PARENT / ESCALATE   │
└─────────────────────────────────────┬───────────────────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 2: ACTOR                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Hold Actor   │  │ Nudge Actor  │  │ Brief Actor  │      │
│  │ - No action  │  │ - Child      │  │ - Parent     │      │
│  │ - Document   │  │   nudges     │  │   briefing   │      │
│  │   reasoning  │  │ - Micro      │  │ - Coaching   │      │
│  └──────────────┘  │   quests     │  │   tips       │      │
│                    └──────────────┘  └──────────────┘      │
│                    ┌──────────────┐                         │
│                    │ Escalate     │                         │
│                    │ - Professional│                         │
│                    │   referral    │                         │
│                    └──────────────┘                         │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 3: CRITIC                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Safety & Policy Guardian                    │   │
│  │  - Scans for clinical/diagnostic language            │   │
│  │  - Checks consent boundaries                         │   │
│  │  - Validates age-appropriateness                     │   │
│  │  - Enforces ethical guardrails                       │   │
│  └────────────┬─────────────────────────────────────────┘   │
│               │                                              │
│       ┌───────┴───────┐                                     │
│       ▼               ▼                                     │
│   ✅ APPROVED    ❌ NEEDS REVISION                          │
│                      │                                      │
│                      └──────────► Revision Handler         │
│                                   (loops back to Planner)  │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
              OUTPUT TO USER
```

---

## Phase 1: PLANNER

The planning phase analyzes behavioral signals and decides the appropriate level of intervention.

### 1.1 Signal Interpreter Agent

**Purpose:** Detect meaningful pattern shifts in behavioral data.

**Process:**

1. Receives daily metrics (screen time, sleep, routine, etc.)
2. Compares recent data (last 3 days) vs baseline (previous period)
3. Identifies statistically significant deviations
4. Generates structured findings with evidence

**Output Structure:**

```python
Finding {
    finding_id: "late_night_screen_spike"
    title: "Increased late-night screen activity"
    evidence: {
        baseline_value: 8.0,      # minutes
        recent_value: 55.0,       # minutes
        days_affected: 3,
        metric_name: "late_night_screen_minutes"
    }
    severity: "medium" | "low" | "high"
    confidence: 0.85  # 0.0-1.0
}
```

**Key Features:**

- Uses statistical thresholds (>1.5x baseline for medium, >2.0x for high)
- Considers multiple days to avoid false positives
- No content analysis - only behavioral metrics
- Confidence scoring based on consistency

**Data Sources:**

- Apple Screen Time API
- Google Family Link
- Health/Fitness integrations
- Device usage logs

---

### 1.2 Planner Agent

**Purpose:** Decide the least intrusive helpful action based on findings.

**Decision Options:**

1. **HOLD** - Intentional non-action (most common)
2. **NUDGE_CHILD** - Gentle intervention for the youth
3. **BRIEF_PARENT** - Weekly summary and coaching tips
4. **ESCALATE** - Professional referral (rare, requires consent)

**Decision Logic:**

```
IF no significant findings:
    → HOLD

IF minor patterns (low severity, short duration):
    → HOLD (monitor longer)

IF moderate patterns (medium severity, 3+ days):
    → NUDGE_CHILD (if appropriate)
    → BRIEF_PARENT (if parent-level more suitable)

IF severe patterns + consent granted:
    → ESCALATE (requires school_opt_in or pro_opt_in)

Always prioritize: HOLD > NUDGE > BRIEF > ESCALATE
```

**Contextual Factors:**

- **Consent settings** - What escalation paths are allowed
- **History** - Previous actions and their outcomes
- **Duration** - How long pattern has persisted
- **Severity** - Magnitude of deviation from baseline

**LLM-Powered Analysis:**
The Planner uses an LLM (GPT-4o-mini by default) to evaluate:

- Complex pattern interactions
- Historical context and effectiveness
- Appropriate response level
- Risk vs benefit of intervention

**Output:**

```python
PlannerOutput {
    plan: "NUDGE_CHILD",
    rationale: "3-day pattern of late-night screen use affecting sleep",
    why_not_stronger: "No escalation consent, pattern manageable at child level",
    signal_summary: "Late-night screen +6x baseline, sleep -2.5hrs",
    confidence: 0.75
}
```

---

## Phase 2: ACTOR

Actors execute the plan decided by the Planner. Only one actor runs per cycle.

### 2.1 Hold Actor

**Purpose:** Document the decision to take no action.

**When Used:**

- No significant patterns detected
- Patterns too minor to warrant intervention
- Monitoring period not yet sufficient
- Previous action still in effect

**Output:**

```python
HoldOutput {
    decision: "HOLD",
    rationale: "Behavioral rhythms within normal range",
    next_check_recommendation: "Continue monitoring at regular intervals",
    signals_observed: "Screen time stable, sleep consistent, routine maintained"
}
```

**Philosophy:** Doing nothing is often the right answer. Over-intervention can create anxiety and erode trust.

---

### 2.2 Youth Nudge Composer Actor

**Purpose:** Create gentle, age-appropriate interventions disguised as engaging activities.

**Nudge Categories:**

- 🏃 Movement - Physical activity breaks
- 🌅 Routine reset - Sleep and wake patterns
- 💧 Hydration - Water intake reminders
- ☀️ Sunlight - Outdoor time
- 🫁 Breathing - Calm-down exercises
- 👥 Social checkin - Connection prompts
- 🎯 Focus break - Screen time breaks

**Themed Disguises:**

The system uses themes to make self-care feel like play:

| Theme            | Frame                   | Example                                                     |
| ---------------- | ----------------------- | ----------------------------------------------------------- |
| **Pet Care** 🐾  | "Your pet needs you!"   | "{Pet} wants to play - do 5 stretches together!"            |
| **Adventure** ⚔️ | "Hero's quest"          | "Every hero needs rest before the next quest!"              |
| **Garden** 🌱    | "Magical garden growth" | "Your garden grows when you water it - time for hydration!" |
| **Space** 🚀     | "Astronaut training"    | "Astronauts need sunlight exposure for health checks!"      |
| **Cozy** 🏡      | "Cozy corner care"      | "Your cozy corner needs you to recharge!"                   |

**Micro-Quests:**

Short, actionable tasks (2-10 minutes) that promote well-being:

```python
MicroQuest {
    title: "Playtime with Luna",
    steps: [
        "Stand up and stretch - Luna is copying you!",
        "Walk to a window together",
        "Look outside for 2 minutes with Luna"
    ],
    duration_minutes: 5
}
```

**Template System:**

- Pre-built templates for each category × theme combination
- Personalized with pet names, character names, garden names
- Age-appropriate language (no clinical terms)
- Always includes opt-out option

**Output:**

```python
YouthNudgeOutput {
    nudge: Nudge {
        title: "Luna Wants to Play!",
        text: "Luna is restless and needs playtime...",
        category: "movement",
        why_shown: "You've been sitting for a while",
        opt_out_hint: "You can snooze this anytime."
    },
    micro_quest: MicroQuest {...},
    driven_by_findings: ["late_night_screen_spike", "reduced_activity"]
}
```

---

### 2.3 Parent Briefing Actor

**Purpose:** Provide weekly summaries and coaching tips for caregivers.

**Components:**

1. **Weekly Summary** - Non-alarming overview of patterns
2. **Top Changes** - 2-3 notable shifts (neutral language)
3. **Suggested Actions** - Practical parent-level interventions
4. **Suggested Activities** - Fun things to do together
5. **Conversation Starter** - Open-ended question for dialogue
6. **Why Now** - Context for why briefing is being sent

**Tone Guidelines:**

- Strength-based, never diagnostic
- "We noticed..." not "Your child has..."
- Focus on opportunities, not problems
- Empower parent agency

**Example:**

```python
ParentBriefingOutput {
    weekly_summary: "This week showed shifts in evening routines...",
    top_changes: [
        "Bedtime shifting 45 mins later on average",
        "Screen time up 30% in late evening hours",
        "Morning wake times less consistent"
    ],
    suggested_actions: [
        "Consider a 15-minute wind-down routine before bed",
        "Co-create a device curfew time together"
    ],
    suggested_activities: [
        "Take an evening walk together",
        "Cook their favorite meal",
        "Do a puzzle or board game after dinner"
    ],
    conversation_starter: "What's been on your mind before bed lately?",
    why_now: "3-day pattern suggests routine adjustment could help",
    driven_by_findings: ["late_night_screen_spike", "bedtime_inconsistency"]
}
```

---

### 2.4 Escalation Gatekeeper Actor

**Purpose:** Handle professional referrals with strict consent enforcement.

**Escalation Levels:**

1. **parent_only** - Alert parent to monitor
2. **counselor** - School counselor (requires school_opt_in)
3. **professional** - Mental health professional (requires pro_opt_in)

**Consent Enforcement:**

```python
if escalation_level == "counselor" and not consent.school_opt_in:
    → BLOCKED - downgrade to parent_only

if escalation_level == "professional" and not consent.pro_opt_in:
    → BLOCKED - downgrade to counselor or parent_only
```

**Red Flags for Escalation:**

- Severe sleep disruption (>5 hours under baseline)
- Extreme social withdrawal (routine_consistency < 0.3)
- Prolonged pattern duration (>14 days)
- Multiple high-severity findings

**Output:**

```python
EscalationOutput {
    escalate: true,
    level: "counselor",
    reason: "14-day pattern of sleep disruption + withdrawal",
    recommended_next_step: "Schedule check-in with school counselor",
    requires_parent_ack: true
}
```

**Important:** This actor is rarely triggered. The system defaults to HOLD or lower intervention levels.

---

## Phase 3: CRITIC

The Safety Guardian validates all outputs before they reach users.

### 3.1 Safety & Policy Guardian Agent

**Purpose:** Enforce ethical guardrails and policy compliance.

**Validation Checks:**

1. **Language Scanning:**
   - ❌ Clinical terms: "depression", "anxiety", "ADHD", "disorder"
   - ❌ Diagnostic language: "diagnosed", "symptoms", "treatment"
   - ❌ Labeling: "troubled", "at-risk", "problematic"
   - ✅ Strength-based: "pattern shift", "rhythm change", "support"

2. **Consent Boundaries:**
   - Verify escalation matches consent settings
   - Ensure parent acknowledgment where required
   - Block unauthorized data sharing

3. **Age Appropriateness:**
   - Check language complexity
   - Verify activity safety
   - Ensure opt-out options present

4. **Transparency Requirements:**
   - All nudges must include "why_shown"
   - All briefings must include "driven_by_findings"
   - Explainability always available

**Decision Flow:**

```
IF violations found:
    approved = false
    revision_instructions = "Remove clinical terms: [list]"
    → Trigger revision loop (max 2 revisions)

IF consent exceeded:
    approved = false
    revision_instructions = "Downgrade escalation level"
    → Trigger revision loop

IF all checks pass:
    approved = true
    → Output to user
```

**Output:**

```python
SafetyGuardianOutput {
    approved: true,
    violations: [],
    revision_instructions: ""
}
```

**Revision Handling:**

If safety check fails:

1. State.needs_revision = true
2. State.revision_count += 1
3. Loop back to Planner (max 2 loops)
4. If still failing → Force downgrade to HOLD

---

## State Management

The system uses a Pydantic-based state model that flows through all agents:

```python
OrchestratorState {
    # Input
    metrics: List[DailyMetrics],
    baseline: Optional[Dict[str, float]],
    child_preferences: ChildPreferences,
    consent: ConsentSettings,
    scenario: str,

    # Memory
    history: List[HistoryEntry],

    # Phase outputs
    signal_output: Optional[SignalInterpreterOutput],
    planner_output: Optional[PlannerOutput],
    hold_output: Optional[HoldOutput],
    youth_nudge_output: Optional[YouthNudgeOutput],
    parent_briefing_output: Optional[ParentBriefingOutput],
    escalation_output: Optional[EscalationOutput],
    safety_output: Optional[SafetyGuardianOutput],

    # Control
    needs_revision: bool = False,
    revision_count: int = 0,
    max_revisions: int = 2
}
```

**State Evolution:**

1. Initial state populated with metrics, preferences, consent
2. Signal Interpreter adds `signal_output`
3. Planner adds `planner_output`
4. One Actor adds its specific output
5. Safety Guardian adds `safety_output`
6. Final state contains full decision trail

---

## Data Flow Example

### Scenario: Stressed Week Detection

**Day 1-3:** Normal patterns

```
Metrics: screen_time=150min, sleep=8.5hrs, late_night=8min
```

**Day 4-7:** Pattern shift

```
Metrics: screen_time=280min, sleep=5.5hrs, late_night=55min
```

**Flow:**

1. **Signal Interpreter:**

   ```
   Finding: "late_night_screen_spike"
   Evidence: baseline=8min, recent=55min (6.9x increase)
   Severity: HIGH, Confidence: 0.92
   Days affected: 4
   ```

2. **Planner:**

   ```
   Decision: NUDGE_CHILD
   Rationale: "4-day pattern of late-night use, sleep degraded"
   Why not BRIEF_PARENT: "Try child-level intervention first"
   Confidence: 0.78
   ```

3. **Youth Nudge Actor:**

   ```
   Nudge: "Luna's Bedtime Routine" (routine_reset category)
   Theme: pet_care
   Text: "Luna needs help winding down for sleep..."

   Micro-Quest: "Evening Wind-Down"
   Steps:
     - Dim lights with Luna
     - Do 3 slow breaths together
     - Tell Luna a quiet story
   Duration: 8 minutes

   Driven by: ["late_night_screen_spike", "sleep_degradation"]
   ```

4. **Safety Guardian:**

   ```
   Approved: true
   Violations: []
   Language check: PASSED (no clinical terms)
   Consent check: PASSED (child-level, no escalation)
   Transparency: PASSED (why_shown and driven_by_findings present)
   ```

5. **Output to User:**
   - Youth sees themed nudge with micro-quest
   - Explainability available via "Why am I seeing this?"
   - Opt-out available at any time

---

## Configuration & Personalization

### Child Preferences

```python
ChildPreferences {
    preferred_categories: ["movement", "breathing", "sunlight"],
    active_hours: (9, 21),  # When to show nudges
    theme: "pet_care",
    pet_name: "Luna",
    character_name: "Hero",
    garden_name: "Serenity Garden"
}
```

### Consent Settings

```python
ConsentSettings {
    school_opt_in: false,     # Allow school counselor escalation
    pro_opt_in: false,        # Allow professional referral
    parent_acknowledged: true # Parent aware of system
}
```

### Scenarios

Pre-built test scenarios for development:

- **normal_week** - Stable, healthy patterns
- **stressed_week** - Degraded sleep, increased screen time
- **recovery_week** - Improving from stressed state

---

## LLM Integration

### Model Configuration

```python
# Default model: GPT-4o-mini (fast, cost-effective)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,  # Balance creativity and consistency
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### Agents Using LLM

1. **Signal Interpreter** - Pattern detection and evidence gathering
2. **Planner** - Decision-making with historical context
3. **Parent Briefing** - Natural language generation for summaries
4. **Safety Guardian** - Language analysis and policy enforcement

### Fallback Behavior

If LLM fails:

- Signal Interpreter → Returns empty findings
- Planner → Defaults to HOLD
- Parent Briefing → Uses template-based output
- Safety Guardian → Blocks output (safe failure mode)

---

## Graph Execution

### Entry Point

```python
graph.set_entry_point("signal_interpreter")
```

### Execution Flow

```
START
  ↓
signal_interpreter
  ↓
planner
  ↓
[Conditional routing based on plan]
  ├→ hold_actor
  ├→ nudge_actor
  ├→ brief_actor
  └→ escalate_actor
  ↓
safety_critic
  ↓
[Conditional: approved or needs_revision?]
  ├→ revision_handler → [loops back to planner]
  └→ END
```

### Conditional Routing

**Route by Plan:**

```python
def route_by_plan(state) -> str:
    plan = state.planner_output.plan
    return {
        "HOLD": "hold",
        "NUDGE_CHILD": "nudge_child",
        "BRIEF_PARENT": "brief_parent",
        "ESCALATE": "escalate"
    }.get(plan, "hold")  # Default to hold
```

**Safety Check:**

```python
def check_safety_result(state) -> str:
    if not state.safety_output.approved and state.revision_count < 2:
        return "needs_revision"
    return "complete"
```

---

## Explainability

### "Why Am I Seeing This?"

Every output includes traceability:

```python
{
    "nudge_driven_by": ["late_night_screen_spike", "sleep_degradation"],
    "briefing_driven_by": ["routine_inconsistency"],
    "explanation": "These outputs were generated based on 2 finding(s):
                    late_night_screen_spike, sleep_degradation. Each finding
                    represents a shift in behavioral rhythm compared to baseline."
}
```

### Full Decision Trail

The UI visualizes:

- What patterns were detected (findings)
- Why the planner chose this action (rationale)
- What evidence supports it (metrics)
- How the actor executed it (output details)
- What safety checks validated (violations or approval)

---

## Ethical Safeguards Summary

| Safeguard                   | Implementation                                         |
| --------------------------- | ------------------------------------------------------ |
| **No Surveillance**         | Zero content analysis, only behavioral metrics         |
| **No Diagnosis**            | Language scanner blocks clinical/diagnostic terms      |
| **Transparency**            | All outputs include "why_shown" and driven_by_findings |
| **Consent Enforcement**     | Escalation gated by explicit opt-in settings           |
| **Restraint Default**       | HOLD is preferred, intervention requires justification |
| **Age Appropriate**         | Themed nudges disguise self-care as play               |
| **Opt-out Always**          | Every interaction includes snooze/skip option          |
| **Parent Inclusion**        | Briefings empower caregivers, never bypass             |
| **Professional Boundaries** | No therapy, coaching, or clinical advice               |
| **Revision Loop**           | Failed safety checks trigger re-generation             |

---

## Usage

### Command Line

```bash
# With LLM (requires OPENAI_API_KEY)
python app.py --use-llm

# Deterministic mode (no LLM, template-based)
python app.py
```

### Streamlit UI

```bash
streamlit run ui.py
```

Features:

- Upload sample data or use presets
- Configure preferences and consent
- Run orchestrator with visual feedback
- View agent thought processes
- Explore explainability
- See full decision trail

### Programmatic

```python
from app import build_graph, OrchestratorState, generate_metrics

# Initialize
graph = build_graph()

# Create state
state = OrchestratorState(
    metrics=generate_metrics("normal_week"),
    scenario="normal_week"
)

# Run
result = graph.invoke(state)

# Access outputs
print(result.planner_output.plan)
print(result.youth_nudge_output)
```

---

## Future Enhancements

### Potential Additions

- [ ] Multi-day memory and learning
- [ ] A/B testing of nudge effectiveness
- [ ] Customizable severity thresholds
- [ ] Integration with actual device APIs
- [ ] Parent dashboard with trends
- [ ] Youth feedback loop ("Did this help?")
- [ ] Counselor collaboration portal
- [ ] Expanded theme library

### Research Directions

- Efficacy studies of themed nudges
- Long-term pattern prediction
- Personalized baseline learning
- Cross-cultural theme adaptation
- Parent-child communication scaffolding

---

## Technical Stack

- **LangGraph** - State machine and agent orchestration
- **LangChain** - LLM integration and prompt management
- **Pydantic** - Type-safe state and output schemas
- **OpenAI GPT-4o-mini** - LLM-powered reasoning
- **Streamlit** - Web UI with explainability
- **Python 3.11+** - Core implementation

---

## Conclusion

The Support Orchestrator demonstrates how agentic AI systems can provide youth well-being support while maintaining strict ethical boundaries. By combining behavioral analysis, LLM-powered reasoning, and human-centered design, it creates a privacy-first, transparent, and age-appropriate intervention framework.

The three-phase architecture (Planner-Actor-Critic) ensures every decision is:

1. **Contextual** - Informed by patterns and history
2. **Appropriate** - Least intrusive, consent-respecting
3. **Safe** - Validated against ethical guardrails

This is not therapy, diagnosis, or surveillance—it's supportive scaffolding for healthy digital well-being habits, with transparency and parental empowerment at its core.
