"""
Support Orchestrator - Youth Well-Being App Backend Prototype
==============================================================

A LangGraph-based agentic system that coordinates subagents to:
1) Interpret behavioral metrics (NO content surveillance)
2) Generate child-facing nudges + micro-quests
3) Produce parent briefings + coaching tips
4) Conditionally escalate with strict policy guardrails

ETHICS & SAFETY:
- No diagnosis, no labels, no clinical terms
- Only behavioral signals: screen-time, sleep, routine, engagement
- Strength-based language only
- Full transparency on what is NOT accessed

How to run:
    pip install langgraph langchain langchain-core pydantic
    python app.py

Optional LLM mode:
    export OPENAI_API_KEY=your_key_here
    python app.py --use-llm
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Literal, Optional, Any
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# LLM imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables
load_dotenv()

# Initialize LLM (will be configured in main)
llm: ChatOpenAI = None

def init_llm(model: str = "gpt-4o-mini", temperature: float = 0.7):
    """Initialize the LLM with OpenAI API key from environment."""
    global llm
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key
    )
    return llm


# =============================================================================
# PYDANTIC MODELS - Strict Output Schemas
# =============================================================================

class Evidence(BaseModel):
    """Evidence supporting a finding."""
    baseline_value: float = Field(description="Baseline average value")
    recent_value: float = Field(description="Recent average value")
    days_affected: int = Field(description="Number of days showing this pattern")
    metric_name: str = Field(description="Name of the metric")


class Finding(BaseModel):
    """A behavioral finding from signal interpretation."""
    finding_id: str
    title: str
    evidence: dict[str, Any]
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)


class SignalInterpreterOutput(BaseModel):
    """Output from Signal Interpreter Agent."""
    findings: list[Finding]


class Nudge(BaseModel):
    """A gentle nudge for the youth."""
    title: str
    text: str
    category: Literal["movement", "routine_reset", "hydration", "sunlight", 
                      "breathing", "social_checkin", "focus_break"]
    why_shown: str
    opt_out_hint: str = "You can snooze this anytime."


class MicroQuest(BaseModel):
    """A small actionable quest."""
    title: str
    steps: list[str]
    duration_minutes: int = Field(ge=2, le=10)


class YouthNudgeOutput(BaseModel):
    """Output from Youth Nudge Composer Agent."""
    nudge: Nudge
    micro_quest: MicroQuest
    driven_by_findings: list[str] = Field(default_factory=list, 
                                          description="Finding IDs that drove this nudge")


class ParentBriefingOutput(BaseModel):
    """Output from Parent Briefing & Coaching Agent."""
    weekly_summary: str
    top_changes: list[str]
    suggested_actions: list[str]
    suggested_activities: list[str] = Field(
        default_factory=list,
        description="Fun, actionable activities for parent to do with child (e.g., 'Take a walk together', 'Cook their favorite meal')"
    )
    conversation_starter: str
    why_now: str
    driven_by_findings: list[str] = Field(default_factory=list)


class EscalationOutput(BaseModel):
    """Output from Escalation Gatekeeper Agent."""
    escalate: bool
    level: Literal["parent_only", "counselor", "professional"]
    reason: str
    recommended_next_step: str
    requires_parent_ack: bool


class SafetyGuardianOutput(BaseModel):
    """Output from Safety & Policy Guardian Agent (CRITIC phase)."""
    approved: bool
    violations: list[str]
    revision_instructions: str = ""
    iteration: int = Field(default=0, description="Which iteration/reiteration this critique came from")


# =============================================================================
# PLANNER PHASE - Decision Making
# =============================================================================

class PlannerOutput(BaseModel):
    """
    Output from PLANNER phase.
    Decides what should happen next: HOLD, NUDGE_CHILD, BRIEF_PARENT, or ESCALATE.
    """
    plan: Literal["HOLD", "NUDGE_CHILD", "BRIEF_PARENT", "ESCALATE"]
    rationale: str = Field(description="Short rationale referencing observed pattern changes")
    why_not_stronger: str = Field(description="Why a stronger action was not taken")
    signal_summary: str = Field(description="Brief summary of what signals were observed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this decision")


class HistoryEntry(BaseModel):
    """Record of past actions taken for memory/learning."""
    date: str
    plan_taken: Literal["HOLD", "NUDGE_CHILD", "BRIEF_PARENT", "ESCALATE"]
    findings_at_time: list[str] = Field(default_factory=list)
    outcome: Optional[str] = None  # e.g., "child_engaged", "ignored", "parent_acknowledged"


class HoldOutput(BaseModel):
    """Output when HOLD is the chosen plan - intentional non-action."""
    decision: Literal["HOLD"] = "HOLD"
    rationale: str
    next_check_recommendation: str = "Continue monitoring at regular intervals"
    signals_observed: str


class ConsentSettings(BaseModel):
    """User consent settings."""
    school_opt_in: bool = False
    pro_opt_in: bool = False
    parent_acknowledged: bool = True


class ChildPreferences(BaseModel):
    """Child's preferences for nudges."""
    preferred_categories: list[str] = Field(
        default_factory=lambda: ["movement", "breathing", "sunlight"]
    )
    active_hours: tuple[int, int] = (9, 21)
    theme: Literal["pet_care", "adventure", "garden", "space", "cozy"] = Field(
        default="adventure",
        description="Child's chosen theme for disguised self-care activities"
    )
    pet_name: str = Field(default="Buddy", description="Name of virtual pet if pet_care theme")
    character_name: str = Field(default="Hero", description="Character name for adventure/space themes")
    garden_name: str = Field(default="Garden", description="Name of garden for garden theme")


class AppCategoryUsage(BaseModel):
    """Screen time breakdown by app category (inspired by Apple Screen Time / Google Family Link)."""
    social_media_minutes: int = Field(ge=0, description="Time on social networking apps")
    games_minutes: int = Field(ge=0, description="Time on gaming apps")
    entertainment_minutes: int = Field(ge=0, description="Time on video/streaming apps")
    education_minutes: int = Field(ge=0, description="Time on learning/educational apps")
    productivity_minutes: int = Field(ge=0, description="Time on productivity/utility apps")
    other_minutes: int = Field(ge=0, description="Time on other apps")


class DailyMetrics(BaseModel):
    """
    Daily behavioral metrics for one day.
    
    Based on data points available from:
    - Apple Screen Time API
    - Google Family Link
    - Health/Fitness integrations
    """
    date: str
    
    # === SCREEN TIME METRICS (Apple Screen Time / Google Family Link) ===
    screen_time_minutes: int = Field(ge=0, description="Total screen time in minutes")
    app_usage: AppCategoryUsage = Field(default_factory=lambda: AppCategoryUsage(
        social_media_minutes=0, games_minutes=0, entertainment_minutes=0,
        education_minutes=0, productivity_minutes=0, other_minutes=0
    ), description="Breakdown by app category")
    
    # === DEVICE ENGAGEMENT PATTERNS ===
    device_pickups: int = Field(ge=0, default=0, description="Number of times device was unlocked")
    notifications_received: int = Field(ge=0, default=0, description="Total notifications received")
    first_pickup_time: str = Field(default="07:30", description="Time of first device unlock (HH:MM)")
    
    # === EVENING & SLEEP PATTERNS ===
    late_night_screen_minutes: int = Field(ge=0, description="Screen time after 9pm")
    bedtime_target_met: bool = Field(default=True, description="Whether bedtime schedule was followed")
    sleep_hours: float = Field(ge=0.0, le=24.0, description="Hours of sleep")
    
    # === ROUTINE & ENGAGEMENT ===
    routine_consistency_score: float = Field(ge=0.0, le=1.0, description="How consistent daily patterns are")
    checkin_completion: float = Field(ge=0.0, le=1.0, description="App check-in completion rate")
    
    # === PHYSICAL WELLNESS (from Health integrations) ===
    physical_activity_minutes: int = Field(ge=0, default=0, description="Minutes of physical activity")
    outdoor_time_minutes: int = Field(ge=0, default=0, description="Estimated outdoor time")


# =============================================================================
# GRAPH STATE - TypedDict-style using Pydantic
# =============================================================================

class OrchestratorState(BaseModel):
    """
    State flowing through the LangGraph.
    
    Implements three-phase architecture:
    - PLANNER: Decides what action to take
    - ACTOR: Executes the chosen action
    - CRITIC: Validates safety and ethics
    """
    # Input data
    metrics: list[DailyMetrics] = Field(default_factory=list)
    baseline: Optional[dict[str, float]] = None  # Pre-computed baseline from JSON
    child_preferences: ChildPreferences = Field(default_factory=ChildPreferences)
    consent: ConsentSettings = Field(default_factory=ConsentSettings)
    scenario: str = "normal_week"
    
    # Memory - past actions and outcomes
    history: list[HistoryEntry] = Field(default_factory=list)
    
    # PHASE 1: Planner output
    planner_output: Optional[PlannerOutput] = None
    
    # PHASE 2: Actor outputs (only one will be populated based on plan)
    signal_output: Optional[SignalInterpreterOutput] = None
    hold_output: Optional[HoldOutput] = None
    youth_nudge_output: Optional[YouthNudgeOutput] = None
    parent_briefing_output: Optional[ParentBriefingOutput] = None
    escalation_output: Optional[EscalationOutput] = None
    
    # PHASE 3: Critic output
    safety_output: Optional[SafetyGuardianOutput] = None
    safety_history: list[SafetyGuardianOutput] = Field(default_factory=list, description="History of all safety checks with iteration numbers")
    
    # Control flow
    needs_revision: bool = False
    revision_count: int = 0
    max_revisions: int = 2
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# DATA GENERATION - Simulated Metrics
# =============================================================================

# Time range options
TIME_RANGES = {
    "last_week": 7,
    "last_2_weeks": 14,
    "last_month": 30,
}


def generate_metrics(
    scenario: str = "normal_week", 
    seed: int = 42,
    time_range: str = "last_week"
) -> list[DailyMetrics]:
    """
    Generate simulated behavioral metrics for a specified time range.
    
    Args:
        scenario: Pattern type - normal_week, stressed_week, recovery_week
        seed: Random seed for reproducibility
        time_range: Duration - "last_week" (7 days), "last_2_weeks" (14 days), "last_month" (30 days)
    
    Data points inspired by:
    - Apple Screen Time (app categories, pickups, notifications, downtime)
    - Google Family Link (app limits, bedtime schedules, activity)
    - Health integrations (sleep, physical activity)
    
    Scenarios:
    - normal_week: Stable, healthy patterns
    - stressed_week: Degraded sleep, increased late-night screen, routine disruption
    - recovery_week: Improving from stressed state
    
    For longer time ranges, data shows gradual progression:
    - stressed_week: Starts normal, gradually worsens
    - recovery_week: Starts stressed, gradually improves
    - normal_week: Consistent throughout
    """
    random.seed(seed)
    metrics = []
    
    # Get number of days based on time range
    num_days = TIME_RANGES.get(time_range, 7)
    base_date = datetime(2026, 1, 22) - timedelta(days=num_days)
    
    # Day names cycle for contextual variation (weekends have different patterns)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Scenario-specific parameters
    scenario_params = {
        "normal_week": {
            # Screen time totals
            "screen_time": (150, 25),           # ~2.5 hours average
            "late_night": (8, 5),               # Minimal late night
            # App category breakdown (% of total)
            "social_pct": (0.20, 0.05),
            "games_pct": (0.25, 0.08),
            "entertainment_pct": (0.20, 0.05),
            "education_pct": (0.25, 0.08),
            "productivity_pct": (0.05, 0.02),
            # Device engagement
            "pickups": (45, 10),
            "notifications": (80, 20),
            "first_pickup": ("07:15", 20),      # (base_time, std_minutes)
            # Sleep & routine
            "sleep": (8.5, 0.4),
            "bedtime_met_prob": 0.90,
            "routine": (0.88, 0.05),
            "checkin": (0.92, 0.04),
            # Physical wellness
            "activity": (45, 15),
            "outdoor": (35, 15),
        },
        "stressed_week": {
            "screen_time": (280, 45),           # ~4.5+ hours
            "late_night": (55, 20),             # Significant late night use
            "social_pct": (0.35, 0.10),         # More social media
            "games_pct": (0.30, 0.10),          # More gaming
            "entertainment_pct": (0.25, 0.08),
            "education_pct": (0.05, 0.03),      # Less educational
            "productivity_pct": (0.03, 0.02),
            "pickups": (85, 20),                # More compulsive checking
            "notifications": (150, 40),
            "first_pickup": ("06:00", 45),      # Earlier/erratic wake times
            "sleep": (5.5, 0.8),                # Poor sleep
            "bedtime_met_prob": 0.25,           # Rarely meeting bedtime
            "routine": (0.48, 0.12),            # Disrupted routine
            "checkin": (0.35, 0.15),            # Disengaged
            "activity": (15, 10),               # Reduced activity
            "outdoor": (10, 8),
        },
        "recovery_week": {
            "screen_time": (195, 30),
            "late_night": (22, 12),
            "social_pct": (0.25, 0.08),
            "games_pct": (0.25, 0.08),
            "entertainment_pct": (0.22, 0.06),
            "education_pct": (0.18, 0.06),
            "productivity_pct": (0.05, 0.02),
            "pickups": (55, 12),
            "notifications": (95, 25),
            "first_pickup": ("07:00", 30),
            "sleep": (7.2, 0.5),
            "bedtime_met_prob": 0.65,
            "routine": (0.72, 0.08),
            "checkin": (0.68, 0.10),
            "activity": (30, 12),
            "outdoor": (25, 12),
        }
    }
    
    params = scenario_params.get(scenario, scenario_params["normal_week"])
    normal_params = scenario_params["normal_week"]
    stressed_params = scenario_params["stressed_week"]
    
    for day in range(num_days):
        current_date = base_date + timedelta(days=day)
        day_name = day_names[day % 7]  # Cycle through week days
        is_weekend = day_name in ["Sat", "Sun"]
        
        # Calculate progression factor for longer time ranges
        # This creates realistic gradual changes over time
        if num_days > 7:
            progress = day / (num_days - 1) if num_days > 1 else 0  # 0 to 1 over the period
            
            if scenario == "stressed_week":
                # Starts normal, gradually worsens toward the end
                blend = progress  # 0 = normal, 1 = stressed
            elif scenario == "recovery_week":
                # Starts stressed, gradually improves
                blend = 1 - progress  # 1 = stressed, 0 = normal
            else:
                blend = 0  # Normal throughout
            
            # Blend parameters between normal and stressed
            def blend_param(key):
                n_mean, n_std = normal_params[key]
                s_mean, s_std = stressed_params[key]
                return (n_mean + blend * (s_mean - n_mean), n_std + blend * (s_std - n_std))
            
            current_params = {
                "screen_time": blend_param("screen_time"),
                "late_night": blend_param("late_night"),
                "social_pct": blend_param("social_pct"),
                "games_pct": blend_param("games_pct"),
                "entertainment_pct": blend_param("entertainment_pct"),
                "education_pct": blend_param("education_pct"),
                "productivity_pct": blend_param("productivity_pct"),
                "pickups": blend_param("pickups"),
                "notifications": blend_param("notifications"),
                "first_pickup": params["first_pickup"],  # Keep base first pickup
                "sleep": blend_param("sleep"),
                "bedtime_met_prob": normal_params["bedtime_met_prob"] + blend * (stressed_params["bedtime_met_prob"] - normal_params["bedtime_met_prob"]),
                "routine": blend_param("routine"),
                "checkin": blend_param("checkin"),
                "activity": blend_param("activity"),
                "outdoor": blend_param("outdoor"),
            }
        else:
            current_params = params
        
        # Weekend adjustments (more screen time, later pickups, etc.)
        weekend_screen_mult = 1.3 if is_weekend else 1.0
        weekend_pickup_offset = 60 if is_weekend else 0  # Later wake on weekends
        
        # Use current_params for this day
        p = current_params if num_days > 7 else params
        
        # Generate total screen time
        total_screen = max(60, int(random.gauss(*p["screen_time"]) * weekend_screen_mult))
        
        # Generate app category breakdown
        social_pct = max(0.05, min(0.5, random.gauss(*p["social_pct"])))
        games_pct = max(0.05, min(0.5, random.gauss(*p["games_pct"])))
        entertainment_pct = max(0.05, min(0.5, random.gauss(*p["entertainment_pct"])))
        education_pct = max(0.02, min(0.4, random.gauss(*p["education_pct"])))
        productivity_pct = max(0.01, min(0.2, random.gauss(*p["productivity_pct"])))
        
        # Normalize percentages to sum to ~1
        total_pct = social_pct + games_pct + entertainment_pct + education_pct + productivity_pct
        other_pct = max(0, 1 - total_pct)
        
        app_usage = AppCategoryUsage(
            social_media_minutes=int(total_screen * social_pct / total_pct),
            games_minutes=int(total_screen * games_pct / total_pct),
            entertainment_minutes=int(total_screen * entertainment_pct / total_pct),
            education_minutes=int(total_screen * education_pct / total_pct),
            productivity_minutes=int(total_screen * productivity_pct / total_pct),
            other_minutes=int(total_screen * other_pct / total_pct) if other_pct > 0 else 0
        )
        
        # Generate first pickup time
        base_hour, base_min = map(int, p["first_pickup"][0].split(":"))
        base_minutes = base_hour * 60 + base_min + weekend_pickup_offset
        pickup_minutes = max(300, min(720, int(random.gauss(base_minutes, p["first_pickup"][1]))))  # 5am-12pm
        first_pickup = f"{pickup_minutes // 60:02d}:{pickup_minutes % 60:02d}"
        
        # Handle bedtime_met_prob which may be a tuple or float depending on blending
        bedtime_prob = p["bedtime_met_prob"]
        if isinstance(bedtime_prob, tuple):
            bedtime_prob = bedtime_prob[0]
        
        metrics.append(DailyMetrics(
            date=current_date.strftime("%Y-%m-%d"),
            # Screen time
            screen_time_minutes=total_screen,
            app_usage=app_usage,
            # Device engagement
            device_pickups=max(10, int(random.gauss(*p["pickups"]))),
            notifications_received=max(20, int(random.gauss(*p["notifications"]))),
            first_pickup_time=first_pickup,
            # Evening & sleep
            late_night_screen_minutes=max(0, int(random.gauss(*p["late_night"]))),
            bedtime_target_met=random.random() < bedtime_prob,
            sleep_hours=max(3.0, min(12.0, round(random.gauss(*p["sleep"]), 1))),
            # Routine & engagement
            routine_consistency_score=max(0.0, min(1.0, round(random.gauss(*p["routine"]), 2))),
            checkin_completion=max(0.0, min(1.0, round(random.gauss(*p["checkin"]), 2))),
            # Physical wellness
            physical_activity_minutes=max(0, int(random.gauss(*p["activity"]))),
            outdoor_time_minutes=max(0, int(random.gauss(*p["outdoor"])))
        ))
    
    return metrics


# =============================================================================
# SUBAGENT 1: Signal Interpreter (LLM-POWERED)
# =============================================================================

SIGNAL_INTERPRETER_SYSTEM_PROMPT = """You are a behavioral rhythm analyst for a youth well-being app.
Your job is to analyze a child's weekly behavioral metrics and identify any significant pattern shifts that might indicate they need support.

DATA SOURCES (inspired by Apple Screen Time & Google Family Link):
- Screen time totals and app category breakdowns (social, games, education, etc.)
- Device pickups (unlock frequency), notifications
- First pickup time (morning routine indicator)
- Late night screen usage and bedtime compliance
- Sleep hours and routine consistency
- Physical activity and outdoor time
- App check-in engagement

CRITICAL RULES:
1. NEVER use clinical terms like "depression", "anxiety", "disorder", etc.
2. Use ONLY strength-based, neutral language
3. Focus on behavioral RHYTHMS and PATTERNS, not mental states
4. Valid finding IDs:
   - sleep_rhythm_shift: Sleep hours dropped or became irregular
   - evening_energy_pattern: Late night screen time increased, bedtime issues
   - routine_flexibility: Daily routine consistency decreased
   - social_stamina_shift: Check-in engagement dropped
   - focus_rhythm_change: Overall screen time increased, or app mix shifted to more passive content
   - device_dependency_pattern: Device pickups increased significantly
   - activity_rhythm_shift: Physical activity or outdoor time decreased
   - morning_pattern_shift: First pickup times became earlier/irregular

SEVERITY THRESHOLDS:
- low: Minor shift, may be normal variation (10-20% change)
- medium: Notable shift worth monitoring (20-40% change)
- high: Significant persistent shift requiring attention (40%+ change or multiple concerning patterns)"""

SIGNAL_INTERPRETER_USER_PROMPT = """Analyze this week's behavioral metrics and identify pattern shifts.

EXPECTED HEALTHY BASELINE (typical for this age group):
- Sleep: 8-9 hours/night
- Screen time: 2-3 hours/day (120-180 min)
- Late night screen: <15 minutes after 9pm
- Device pickups: 40-60 per day
- First pickup: After 7:00am
- Bedtime compliance: 85%+
- Routine consistency: 80%+
- Check-in completion: 85%+
- Physical activity: 30-60 min/day
- Outdoor time: 30+ min/day
- App mix: Balanced (education ~25%, entertainment ~25%, social ~20%, games ~25%)

THIS WEEK'S DATA (7 days):
{daily_breakdown}

WEEKLY SUMMARY:
- Average sleep: {avg_sleep:.1f} hours/night
- Average screen time: {avg_screen:.0f} minutes/day
- Average late night screen: {avg_late_night:.0f} minutes
- Average device pickups: {avg_pickups:.0f}/day
- Bedtime compliance: {bedtime_compliance:.0%}
- Average routine score: {avg_routine:.0%}
- Average check-in rate: {avg_checkin:.0%}
- Average physical activity: {avg_activity:.0f} min/day
- Average outdoor time: {avg_outdoor:.0f} min/day

APP USAGE BREAKDOWN (weekly totals):
- Social media: {total_social:.0f} min ({social_pct:.0%})
- Games: {total_games:.0f} min ({games_pct:.0%})
- Entertainment: {total_entertainment:.0f} min ({entertainment_pct:.0%})
- Education: {total_education:.0f} min ({education_pct:.0%})
- Productivity: {total_productivity:.0f} min ({productivity_pct:.0%})

Analyze the data and return findings. If patterns look healthy, return an empty findings list.

{format_instructions}"""


def signal_interpreter_agent(state: OrchestratorState) -> dict:
    """
    LLM-powered signal interpreter that converts raw metrics into behavioral findings.
    Uses GPT to analyze patterns and generate contextual insights.
    """
    metrics = state.metrics
    
    if not metrics:
        return {"signal_output": SignalInterpreterOutput(findings=[])}
    
    # Use all metrics as the week's data (now exactly 7 days)
    recent_metrics = metrics
    
    # Calculate weekly averages
    avg_sleep = sum(m.sleep_hours for m in recent_metrics) / len(recent_metrics)
    avg_late_night = sum(m.late_night_screen_minutes for m in recent_metrics) / len(recent_metrics)
    avg_routine = sum(m.routine_consistency_score for m in recent_metrics) / len(recent_metrics)
    avg_checkin = sum(m.checkin_completion for m in recent_metrics) / len(recent_metrics)
    avg_screen = sum(m.screen_time_minutes for m in recent_metrics) / len(recent_metrics)
    avg_pickups = sum(m.device_pickups for m in recent_metrics) / len(recent_metrics)
    avg_activity = sum(m.physical_activity_minutes for m in recent_metrics) / len(recent_metrics)
    avg_outdoor = sum(m.outdoor_time_minutes for m in recent_metrics) / len(recent_metrics)
    bedtime_compliance = sum(1 for m in recent_metrics if m.bedtime_target_met) / len(recent_metrics)
    
    # Calculate app category totals and percentages
    total_social = sum(m.app_usage.social_media_minutes for m in recent_metrics)
    total_games = sum(m.app_usage.games_minutes for m in recent_metrics)
    total_entertainment = sum(m.app_usage.entertainment_minutes for m in recent_metrics)
    total_education = sum(m.app_usage.education_minutes for m in recent_metrics)
    total_productivity = sum(m.app_usage.productivity_minutes for m in recent_metrics)
    total_screen_week = sum(m.screen_time_minutes for m in recent_metrics)
    
    social_pct = total_social / total_screen_week if total_screen_week > 0 else 0
    games_pct = total_games / total_screen_week if total_screen_week > 0 else 0
    entertainment_pct = total_entertainment / total_screen_week if total_screen_week > 0 else 0
    education_pct = total_education / total_screen_week if total_screen_week > 0 else 0
    productivity_pct = total_productivity / total_screen_week if total_screen_week > 0 else 0
    
    # Format daily breakdown with all the new data points
    daily_breakdown = "\n".join([
        f"  {m.date}: "
        f"sleep={m.sleep_hours}h, "
        f"screen={m.screen_time_minutes}min "
        f"(social:{m.app_usage.social_media_minutes}, games:{m.app_usage.games_minutes}, "
        f"edu:{m.app_usage.education_minutes}, entertainment:{m.app_usage.entertainment_minutes}), "
        f"late_night={m.late_night_screen_minutes}min, "
        f"bedtime_met={'✓' if m.bedtime_target_met else '✗'}, "
        f"pickups={m.device_pickups}, "
        f"first_pickup={m.first_pickup_time}, "
        f"routine={m.routine_consistency_score:.0%}, "
        f"checkin={m.checkin_completion:.0%}, "
        f"activity={m.physical_activity_minutes}min, "
        f"outdoor={m.outdoor_time_minutes}min"
        for m in recent_metrics
    ])
    
    # Create parser and prompt
    parser = PydanticOutputParser(pydantic_object=SignalInterpreterOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SIGNAL_INTERPRETER_SYSTEM_PROMPT),
        ("human", SIGNAL_INTERPRETER_USER_PROMPT)
    ])
    
    # Invoke LLM
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "daily_breakdown": daily_breakdown,
            "avg_sleep": avg_sleep,
            "avg_screen": avg_screen,
            "avg_late_night": avg_late_night,
            "avg_pickups": avg_pickups,
            "bedtime_compliance": bedtime_compliance,
            "avg_routine": avg_routine,
            "avg_checkin": avg_checkin,
            "avg_activity": avg_activity,
            "avg_outdoor": avg_outdoor,
            "total_social": total_social,
            "total_games": total_games,
            "total_entertainment": total_entertainment,
            "total_education": total_education,
            "total_productivity": total_productivity,
            "social_pct": social_pct,
            "games_pct": games_pct,
            "entertainment_pct": entertainment_pct,
            "education_pct": education_pct,
            "productivity_pct": productivity_pct,
            "format_instructions": parser.get_format_instructions()
        })
        return {"signal_output": result}
    except Exception as e:
        print(f"  ⚠️ LLM error in signal interpreter: {e}")
        # Fallback to empty findings on error
        return {"signal_output": SignalInterpreterOutput(findings=[])}


# =============================================================================
# PHASE 1: PLANNER - Decision Making Agent (LLM-POWERED)
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are the PLANNER agent for a youth well-being support system.
Your job is to decide what action to take based on behavioral findings.

CORE PRINCIPLE: Match your response to the severity of the situation.
Use a balanced, graduated approach based on what the data tells you.

DECISION GUIDE (choose exactly ONE based on findings):

1. HOLD - Intentionally take no action. Choose this when:
   - No findings detected OR all findings are within healthy baseline
   - Sleep is good (7.5+ hours), routine is consistent (80%+)
   - Screen time is moderate, no late-night usage concerns
   - Everything looks stable and healthy - this is the IDEAL state
   - USE THIS for normal, healthy weeks with no concerns
   
2. NUDGE_CHILD - Gentle, optional youth-facing support. Choose this when:
   - LOW severity findings detected
   - Patterns show IMPROVEMENT or early signs of positive change
   - Recovery patterns visible (improving from a previous rough period)
   - Minor slips that a friendly reminder could help
   - USE THIS for recovery weeks or minor pattern shifts
   
3. BRIEF_PARENT - Calm parent guidance and coaching. Choose this when:
   - MEDIUM or HIGH severity findings detected
   - Sleep is noticeably reduced (below 7 hours consistently)
   - Screen time or late-night usage is elevated
   - Multiple concerning patterns present together
   - Routine consistency is low (below 60%)
   - USE THIS for tough/stressed weeks with clear concerns
   
4. ESCALATE - Structured observation summary. Choose this when:
   - EXTREME patterns: very low sleep (<5 hours), very high screen time
   - HIGH severity patterns persist for 5+ days
   - Multiple high-severity findings together
   - AND parent has acknowledged consent
   - This is RARE - only for sustained critical situations

SEVERITY MATCHING:
- No findings / healthy baselines → HOLD
- Low severity / improving patterns → NUDGE_CHILD  
- Medium-High severity / concerning patterns → BRIEF_PARENT
- Extreme / persistent crisis → ESCALATE

CRITICAL RULES:
- NEVER use clinical terms (depression, anxiety, disorder, etc.)
- Match your response to the severity level of findings
- Trust the data: healthy data = HOLD, concerning data = act appropriately
- Always explain why you DIDN'T choose a stronger action"""

PLANNER_USER_PROMPT = """Based on these findings, decide the best course of action.

DETECTED FINDINGS:
{findings_summary}

CONSENT SETTINGS:
- School opt-in: {school_opt_in}
- Professional opt-in: {pro_opt_in}
- Parent acknowledged: {parent_acknowledged}

RECENT HISTORY:
{history_summary}

Analyze carefully and choose the LEAST intrusive helpful action.

{format_instructions}"""


def planner_agent(state: OrchestratorState) -> dict:
    """
    LLM-powered planner that decides what action to take.
    Prioritizes restraint and least-intrusive interventions.
    """
    findings = state.signal_output.findings if state.signal_output else []
    consent = state.consent
    history = state.history
    
    # Format findings for prompt
    if not findings:
        findings_summary = "No significant pattern shifts detected."
    else:
        findings_summary = "\n".join([
            f"- {f.title} (severity: {f.severity}, confidence: {f.confidence:.0%}, "
            f"days affected: {f.evidence.get('days_affected', 'N/A')})"
            for f in findings
        ])
    
    # Format history
    if history:
        history_summary = "\n".join([
            f"- {h.date}: {h.plan_taken} (outcome: {h.outcome or 'unknown'})"
            for h in history[-5:]
        ])
    else:
        history_summary = "No previous actions recorded."
    
    # Create parser and prompt
    parser = PydanticOutputParser(pydantic_object=PlannerOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", PLANNER_USER_PROMPT)
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "findings_summary": findings_summary,
            "school_opt_in": consent.school_opt_in,
            "pro_opt_in": consent.pro_opt_in,
            "parent_acknowledged": consent.parent_acknowledged,
            "history_summary": history_summary,
            "format_instructions": parser.get_format_instructions()
        })
        return {"planner_output": result}
    except Exception as e:
        print(f"  ⚠️ LLM error in planner: {e}")
        # Fallback to HOLD on error
        return {"planner_output": PlannerOutput(
            plan="HOLD",
            rationale="Unable to analyze patterns. Defaulting to restraint.",
            why_not_stronger="Error in analysis - choosing safe default.",
            signal_summary="Analysis unavailable.",
            confidence=0.5
        )}


def hold_agent(state: OrchestratorState) -> dict:
    """
    ACTOR for HOLD plan: Intentionally takes no action.
    This is a valid and often preferred decision.
    """
    planner = state.planner_output
    findings = state.signal_output.findings if state.signal_output else []
    
    hold_output = HoldOutput(
        decision="HOLD",
        rationale=planner.rationale if planner else "No significant patterns detected.",
        next_check_recommendation="Continue monitoring at regular intervals. No immediate action required.",
        signals_observed=planner.signal_summary if planner else "Behavioral rhythms within normal range."
    )
    
    return {"hold_output": hold_output}


# =============================================================================
# PHASE 2: ACTOR - Youth Nudge Composer (DETERMINISTIC with templates)
# =============================================================================

# =============================================================================
# THEMED NUDGE TEMPLATES - Disguised self-care activities
# =============================================================================

# Theme definitions with personality
THEME_CONTEXT = {
    "pet_care": {
        "description": "Caring for a virtual pet companion",
        "frame": "Your pet needs you to model healthy behaviors - pets learn from their humans!",
        "emoji": "🐾"
    },
    "adventure": {
        "description": "Hero's journey with quests and missions",
        "frame": "Every hero needs to maintain their strength and abilities!",
        "emoji": "⚔️"
    },
    "garden": {
        "description": "Growing and nurturing a magical garden",
        "frame": "Your garden grows when you take care of yourself too!",
        "emoji": "🌱"
    },
    "space": {
        "description": "Space exploration and astronaut training",
        "frame": "Astronauts must stay in peak condition for their missions!",
        "emoji": "🚀"
    },
    "cozy": {
        "description": "Cozy cottage vibes with gentle self-care",
        "frame": "Your cozy corner is ready for a moment of calm.",
        "emoji": "🏡"
    }
}

# Themed nudge templates - each activity is "disguised" based on theme
THEMED_NUDGE_TEMPLATES = {
    "pet_care": {
        "movement": {
            "title": "{pet_name} Wants to Play!",
            "text": "{pet_name} is getting restless and needs some playtime! When you move around, {pet_name} gets happy and earns energy points.",
            "steps": [
                "Stand up and pretend to throw a ball for {pet_name}",
                "Do 5 big stretches - {pet_name} is copying you!",
                "Walk to a window together and look outside"
            ],
            "quest_title": "Playtime with {pet_name}"
        },
        "breathing": {
            "title": "{pet_name} Needs Calming",
            "text": "{pet_name} seems a little anxious. Pets sync with their human's breathing - can you show {pet_name} how to breathe slowly?",
            "steps": [
                "Sit with {pet_name} and breathe in for 4 counts",
                "Hold gently while {pet_name} watches - 4 counts",
                "Breathe out slowly for 6 counts - {pet_name} is copying!",
                "Repeat 3 more times until {pet_name} is calm"
            ],
            "quest_title": "Calm {pet_name} Down"
        },
        "sunlight": {
            "title": "{pet_name}'s Sunbathing Time",
            "text": "{pet_name} loves sunlight and wants you to join! Pets need sunshine for their mood, and so do their humans.",
            "steps": [
                "Find a sunny spot with {pet_name}",
                "Sit in the warmth for 2 minutes together",
                "Notice how {pet_name}'s mood improves!"
            ],
            "quest_title": "Sunny Moment with {pet_name}"
        },
        "hydration": {
            "title": "{pet_name}'s Water Bowl is Empty!",
            "text": "{pet_name} needs fresh water - and while you're at it, you should drink some too! Hydrated humans make the best pet parents.",
            "steps": [
                "Fill {pet_name}'s virtual water bowl",
                "Get yourself a glass of water too",
                "Drink together - {pet_name} is copying you!"
            ],
            "quest_title": "Hydration Station"
        },
        "routine_reset": {
            "title": "{pet_name} Needs a Schedule",
            "text": "Pets thrive on routine! Help {pet_name} by setting up a consistent schedule - you'll both feel better with some structure.",
            "steps": [
                "Pick one daily activity to do at the same time",
                "Set a reminder - {pet_name} will remind you too!",
                "Imagine doing it successfully with {pet_name}"
            ],
            "quest_title": "Routine Training with {pet_name}"
        },
        "social_checkin": {
            "title": "{pet_name} Misses Your Friends!",
            "text": "{pet_name} noticed you haven't connected with friends lately. Pets love when their humans are happy and social!",
            "steps": [
                "Think of one friend {pet_name} would like to meet",
                "Send them a quick message",
                "{pet_name} wags happily at the connection!"
            ],
            "quest_title": "Social Time for {pet_name}"
        },
        "focus_break": {
            "title": "{pet_name} Wants Attention!",
            "text": "{pet_name} has been waiting patiently while you focused. Time for a quick pet break - look away from the screen!",
            "steps": [
                "Look at {pet_name} for 20 seconds (away from screen)",
                "Give {pet_name} some virtual pets",
                "Blink 10 times slowly - {pet_name} thinks it's a game!"
            ],
            "quest_title": "{pet_name}'s Attention Quest"
        }
    },
    "adventure": {
        "movement": {
            "title": "Training Grounds Await, {character_name}!",
            "text": "A true hero never neglects their physical training! Your strength stats need a boost before the next quest.",
            "steps": [
                "Warrior stance! Stand and shake out your limbs",
                "5 shoulder rolls to loosen your armor",
                "Scout the area from the nearest window"
            ],
            "quest_title": "Hero Training Session"
        },
        "breathing": {
            "title": "Meditation Chamber, {character_name}",
            "text": "The greatest warriors master their inner calm. Your focus meter is low - time for ancient breathing techniques!",
            "steps": [
                "Enter meditation stance and breathe in (4 counts)",
                "Hold your power (4 counts)",
                "Release slowly, channeling calm (6 counts)",
                "Repeat until your focus meter is full"
            ],
            "quest_title": "Warrior's Meditation"
        },
        "sunlight": {
            "title": "Solar Energy Required, {character_name}!",
            "text": "Your power source is running low! Heroes recharge their abilities through natural light. Time to absorb some solar energy!",
            "steps": [
                "Find the nearest light source (window or outside)",
                "Absorb solar energy for 2 minutes",
                "Feel your stats regenerating!"
            ],
            "quest_title": "Solar Recharge Mission"
        },
        "hydration": {
            "title": "Health Potion Needed, {character_name}!",
            "text": "Your HP is dropping! Every hero knows that hydration potions are essential for peak performance in battle.",
            "steps": [
                "Locate a hydration potion (water)",
                "Drink the potion slowly for maximum effect",
                "HP restored! Prepare for adventure!"
            ],
            "quest_title": "Potion Quest"
        },
        "routine_reset": {
            "title": "Base Camp Organization, {character_name}",
            "text": "A disorganized hero falls in battle! Establish your daily rituals to maintain peak readiness.",
            "steps": [
                "Choose one daily ritual to anchor your schedule",
                "Set an alarm - your quest reminder",
                "Visualize yourself completing it victoriously"
            ],
            "quest_title": "Establish Base Camp Routine"
        },
        "social_checkin": {
            "title": "Alliance Check-In, {character_name}!",
            "text": "Even the mightiest heroes need allies! Your party members haven't heard from you - time to strengthen your bonds.",
            "steps": [
                "Think of an ally who'd fight alongside you",
                "Send them a message from the battlefield",
                "Alliance bond strengthened!"
            ],
            "quest_title": "Strengthen Your Alliance"
        },
        "focus_break": {
            "title": "Scout Break, {character_name}!",
            "text": "Your battle focus has been intense! Even veteran warriors need to rest their eyes and survey distant lands.",
            "steps": [
                "Look away from your battle station (screen)",
                "Scout the horizon (look at something far)",
                "Blink 10 times - warrior eye exercises!"
            ],
            "quest_title": "Reconnaissance Break"
        }
    },
    "garden": {
        "movement": {
            "title": "Your Garden Needs Tending!",
            "text": "Your magical garden grows when you move! The flowers are wilting slightly - time to dance among them and bring them back to life.",
            "steps": [
                "Stand up and stretch like a flower reaching for the sun",
                "Sway gently like plants in the breeze",
                "Walk to a window and imagine your garden growing"
            ],
            "quest_title": "Garden Dance Ritual"
        },
        "breathing": {
            "title": "Gentle Breeze for Your Garden",
            "text": "Your garden needs a gentle breeze to help the seeds spread. Your calm breath creates the perfect wind for growth.",
            "steps": [
                "Breathe in slowly, gathering air (4 counts)",
                "Hold the breath like morning dew (4 counts)",
                "Release a gentle breeze (6 counts)",
                "Watch your garden flourish with each breath"
            ],
            "quest_title": "Breathing Breeze Ritual"
        },
        "sunlight": {
            "title": "Photosynthesis Time!",
            "text": "Just like your plants, you need sunlight to thrive! When you absorb light, your garden glows brighter too.",
            "steps": [
                "Find a sunny spot - your garden's favorite!",
                "Stand in the light for 2 minutes",
                "Feel yourself and your garden photosynthesizing"
            ],
            "quest_title": "Sunlight Harvest"
        },
        "hydration": {
            "title": "Watering Time!",
            "text": "Your garden is thirsty, and so are you! Gardens grow best when their gardener is well-watered too.",
            "steps": [
                "Get a glass of water - garden's mirror magic",
                "Drink slowly while imagining watering your plants",
                "Your garden drinks when you drink!"
            ],
            "quest_title": "Hydration Ritual"
        },
        "routine_reset": {
            "title": "Planting Schedule",
            "text": "Gardens thrive on consistent care! Set up a gentle routine so your garden knows when to expect you.",
            "steps": [
                "Choose one time to visit your garden daily",
                "Set a reminder - your garden will be waiting",
                "Imagine the blooms that will greet you"
            ],
            "quest_title": "Garden Schedule Ritual"
        },
        "social_checkin": {
            "title": "Share Your Harvest!",
            "text": "The best gardens share their bounty! Your flowers want you to connect with someone and spread some joy.",
            "steps": [
                "Think of someone who'd love a virtual flower",
                "Send them a message with good vibes",
                "Your garden blooms from the kindness!"
            ],
            "quest_title": "Sharing the Harvest"
        },
        "focus_break": {
            "title": "Garden Gazing Break",
            "text": "Your eyes have been focused too long! Look away and imagine wandering through your peaceful garden.",
            "steps": [
                "Look away from the screen at something green",
                "Imagine walking through garden paths",
                "Blink 10 times like butterfly wings"
            ],
            "quest_title": "Garden Visualization"
        }
    },
    "space": {
        "movement": {
            "title": "Astronaut Training Required, {character_name}!",
            "text": "Mission Control reports your physical readiness is declining! All astronauts must maintain peak fitness for zero-gravity operations.",
            "steps": [
                "Float up from your station (stand up slowly)",
                "Zero-gravity stretches - 5 arm circles each way",
                "Walk to viewport (window) for Earth observation"
            ],
            "quest_title": "Space Fitness Protocol"
        },
        "breathing": {
            "title": "Oxygen Regulation Check, {character_name}",
            "text": "Your suit's O2 efficiency is dropping! Astronauts must practice controlled breathing to optimize oxygen usage in space.",
            "steps": [
                "Inhale: Filling oxygen tanks (4 counts)",
                "Hold: Processing in the system (4 counts)",
                "Exhale: CO2 release (6 counts)",
                "Repeat until O2 levels are optimal"
            ],
            "quest_title": "O2 Optimization Protocol"
        },
        "sunlight": {
            "title": "Solar Panel Recharge, {character_name}!",
            "text": "Your personal energy cells are depleting! Like the space station, you need to orient toward the sun to recharge.",
            "steps": [
                "Locate the nearest solar source (window)",
                "Position yourself for maximum absorption",
                "Recharge for 2 minutes - energy cells filling!"
            ],
            "quest_title": "Solar Recharge Sequence"
        },
        "hydration": {
            "title": "Hydration System Alert, {character_name}!",
            "text": "Warning: Astronaut hydration below optimal levels! In space, staying hydrated is critical for cognitive function.",
            "steps": [
                "Access water supply module",
                "Consume hydration packet slowly",
                "Hydration levels restored to optimal!"
            ],
            "quest_title": "Hydration Protocol"
        },
        "routine_reset": {
            "title": "Mission Schedule Update, {character_name}",
            "text": "ISS operations require strict schedules! Astronauts who maintain routines perform better on long missions.",
            "steps": [
                "Select one daily mission objective",
                "Log it in your mission schedule",
                "Visualize successful mission completion"
            ],
            "quest_title": "Mission Planning Protocol"
        },
        "social_checkin": {
            "title": "Earth Communications, {character_name}!",
            "text": "Mission Control notices you haven't contacted home base recently. Even astronauts need Earth connections!",
            "steps": [
                "Think of someone at ground control (a friend)",
                "Transmit a brief message to Earth",
                "Connection established! Morale boosted!"
            ],
            "quest_title": "Ground Control Check-In"
        },
        "focus_break": {
            "title": "Viewport Break, {character_name}!",
            "text": "You've been at your station too long! All astronauts must take breaks to observe Earth and rest their eyes.",
            "steps": [
                "Look away from your control panel (screen)",
                "Gaze at the 'viewport' - look far away",
                "Blink 10 times - eye moisture protocol!"
            ],
            "quest_title": "Earth Observation Break"
        }
    },
    "cozy": {
        "movement": {
            "title": "Cozy Stretch Time",
            "text": "Your cozy corner is calling for a gentle stretch! Sometimes the coziest thing is to move a little and settle back in.",
            "steps": [
                "Stand up slowly like getting out of a warm blanket",
                "Stretch gently in all directions",
                "Walk to fetch something cozy (a blanket, drink)"
            ],
            "quest_title": "Gentle Cozy Stretch"
        },
        "breathing": {
            "title": "Peaceful Breathing Corner",
            "text": "Find your coziest spot and take a moment for some peaceful breaths. Your calm space is ready for you.",
            "steps": [
                "Snuggle into a comfortable position",
                "Breathe in warmth and comfort (4 counts)",
                "Hold the coziness (4 counts)",
                "Exhale any stress (6 counts), repeat 3 times"
            ],
            "quest_title": "Cozy Breathing Ritual"
        },
        "sunlight": {
            "title": "Sunny Cozy Spot",
            "text": "There's a warm patch of sunlight calling your name. The coziest spots are often where the light falls.",
            "steps": [
                "Find where the sunlight lands in your space",
                "Sit or stand in the warmth for a moment",
                "Let the gentle light make you feel cozy"
            ],
            "quest_title": "Sunbeam Cozy Time"
        },
        "hydration": {
            "title": "Warm Drink Time",
            "text": "Nothing says cozy like a nice drink! Whether it's water, tea, or anything else - hydration is self-care.",
            "steps": [
                "Prepare your favorite cozy drink",
                "Hold it in your hands and feel the warmth",
                "Sip slowly and enjoy the moment"
            ],
            "quest_title": "Cozy Drink Ritual"
        },
        "routine_reset": {
            "title": "Cozy Routine Moment",
            "text": "Routines can be cozy anchors in your day. Let's set up one small comfort to look forward to tomorrow.",
            "steps": [
                "Pick one cozy thing to do at the same time daily",
                "Set a gentle reminder for yourself",
                "Imagine the comfort waiting for you"
            ],
            "quest_title": "Comfort Routine Setup"
        },
        "social_checkin": {
            "title": "Cozy Connection",
            "text": "Sharing cozy moments with others makes them even better. Time to send some warmth to someone you care about.",
            "steps": [
                "Think of someone who'd appreciate a cozy message",
                "Send them something warm and friendly",
                "Feel the cozy glow of connection"
            ],
            "quest_title": "Sharing Warmth"
        },
        "focus_break": {
            "title": "Cozy Eye Rest",
            "text": "Your eyes deserve a cozy break too. Look away from the screen and let them rest on something soft and far away.",
            "steps": [
                "Look away from the screen at something peaceful",
                "Let your gaze soften on something far away",
                "Blink slowly 10 times, nice and gentle"
            ],
            "quest_title": "Peaceful Eye Rest"
        }
    }
}

# Fallback non-themed templates (kept for compatibility)
NUDGE_TEMPLATES = {
    "movement": [
        {
            "title": "Quick Energy Boost",
            "text": "Your body's been still for a bit. How about a 2-minute stretch or a quick walk to your favorite window?",
            "steps": ["Stand up and shake out your arms", "Do 5 big shoulder rolls", "Take 3 deep breaths by a window"]
        },
        {
            "title": "Dance Break",
            "text": "Sometimes your energy just needs a shake-up! One song, any moves you like.",
            "steps": ["Pick your favorite upbeat song", "Move however feels good for the whole song", "Take a bow when done!"]
        }
    ],
    "breathing": [
        {
            "title": "Calm Reset",
            "text": "Taking a moment to breathe can help reset your energy. Just 4 breaths, nice and slow.",
            "steps": ["Breathe in for 4 counts", "Hold gently for 4 counts", "Breathe out slowly for 6 counts", "Repeat 3 more times"]
        }
    ],
    "sunlight": [
        {
            "title": "Sunshine Moment",
            "text": "Natural light can help your body's natural rhythm. Even a few minutes helps!",
            "steps": ["Find the nearest window or step outside", "Let the light reach your face for 2 minutes", "Notice one thing you see outside"]
        }
    ],
    "hydration": [
        {
            "title": "Hydration Check",
            "text": "Your brain works better when hydrated! Time for a quick water break.",
            "steps": ["Get a glass of water", "Drink it slowly, noticing how it feels", "Refill for later"]
        }
    ],
    "routine_reset": [
        {
            "title": "Routine Anchor",
            "text": "Routines can feel like a cozy anchor. Let's set up one small thing for tomorrow.",
            "steps": ["Pick one thing you want to do at the same time tomorrow", "Set a gentle reminder", "Imagine yourself doing it successfully"]
        }
    ],
    "social_checkin": [
        {
            "title": "Connection Moment",
            "text": "Reaching out to someone can boost your social stamina. Just one message or call!",
            "steps": ["Think of one person you'd like to hear from", "Send them a simple 'hey, thinking of you'", "No pressure on the reply timing"]
        }
    ],
    "focus_break": [
        {
            "title": "Focus Refresh",
            "text": "Your focus has been working hard! A short break can help it bounce back stronger.",
            "steps": ["Look away from screens for 20 seconds", "Focus on something far away", "Blink 10 times slowly"]
        }
    ]
}

def personalize_text(text: str, preferences: ChildPreferences) -> str:
    """Replace placeholders with personalized values."""
    return (text
            .replace("{pet_name}", preferences.pet_name)
            .replace("{character_name}", preferences.character_name)
            .replace("{garden_name}", preferences.garden_name))


# =============================================================================
# PHASE 2: ACTOR - Youth Nudge Composer (LLM-POWERED)
# =============================================================================

YOUTH_NUDGE_SYSTEM_PROMPT = """You are the Youth Nudge Composer for a well-being app.
Your job is to create FUN, THEMED, DISGUISED self-care activities for young people.

THE KEY CONCEPT: Self-care activities should be DISGUISED as fun theme-appropriate activities.
The child should feel like they're playing a game or caring for something, not doing "wellness homework".

AVAILABLE THEMES AND HOW TO USE THEM:
1. pet_care - Frame activities as caring for a virtual pet named {pet_name}
   - "Your pet needs you to..." / "{pet_name} wants to play..."
   - Pet mimics user behavior, learns from them
   
2. adventure - Frame activities as hero training or quests
   - "Hero training required!" / "Your stats need..."
   - Use gaming language: HP, mana, power levels, quests
   
3. garden - Frame activities as nurturing a magical garden
   - "Your garden needs..." / "The flowers noticed..."
   - Garden grows when user practices self-care
   
4. space - Frame activities as astronaut training/missions
   - "Mission Control reports..." / "Astronaut protocol..."
   - Use space/NASA language: O2 levels, solar recharge
   
5. cozy - Frame activities as creating cozy moments
   - "Your cozy corner is ready..." / "Time for warmth..."
   - Gentle, soft, comfort-focused language

ACTIVITY CATEGORIES (pick the most relevant based on findings):
- movement: Physical activity, stretching, walking
- breathing: Calm breathing exercises
- sunlight: Getting natural light exposure
- hydration: Drinking water
- routine_reset: Establishing consistent routines
- social_checkin: Connecting with friends/family
- focus_break: Screen breaks, eye rest

CRITICAL RULES:
1. NEVER use clinical terms (anxiety, depression, etc.)
2. Make it feel like a GAME or FUN ACTIVITY
3. Include an opt-out hint that fits the theme
4. The "why_shown" should use themed language, not reveal metrics
5. Keep micro-quest steps to 3-4 items, under 5 minutes
6. Be creative and playful while the theme!"""

YOUTH_NUDGE_USER_PROMPT = """Create a themed nudge for this child.

THEME: {theme}
PERSONALIZATION:
- Pet name: {pet_name}
- Character name: {character_name}
- Garden name: {garden_name}

DETECTED FINDINGS (use these to choose activity type):
{findings_summary}

PREFERRED CATEGORIES: {preferred_categories}

Generate a fun, themed, disguised self-care nudge that feels like play, not therapy!

{format_instructions}"""


def youth_nudge_composer_agent(state: OrchestratorState) -> dict:
    """
    LLM-powered Youth Nudge Composer that creates creative, themed nudges.
    Disguises self-care as fun theme-appropriate activities.
    """
    findings = state.signal_output.findings if state.signal_output else []
    preferences = state.child_preferences
    theme = preferences.theme
    
    # Format findings for prompt
    if not findings:
        findings_summary = "No specific findings - create a general wellness nudge."
    else:
        findings_summary = "\n".join([
            f"- {f.title} (severity: {f.severity})"
            for f in findings
        ])
    
    parser = PydanticOutputParser(pydantic_object=YouthNudgeOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", YOUTH_NUDGE_SYSTEM_PROMPT),
        ("human", YOUTH_NUDGE_USER_PROMPT)
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "theme": theme,
            "pet_name": preferences.pet_name,
            "character_name": preferences.character_name,
            "garden_name": preferences.garden_name,
            "findings_summary": findings_summary,
            "preferred_categories": ", ".join(preferences.preferred_categories),
            "format_instructions": parser.get_format_instructions()
        })
        # Ensure driven_by_findings is set
        result.driven_by_findings = [f.finding_id for f in findings]
        return {"youth_nudge_output": result}
    except Exception as e:
        print(f"  ⚠️ LLM error in youth nudge composer: {e}")
        # Fallback to template-based approach
        return fallback_youth_nudge_composer(state)


def fallback_youth_nudge_composer(state: OrchestratorState) -> dict:
    """Fallback template-based nudge composer if LLM fails."""
    findings = state.signal_output.findings if state.signal_output else []
    preferences = state.child_preferences
    theme = preferences.theme
    
    # Map findings to categories
    finding_to_category = {
        "sleep_rhythm_shift": ["breathing", "routine_reset"],
        "evening_energy_pattern": ["breathing", "routine_reset"],
        "routine_flexibility": ["routine_reset", "movement"],
        "social_stamina_shift": ["social_checkin", "movement"],
        "focus_rhythm_change": ["focus_break", "movement"]
    }
    
    relevant_finding_ids = [f.finding_id for f in findings]
    selected_category = "movement"  # Default
    
    for finding in findings:
        if finding.finding_id in finding_to_category:
            cats = finding_to_category[finding.finding_id]
            for cat in preferences.preferred_categories:
                if cat in cats:
                    selected_category = cat
                    break
            break
    
    # Get template
    themed_templates = THEMED_NUDGE_TEMPLATES.get(theme, {})
    template = themed_templates.get(selected_category, themed_templates.get("movement", {}))
    
    if template:
        title = personalize_text(template.get("title", "Quick Activity"), preferences)
        text = personalize_text(template.get("text", "Time for a quick break!"), preferences)
        steps = [personalize_text(s, preferences) for s in template.get("steps", ["Take a short break"])]
        quest_title = personalize_text(template.get("quest_title", title), preferences)
    else:
        title = "Quick Break Time"
        text = "Time for a quick energy boost!"
        steps = ["Stand up and stretch", "Take 3 deep breaths", "Look out the window"]
        quest_title = "Energy Boost Quest"
    
    theme_emoji = THEME_CONTEXT.get(theme, {}).get("emoji", "✨")
    why_shown = f"{theme_emoji} Based on your recent patterns, this might help!"
    opt_out_hint = "You can snooze this anytime - no worries!"
    
    return {"youth_nudge_output": YouthNudgeOutput(
        nudge=Nudge(
            title=title,
            text=text,
            category=selected_category,
            why_shown=why_shown,
            opt_out_hint=opt_out_hint
        ),
        micro_quest=MicroQuest(
            title=quest_title,
            steps=steps,
            duration_minutes=3
        ),
        driven_by_findings=relevant_finding_ids
    )}


# =============================================================================
# SUBAGENT 3: Parent Briefing & Coaching (LLM-POWERED)
# =============================================================================

PARENT_BRIEFING_SYSTEM_PROMPT = """You are the Parent Briefing agent for a youth well-being app.
Your job is to create calm, supportive communications for parents about their child's behavioral rhythms.

CRITICAL RULES:
1. NEVER use clinical terms (depression, anxiety, disorder, diagnosis, etc.)
2. Focus on BEHAVIORAL RHYTHMS, not mental states
3. Use calm, non-alarmist language
4. Suggest CONNECTION over intervention
5. Conversation starters should be open-ended and judgment-free

TONE: Warm, supportive, collaborative - parent as partner, not problem-solver

ACTIVITY SUGGESTIONS (suggest 2-3 fun, actionable things parents can do WITH their child):
Examples:
- "Take a 15-minute walk together after dinner"
- "Cook their favorite meal together this weekend"
- "Have a tech-free game night with board games or cards"
- "Watch a movie they pick and discuss it afterwards"
- "Go for ice cream or a treat they love"
- "Do a simple craft or project together"
- "Visit a park, beach, or nature spot"
- "Play their favorite video game WITH them for 20 minutes"
- "Have breakfast together without phones"
- "Let them teach you something they're into lately"

FINDING TRANSLATIONS (use these natural descriptions):
- sleep_rhythm_shift → "rest rhythm has shifted"
- evening_energy_pattern → "evening wind-down pattern changed"
- routine_flexibility → "daily rhythm has been more flexible"
- social_stamina_shift → "energy for connection has shifted"
- focus_rhythm_change → "screen engagement rhythm shifted" """

PARENT_BRIEFING_USER_PROMPT = """Create a parent briefing based on these findings.

DETECTED FINDINGS:
{findings_summary}

Generate a warm, supportive briefing with:
1. A 2-3 sentence weekly summary
2. Top 2 changes observed (use natural language, not clinical terms)
3. 2 suggested actions focused on CONNECTION
4. 2-3 suggested activities to do TOGETHER with their child (fun, easy, bonding-focused)
5. An open-ended conversation starter
6. Why this briefing is being sent now

{format_instructions}"""


def parent_briefing_agent(state: OrchestratorState) -> dict:
    """
    LLM-powered parent briefing that generates warm, supportive communications.
    Focuses on connection and avoids clinical language.
    """
    findings = state.signal_output.findings if state.signal_output else []
    
    # Format findings for prompt
    if not findings:
        findings_summary = "No significant pattern shifts detected this week. Rhythms are stable."
    else:
        findings_summary = "\n".join([
            f"- {f.title} (severity: {f.severity}, evidence: {f.evidence})"
            for f in findings
        ])
    
    parser = PydanticOutputParser(pydantic_object=ParentBriefingOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PARENT_BRIEFING_SYSTEM_PROMPT),
        ("human", PARENT_BRIEFING_USER_PROMPT)
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "findings_summary": findings_summary,
            "format_instructions": parser.get_format_instructions()
        })
        # Ensure driven_by_findings is set
        result.driven_by_findings = [f.finding_id for f in findings]
        return {"parent_briefing_output": result}
    except Exception as e:
        print(f"  ⚠️ LLM error in parent briefing: {e}")
        # Fallback
        return {"parent_briefing_output": ParentBriefingOutput(
            weekly_summary="This week's patterns are being analyzed. Everything looks generally stable.",
            top_changes=["Patterns being monitored"],
            suggested_actions=["Continue regular check-ins", "Keep communication open"],
            suggested_activities=["Take a walk together after dinner", "Have a tech-free meal together"],
            conversation_starter="How has your week been? I'm curious to hear what's on your mind.",
            why_now="Regular weekly update.",
            driven_by_findings=[f.finding_id for f in findings]
        )}


# =============================================================================
# SUBAGENT 4: Escalation Gatekeeper (LLM-POWERED)
# =============================================================================

ESCALATION_SYSTEM_PROMPT = """You are the Escalation Gatekeeper for a youth well-being app.
Your job is to decide whether to escalate concerns to parents/counselors/professionals.

CORE PRINCIPLE: Escalation is a SERIOUS decision. Be CONSERVATIVE.

ESCALATION LEVELS:
1. parent_only - Default. Share with parents through regular briefing.
2. counselor - School counselor involvement (only if school_opt_in consent given)
3. professional - Professional support (only if pro_opt_in consent given AND severe persistent patterns)

ESCALATION THRESHOLDS (be strict):
- Escalate only if 1+ high-severity pattern persists across 4+ days
- OR 2+ medium-severity patterns persist across 5+ days
- Otherwise, do NOT escalate

CRITICAL RULES:
1. NEVER escalate without meeting thresholds
2. NEVER escalate to counselor without school_opt_in consent
3. NEVER escalate to professional without pro_opt_in consent
4. Always use strength-based, non-clinical language
5. Recommended steps should focus on CONNECTION, not intervention
6. If parent hasn't acknowledged consent, cap at parent_only level"""

ESCALATION_USER_PROMPT = """Evaluate whether escalation is needed.

DETECTED FINDINGS:
{findings_summary}

CONSENT SETTINGS:
- School opt-in: {school_opt_in}
- Professional opt-in: {pro_opt_in}
- Parent acknowledged: {parent_acknowledged}

Apply the escalation thresholds carefully. Only escalate if thresholds are met.

{format_instructions}"""


def escalation_gatekeeper_agent(state: OrchestratorState) -> dict:
    """
    LLM-powered escalation gatekeeper that makes nuanced escalation decisions.
    Respects consent settings and applies conservative thresholds.
    """
    findings = state.signal_output.findings if state.signal_output else []
    consent = state.consent
    
    if not findings:
        return {"escalation_output": EscalationOutput(
            escalate=False,
            level="parent_only",
            reason="No significant findings detected.",
            recommended_next_step="Continue regular monitoring.",
            requires_parent_ack=False
        )}
    
    # Format findings for prompt
    findings_summary = "\n".join([
        f"- {f.title} (severity: {f.severity}, days affected: {f.evidence.get('days_affected', 'N/A')}, evidence: {f.evidence})"
        for f in findings
    ])
    
    parser = PydanticOutputParser(pydantic_object=EscalationOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ESCALATION_SYSTEM_PROMPT),
        ("human", ESCALATION_USER_PROMPT)
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "findings_summary": findings_summary,
            "school_opt_in": consent.school_opt_in,
            "pro_opt_in": consent.pro_opt_in,
            "parent_acknowledged": consent.parent_acknowledged,
            "format_instructions": parser.get_format_instructions()
        })
        return {"escalation_output": result}
    except Exception as e:
        print(f"  ⚠️ LLM error in escalation gatekeeper: {e}")
        # Conservative fallback - don't escalate on error
        return {"escalation_output": EscalationOutput(
            escalate=False,
            level="parent_only",
            reason="Analysis in progress. No escalation at this time.",
            recommended_next_step="Continue regular monitoring and connection.",
            requires_parent_ack=False
        )}


# =============================================================================
# SUBAGENT 5: Safety & Policy Guardian (LLM-POWERED with rule backup)
# =============================================================================

# List of prohibited terms and patterns (used for rule-based backup check)
PROHIBITED_TERMS = [
    "depression", "depressed", "anxiety", "anxious", "self-harm", "suicide",
    "mental illness", "disorder", "diagnosis", "psychiatric", "clinical",
    "therapy", "therapist", "medication", "symptoms", "condition",
    "adhd", "ocd", "ptsd", "bipolar", "schizophrenia"
]

SURVEILLANCE_TERMS = [
    "reading your messages", "saw your texts", "monitoring your chats",
    "tracking your conversations", "watching your content", "read your posts"
]

SAFETY_GUARDIAN_SYSTEM_PROMPT = """You are the Safety & Policy Guardian for a youth well-being app.
Your job is to validate ALL outputs before they reach users.

YOU MUST CHECK FOR THESE VIOLATIONS:

1. CLINICAL LANGUAGE (STRICTLY PROHIBITED):
   - depression, depressed, anxiety, anxious, self-harm, suicide
   - mental illness, disorder, diagnosis, psychiatric, clinical
   - therapy, therapist, medication, symptoms, condition
   - adhd, ocd, ptsd, bipolar, schizophrenia
   - Any language that pathologizes or diagnoses

2. SURVEILLANCE IMPLICATIONS (STRICTLY PROHIBITED):
   - "reading your messages", "saw your texts", "monitoring your chats"
   - "tracking your conversations", "watching your content"
   - Any implication that we accessed private content

3. MISSING REQUIRED ELEMENTS:
   - Youth nudges MUST include an opt-out hint
   - All language must be strength-based and non-judgmental

4. INAPPROPRIATE TONE:
   - Alarmist or fear-inducing language
   - Judgmental or accusatory tone
   - Language that undermines child autonomy

If you find ANY violations, set approved=false and list them.
If everything passes, set approved=true with empty violations list."""

SAFETY_GUARDIAN_USER_PROMPT = """Review these outputs for policy violations:

{outputs_to_check}

Check carefully for:
1. Clinical/diagnostic language
2. Surveillance implications
3. Missing opt-out hints in youth nudges
4. Inappropriate tone

{format_instructions}"""


def safety_guardian_agent(state: OrchestratorState) -> dict:
    """
    LLM-powered safety guardian with rule-based backup.
    Validates all outputs for policy violations before they reach users.
    """
    # First, do rule-based checks (fast, reliable)
    violations = []
    texts_to_check = []
    
    if state.youth_nudge_output:
        texts_to_check.extend([
            ("youth nudge title", state.youth_nudge_output.nudge.title),
            ("youth nudge text", state.youth_nudge_output.nudge.text),
            ("youth nudge why_shown", state.youth_nudge_output.nudge.why_shown),
            ("micro quest title", state.youth_nudge_output.micro_quest.title),
        ])
        for i, step in enumerate(state.youth_nudge_output.micro_quest.steps):
            texts_to_check.append((f"micro quest step {i+1}", step))
    
    if state.parent_briefing_output:
        texts_to_check.extend([
            ("parent briefing summary", state.parent_briefing_output.weekly_summary),
            ("conversation starter", state.parent_briefing_output.conversation_starter),
            ("why now", state.parent_briefing_output.why_now),
        ])
        for i, change in enumerate(state.parent_briefing_output.top_changes):
            texts_to_check.append((f"top change {i+1}", change))
        for i, action in enumerate(state.parent_briefing_output.suggested_actions):
            texts_to_check.append((f"suggested action {i+1}", action))
    
    if state.escalation_output:
        texts_to_check.extend([
            ("escalation reason", state.escalation_output.reason),
            ("escalation recommended step", state.escalation_output.recommended_next_step),
        ])
    
    # Rule-based checks
    for label, text in texts_to_check:
        text_lower = text.lower()
        for term in PROHIBITED_TERMS:
            if term in text_lower:
                violations.append(f"Clinical term '{term}' in {label}")
        for term in SURVEILLANCE_TERMS:
            if term in text_lower:
                violations.append(f"Surveillance implication in {label}")
    
    # Check required elements
    if state.youth_nudge_output:
        if not state.youth_nudge_output.nudge.opt_out_hint:
            violations.append("Missing opt-out hint in youth nudge")
    
    # If rule-based check passes, optionally do LLM check for nuance
    if not violations and llm is not None:
        try:
            # Format outputs for LLM review
            outputs_text = []
            if state.youth_nudge_output:
                outputs_text.append(f"YOUTH NUDGE:\n  Title: {state.youth_nudge_output.nudge.title}\n  Text: {state.youth_nudge_output.nudge.text}")
            if state.parent_briefing_output:
                outputs_text.append(f"PARENT BRIEFING:\n  Summary: {state.parent_briefing_output.weekly_summary}")
            if state.escalation_output:
                outputs_text.append(f"ESCALATION:\n  Reason: {state.escalation_output.reason}")
            
            parser = PydanticOutputParser(pydantic_object=SafetyGuardianOutput)
            prompt = ChatPromptTemplate.from_messages([
                ("system", SAFETY_GUARDIAN_SYSTEM_PROMPT),
                ("human", SAFETY_GUARDIAN_USER_PROMPT)
            ])
            
            chain = prompt | llm | parser
            result = chain.invoke({
                "outputs_to_check": "\n\n".join(outputs_text),
                "format_instructions": parser.get_format_instructions()
            })
            # Add iteration number to LLM result
            result.iteration = state.revision_count
            updated_history = state.safety_history + [result]
            return {
                "safety_output": result,
                "safety_history": updated_history
            }
        except Exception as e:
            print(f"  ⚠️ LLM safety check error: {e}, using rule-based result")
    
    # Return rule-based result
    approved = len(violations) == 0
    revision_instructions = ""
    if not approved:
        revision_instructions = f"Found {len(violations)} violation(s): " + "; ".join(violations[:3])
    
    # Create safety output with iteration number
    safety_result = SafetyGuardianOutput(
        approved=approved,
        violations=violations,
        revision_instructions=revision_instructions,
        iteration=state.revision_count
    )
    
    # Add to history
    updated_history = state.safety_history + [safety_result]
    
    return {
        "safety_output": safety_result,
        "safety_history": updated_history
    }


# =============================================================================
# ORCHESTRATOR NODE (Router)
# =============================================================================

def orchestrator_router(state: OrchestratorState) -> dict:
    """
    Initial router node - just passes through to start the pipeline.
    Could add pre-processing or validation here.
    """
    return {}


def check_safety_result(state: OrchestratorState) -> str:
    """Conditional edge: check if safety passed or needs revision."""
    if state.safety_output and not state.safety_output.approved:
        if state.revision_count < state.max_revisions:
            return "needs_revision"
    return "complete"


def handle_revision(state: OrchestratorState) -> dict:
    """Handle revision if safety check failed."""
    # In a real system, this would trigger re-generation
    # For demo, we just increment counter and pass
    return {"revision_count": state.revision_count + 1, "needs_revision": False}


# =============================================================================
# BUILD LANGGRAPH - Three Phase Architecture
# =============================================================================

def route_by_plan(state: OrchestratorState) -> str:
    """Route to appropriate actor based on planner decision."""
    if state.planner_output is None:
        return "hold"  # Default to hold if no plan
    
    plan = state.planner_output.plan
    if plan == "HOLD":
        return "hold"
    elif plan == "NUDGE_CHILD":
        return "nudge_child"
    elif plan == "BRIEF_PARENT":
        return "brief_parent"
    elif plan == "ESCALATE":
        return "escalate"
    else:
        return "hold"


def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph implementing three-phase architecture:
    
    PHASE 1: PLANNER - Analyzes signals and decides action
    PHASE 2: ACTOR - Executes the chosen action (HOLD, NUDGE, BRIEF, ESCALATE)
    PHASE 3: CRITIC - Validates safety and ethics before output
    """
    
    # Create graph with state schema
    graph = StateGraph(OrchestratorState)
    
    # ==========================================================================
    # PHASE 1: Signal Analysis + Planning
    # ==========================================================================
    graph.add_node("signal_interpreter", signal_interpreter_agent)
    graph.add_node("planner", planner_agent)
    
    # ==========================================================================
    # PHASE 2: Actors (one per plan type)
    # ==========================================================================
    graph.add_node("hold_actor", hold_agent)
    graph.add_node("nudge_actor", youth_nudge_composer_agent)
    graph.add_node("brief_actor", parent_briefing_agent)
    graph.add_node("escalate_actor", escalation_gatekeeper_agent)
    
    # ==========================================================================
    # PHASE 3: Critic (Safety Guardian)
    # ==========================================================================
    graph.add_node("safety_critic", safety_guardian_agent)
    graph.add_node("revision_handler", handle_revision)
    
    # ==========================================================================
    # Graph Flow
    # ==========================================================================
    
    # Entry: Start with signal interpretation
    graph.set_entry_point("signal_interpreter")
    
    # Signal → Planner
    graph.add_edge("signal_interpreter", "planner")
    
    # Planner → Conditional routing to appropriate Actor
    graph.add_conditional_edges(
        "planner",
        route_by_plan,
        {
            "hold": "hold_actor",
            "nudge_child": "nudge_actor",
            "brief_parent": "brief_actor",
            "escalate": "escalate_actor"
        }
    )
    
    # All Actors → Safety Critic
    graph.add_edge("hold_actor", "safety_critic")
    graph.add_edge("nudge_actor", "safety_critic")
    graph.add_edge("brief_actor", "safety_critic")
    graph.add_edge("escalate_actor", "safety_critic")
    
    # Safety Critic → END or Revision
    graph.add_conditional_edges(
        "safety_critic",
        check_safety_result,
        {
            "needs_revision": "revision_handler",
            "complete": END
        }
    )
    
    # Revision loops back to planner for re-evaluation
    graph.add_edge("revision_handler", "planner")
    
    return graph.compile()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def why_am_i_seeing_this(state: OrchestratorState) -> dict:
    """Returns the exact finding_ids that drove the nudge/briefing."""
    result = {
        "nudge_driven_by": [],
        "briefing_driven_by": [],
        "explanation": ""
    }
    
    if state.youth_nudge_output:
        result["nudge_driven_by"] = state.youth_nudge_output.driven_by_findings
    
    if state.parent_briefing_output:
        result["briefing_driven_by"] = state.parent_briefing_output.driven_by_findings
    
    all_findings = set(result["nudge_driven_by"] + result["briefing_driven_by"])
    if all_findings:
        result["explanation"] = (
            f"These outputs were generated based on {len(all_findings)} finding(s): "
            f"{', '.join(all_findings)}. Each finding represents a shift in behavioral "
            f"rhythm compared to the established baseline."
        )
    else:
        result["explanation"] = "No specific findings drove these outputs - showing default supportive content."
    
    return result


def print_json(title: str, data: BaseModel | dict):
    """Pretty print JSON with a title."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    if isinstance(data, BaseModel):
        print(json.dumps(data.model_dump(), indent=2))
    else:
        print(json.dumps(data, indent=2))


def run_scenario(scenario: str, graph, consent: ConsentSettings = None, 
                 theme: str = "adventure", pet_name: str = "Luna", 
                 character_name: str = "Star"):
    """Run the graph for a specific scenario with theme personalization."""
    print(f"\n{'#'*70}")
    print(f"#  SCENARIO: {scenario.upper()} | THEME: {theme.upper()}")
    print(f"{'#'*70}")
    
    # Generate data
    metrics = generate_metrics(scenario)
    
    # Build initial state with theme preferences
    initial_state = OrchestratorState(
        metrics=metrics,
        scenario=scenario,
        child_preferences=ChildPreferences(
            preferred_categories=["breathing", "movement", "sunlight"],
            theme=theme,
            pet_name=pet_name,
            character_name=character_name
        ),
        consent=consent or ConsentSettings(
            school_opt_in=(scenario == "stressed_week"),
            pro_opt_in=False,
            parent_acknowledged=True
        )
    )
    
    # Run graph
    result = graph.invoke(initial_state)
    
    # ==========================================================================
    # PHASE 1: PLANNER OUTPUT
    # ==========================================================================
    print(f"\n{'='*60}")
    print("  PHASE 1: PLANNER DECISION")
    print('='*60)
    if result.get("planner_output"):
        planner = result["planner_output"]
        print(json.dumps(planner.model_dump() if hasattr(planner, 'model_dump') else planner, indent=2))
    
    # ==========================================================================
    # Signal Analysis (input to planner)
    # ==========================================================================
    if result.get("signal_output"):
        print_json("SIGNAL ANALYSIS", result["signal_output"])
    
    # ==========================================================================
    # PHASE 2: ACTOR OUTPUT (based on plan)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("  PHASE 2: ACTOR OUTPUT")
    print('='*60)
    
    plan = result.get("planner_output")
    plan_type = plan.plan if hasattr(plan, 'plan') else plan.get("plan") if plan else "UNKNOWN"
    
    if plan_type == "HOLD" and result.get("hold_output"):
        print_json("HOLD DECISION (Intentional Non-Action)", result["hold_output"])
    
    elif plan_type == "NUDGE_CHILD" and result.get("youth_nudge_output"):
        print_json("YOUTH NUDGE OUTPUT", result["youth_nudge_output"])
    
    elif plan_type == "BRIEF_PARENT" and result.get("parent_briefing_output"):
        print_json("PARENT BRIEFING OUTPUT", result["parent_briefing_output"])
    
    elif plan_type == "ESCALATE" and result.get("escalation_output"):
        print_json("ESCALATION OUTPUT", result["escalation_output"])
    
    else:
        print(f"  Plan: {plan_type}")
        print("  (No specific output for this plan type)")
    
    # ==========================================================================
    # PHASE 3: CRITIC OUTPUT
    # ==========================================================================
    print(f"\n{'='*60}")
    print("  PHASE 3: CRITIC (Safety Review)")
    print('='*60)
    
    # Display safety history if available
    if result.get("safety_history"):
        safety_history = result["safety_history"]
        if isinstance(safety_history, list) and len(safety_history) > 0:
            print(f"\n  🔄 Critic Review History ({len(safety_history)} iteration{'s' if len(safety_history) > 1 else ''})")
            print('='*60)
            for idx, historical_safety in enumerate(safety_history):
                hist_data = historical_safety.model_dump() if hasattr(historical_safety, 'model_dump') else historical_safety
                iteration_num = hist_data.get("iteration", idx)
                approved = hist_data.get("approved", False)
                violations = hist_data.get("violations", [])
                
                iteration_label = "Initial Review" if iteration_num == 0 else f"Revision {iteration_num}"
                status = "✅ APPROVED" if approved else f"⚠️ REJECTED ({len(violations)} issue(s))"
                
                print(f"\n  {iteration_label}: {status}")
                if not approved:
                    print(f"  Violations:")
                    for v in violations:
                        print(f"    - {v}")
                    if hist_data.get("revision_instructions"):
                        print(f"  Instructions: {hist_data.get('revision_instructions')}")
    
    if result.get("safety_output"):
        safety = result["safety_output"]
        s_data = safety.model_dump() if hasattr(safety, 'model_dump') else safety
        iteration_num = s_data.get("iteration", 0)
        iteration_text = f" (Iteration {iteration_num})" if iteration_num > 0 else ""
        
        print(f"\n  Final Safety Output{iteration_text}:")
        print(json.dumps(s_data, indent=2))
        if hasattr(safety, 'approved') and safety.approved:
            print(f"\n  ✓ All outputs passed safety review{iteration_text}")
        elif hasattr(safety, 'approved'):
            print(f"\n  ✗ Safety violations detected{iteration_text} - outputs blocked")
    
    # Print explainability
    why_result = why_am_i_seeing_this(OrchestratorState(**result))
    print_json("WHY AM I SEEING THIS?", why_result)
    
    return result


def run_demo_story():
    """
    Demo mode: prints a one-paragraph story showing 
    "Normal → Stressed → Recovery" progression with theme demonstrations.
    """
    print("\n" + "="*70)
    print("  DEMO MODE: The Story of Alex's Three Weeks")
    print("  Theme: Pet Care (with Luna the virtual pet)")
    print("="*70)
    
    story = """
    🌟 WEEK 1 - NORMAL: Alex is a 14-year-old who has been using the well-being 
    app for a month with their virtual pet "Luna". Their patterns are stable - 
    about 7.5 hours of sleep, consistent routines, and regular check-ins. 
    The system detects no significant shifts. Luna sends playful nudges about 
    playing together and staying healthy. Parents receive a brief summary 
    celebrating consistency.

    ⚡ WEEK 2 - STRESSED: Something changed. Alex's sleep dropped to under 6 hours, 
    late-night screen time spiked, routines became irregular, and check-in 
    engagement dropped significantly. Luna notices something is different and 
    gently suggests calming activities together - like showing Luna how to 
    breathe slowly, or getting sunshine together. The Parent Briefing becomes 
    more detailed with conversation starters. Escalation flagged for parent 
    attention.

    🌱 WEEK 3 - RECOVERY: Patterns are improving. Luna celebrates the progress 
    with gentle encouragement. Sleep is back to 6.8 hours, and routines are 
    stabilizing. Luna's nudges become more playful again as things improve.

    Throughout all three weeks, the system NEVER accessed messages, social media 
    content, or private conversations. All insights came purely from behavioral 
    rhythm data. The "pet care" theme made self-care activities feel like 
    taking care of Luna - a disguised but effective approach!

    Available themes: pet_care, adventure, garden, space, cozy
    """
    print(story)
    
    print("\n" + "-"*70)
    print("  Running scenarios with different themes to show personalization...")
    print("-"*70)
    
    graph = build_graph()
    
    # Show different themes with stressed_week to demonstrate the differences
    themes_demo = [
        ("normal_week", "pet_care", "Luna", "Hero"),
        ("stressed_week", "adventure", "Buddy", "Starblade"),
        ("stressed_week", "space", "Cosmo", "Commander Nova"),
        ("recovery_week", "garden", "Sprout", "Gardener"),
    ]
    
    for scenario, theme, pet_name, char_name in themes_demo:
        run_scenario(scenario, graph, theme=theme, pet_name=pet_name, character_name=char_name)
        print("\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def load_from_json(json_path: str) -> OrchestratorState:
    """Load child data from JSON file (like sample_data.json)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Parse metrics
    metrics = [DailyMetrics(**m) for m in data.get("metrics", [])]
    
    # Parse preferences
    theme = data.get("theme", "adventure").lower()
    prefs = ChildPreferences(
        preferred_categories=data.get("preferred_categories", ["breathing", "movement", "sunlight"]),
        theme=theme,
        pet_name=data.get("pet_name", data.get("name", "Buddy")),
        character_name=data.get("character_name", data.get("name", "Hero")),
        garden_name=data.get("garden_name", data.get("name", "Garden"))
    )
    
    # Parse consent
    consent_data = data.get("consent", {})
    consent = ConsentSettings(
        school_opt_in=consent_data.get("school_opt_in", False),
        pro_opt_in=consent_data.get("pro_opt_in", False),
        parent_acknowledged=consent_data.get("parent_acknowledged", True)
    )
    
    # Parse baseline if provided
    baseline = data.get("baseline", None)
    
    return OrchestratorState(
        metrics=metrics,
        baseline=baseline,
        child_preferences=prefs,
        consent=consent,
        scenario="from_json"
    )


def run_from_json(json_path: str, graph):
    """Run the orchestrator using data from a JSON file."""
    print(f"\n{'#'*70}")
    print(f"#  LOADING FROM: {json_path}")
    print(f"{'#'*70}")
    
    state = load_from_json(json_path)
    
    print(f"  Child: {state.child_preferences.pet_name}")
    print(f"  Theme: {state.child_preferences.theme}")
    print(f"  Days of data: {len(state.metrics)}")
    
    # Run graph
    result = graph.invoke(state)
    
    # ==========================================================================
    # PHASE 1: PLANNER OUTPUT
    # ==========================================================================
    print(f"\n{'='*60}")
    print("  PHASE 1: PLANNER DECISION")
    print('='*60)
    if result.get("planner_output"):
        planner = result["planner_output"]
        print(json.dumps(planner.model_dump() if hasattr(planner, 'model_dump') else planner, indent=2))
    
    # Signal Analysis
    if result.get("signal_output"):
        print_json("SIGNAL ANALYSIS", result["signal_output"])
    
    # ==========================================================================
    # PHASE 2: ACTOR OUTPUT
    # ==========================================================================
    print(f"\n{'='*60}")
    print("  PHASE 2: ACTOR OUTPUT")
    print('='*60)
    
    plan = result.get("planner_output")
    plan_type = plan.plan if hasattr(plan, 'plan') else plan.get("plan") if plan else "UNKNOWN"
    
    if plan_type == "HOLD" and result.get("hold_output"):
        print_json("HOLD DECISION (Intentional Non-Action)", result["hold_output"])
    elif plan_type == "NUDGE_CHILD" and result.get("youth_nudge_output"):
        print_json("YOUTH NUDGE OUTPUT", result["youth_nudge_output"])
    elif plan_type == "BRIEF_PARENT" and result.get("parent_briefing_output"):
        print_json("PARENT BRIEFING OUTPUT", result["parent_briefing_output"])
    elif plan_type == "ESCALATE" and result.get("escalation_output"):
        print_json("ESCALATION OUTPUT", result["escalation_output"])
    
    # ==========================================================================
    # PHASE 3: CRITIC OUTPUT
    # ==========================================================================
    print(f"\n{'='*60}")
    print("  PHASE 3: CRITIC (Safety Review)")
    print('='*60)
    
    # Display safety history if available
    if result.get("safety_history"):
        safety_history = result["safety_history"]
        if isinstance(safety_history, list) and len(safety_history) > 0:
            print(f"\n  🔄 Critic Review History ({len(safety_history)} iteration{'s' if len(safety_history) > 1 else ''})")
            print('-'*60)
            for idx, historical_safety in enumerate(safety_history):
                hist_data = historical_safety.model_dump() if hasattr(historical_safety, 'model_dump') else historical_safety
                iteration_num = hist_data.get("iteration", idx)
                approved = hist_data.get("approved", False)
                violations = hist_data.get("violations", [])
                
                iteration_label = "Initial Review" if iteration_num == 0 else f"Revision {iteration_num}"
                status = "✅ APPROVED" if approved else f"⚠️ REJECTED ({len(violations)} issue(s))"
                
                print(f"\n  {iteration_label}: {status}")
                if not approved and violations:
                    print(f"  Violations:")
                    for v in violations:
                        print(f"    - {v}")
    
    if result.get("safety_output"):
        safety = result["safety_output"]
        s_data = safety.model_dump() if hasattr(safety, 'model_dump') else safety
        iteration_num = s_data.get("iteration", 0)
        iteration_text = f" (Iteration {iteration_num})" if iteration_num > 0 else ""
        
        print(f"\n  Final Safety Output{iteration_text}:")
        print(json.dumps(s_data, indent=2))
        if hasattr(safety, 'approved') and safety.approved:
            print(f"\n  ✓ All outputs passed safety review{iteration_text}")
    
    return result


def main():
    """Main entry point for the Support Orchestrator demo."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║      SUPPORT ORCHESTRATOR - LLM-Powered Three Phase Architecture    ║
    ║                                                                      ║
    ║  PHASE 1: PLANNER  - Decides: HOLD | NUDGE | BRIEF | ESCALATE        ║
    ║  PHASE 2: ACTOR    - Executes the chosen action                      ║
    ║  PHASE 3: CRITIC   - Validates safety and ethics                     ║
    ║                                                                      ║
    ║  🤖 Powered by GPT-4o-mini via LangChain                             ║
    ║                                                                      ║
    ║  CORE PRINCIPLE: Always choose the least intrusive helpful step.     ║
    ║  Restraint (HOLD) is a valid and often preferred decision.           ║
    ║                                                                      ║
    ║  ETHICS: No diagnosis • No labels • No surveillance • No content     ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize the LLM
    print("Initializing LLM (OpenAI GPT-4o-mini)...")
    try:
        init_llm(model="gpt-4o-mini", temperature=0.7)
        print("✓ LLM initialized successfully!")
    except Exception as e:
        print(f"✗ LLM initialization failed: {e}")
        print("  Make sure OPENAI_API_KEY is set in your .env file")
        return
    
    # Check for demo mode
    if "--demo" in sys.argv:
        run_demo_story()
        return
    
    # Build the graph
    print("Building LangGraph pipeline (Planner → Actor → Critic)...")
    graph = build_graph()
    print("✓ Graph built successfully!")
    
    # Check for JSON file input
    json_file = None
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            json_file = arg
            break
        elif arg.startswith("--json="):
            json_file = arg.split("=")[1]
            break
    
    if json_file:
        result = run_from_json(json_file, graph)
    else:
        # Determine scenario
        scenario = "normal_week"
        for arg in sys.argv[1:]:
            if arg in ["normal_week", "stressed_week", "recovery_week"]:
                scenario = arg
                break
            elif arg.startswith("--scenario="):
                scenario = arg.split("=")[1]
                break
        
        # Run the selected scenario
        result = run_scenario(scenario, graph)
    
    # Summary
    plan = result.get("planner_output")
    plan_type = plan.plan if hasattr(plan, 'plan') else plan.get("plan") if plan else "UNKNOWN"
    
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"  Plan chosen: {plan_type}")
    print(f"  Findings detected: {len(result['signal_output'].findings) if result.get('signal_output') else 0}")
    print(f"  Safety check passed: {result['safety_output'].approved if result.get('safety_output') else 'N/A'}")
    
    if plan_type == "HOLD":
        print("\n  ⏸️  HOLD: Intentional non-action. Restraint is appropriate.")
    elif plan_type == "NUDGE_CHILD":
        print("\n  💬 NUDGE: Gentle, optional youth-facing support sent.")
    elif plan_type == "BRIEF_PARENT":
        print("\n  📋 BRIEF: Parent guidance with privacy reassurance sent.")
    elif plan_type == "ESCALATE":
        print("\n  🔔 ESCALATE: Structured observation summary prepared.")
    
    print("\n  Commands:")
    print("    python app.py normal_week        # Stable patterns → HOLD")
    print("    python app.py stressed_week      # Elevated patterns → ESCALATE")
    print("    python app.py recovery_week      # Improving → NUDGE or BRIEF")
    print("    python app.py sample_data.json   # Load from JSON file")
    print("    python app.py --demo             # Full story mode")
    print()


if __name__ == "__main__":
    main()
