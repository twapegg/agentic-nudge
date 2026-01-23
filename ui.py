"""
Support Orchestrator - Dark Mode Explainability UI
===================================================

A sleek, dark-themed Streamlit interface with human-readable language
that visualizes the thought process of each LLM-powered agent.

Run with: streamlit run ui.py
"""

import streamlit as st
import json
import time
from datetime import datetime

from app import (
    init_llm, build_graph, generate_metrics, load_from_json,
    OrchestratorState, ChildPreferences, ConsentSettings, DailyMetrics,
    why_am_i_seeing_this
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Support Orchestrator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS with human-friendly styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e5e5e5;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: #3d3d5c; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #5c5c8a; }
    
    .glass-card {
        background: rgba(30, 30, 50, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #e5e5e5;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .hero-section {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 24px;
        padding: 48px 32px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 30% 20%, rgba(139, 92, 246, 0.4) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(99, 102, 241, 0.4) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.8);
        margin-top: 12px;
        position: relative;
        z-index: 1;
    }
    
    .phase-card {
        background: linear-gradient(135deg, rgba(30,30,50,0.9) 0%, rgba(40,40,70,0.7) 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    .phase-card:hover { transform: scale(1.02); }
    .phase-card.planner { border-top: 4px solid #818cf8; }
    .phase-card.actor { border-top: 4px solid #a78bfa; }
    .phase-card.critic { border-top: 4px solid #34d399; }
    .phase-icon { font-size: 2rem; margin-bottom: 8px; }
    .phase-title { font-weight: 600; color: #f3f4f6; margin: 0; font-size: 1rem; }
    .phase-desc { font-size: 0.85rem; color: #9ca3af; margin: 8px 0 0 0; }
    
    .decision-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 16px 32px;
        border-radius: 100px;
        font-weight: 600;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .decision-hold { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
    .decision-nudge { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; }
    .decision-brief { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; }
    .decision-escalate { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; }
    
    .finding-card {
        background: rgba(30, 30, 50, 0.9);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
        color: #e5e5e5;
    }
    .finding-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
    .finding-high { border-left-color: #ef4444; }
    .finding-medium { border-left-color: #f59e0b; }
    .finding-low { border-left-color: #10b981; }
    .finding-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .finding-title { font-weight: 600; color: #f3f4f6; font-size: 1rem; }
    .finding-id { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #9ca3af; background: rgba(60,60,90,0.8); padding: 4px 8px; border-radius: 6px; }
    
    .confidence-meter { width: 100%; height: 6px; background: #2d2d4a; border-radius: 3px; overflow: hidden; margin-top: 8px; }
    .confidence-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
    
    .thought-bubble {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 16px 0;
        border: 1px solid rgba(99, 102, 241, 0.3);
        position: relative;
    }
    .thought-bubble::before { content: '💭'; position: absolute; top: -12px; left: 20px; font-size: 1.5rem; }
    .thought-label { font-size: 0.75rem; font-weight: 600; color: #818cf8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
    .thought-content { color: #c7d2fe; font-size: 0.95rem; line-height: 1.6; }
    
    .stat-card { background: rgba(40, 40, 70, 0.8); border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); }
    .stat-value { font-size: 1.75rem; font-weight: 700; color: #f3f4f6; }
    .stat-label { font-size: 0.8rem; color: #9ca3af; margin-top: 4px; }
    
    .timeline-connector { display: flex; justify-content: center; align-items: center; padding: 24px 0; position: relative; }
    .timeline-line { width: 2px; height: 40px; background: linear-gradient(180deg, #818cf8 0%, #a78bfa 100%); border-radius: 2px; }
    .timeline-dot { width: 12px; height: 12px; background: #a78bfa; border-radius: 50%; position: absolute; animation: glow 2s ease-in-out infinite; }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(167, 139, 250, 0.5); }
        50% { box-shadow: 0 0 20px rgba(167, 139, 250, 0.8); }
    }
    
    .output-card { background: rgba(30, 30, 50, 0.9); border-radius: 20px; padding: 24px; box-shadow: 0 4px 24px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); color: #e5e5e5; }
    .output-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.1); }
    .output-icon { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; }
    .output-icon.nudge { background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(37, 99, 235, 0.3) 100%); }
    .output-icon.quest { background: linear-gradient(135deg, rgba(245, 158, 11, 0.3) 0%, rgba(217, 119, 6, 0.3) 100%); }
    .output-icon.brief { background: linear-gradient(135deg, rgba(251, 146, 60, 0.3) 0%, rgba(234, 88, 12, 0.3) 100%); }
    .output-icon.escalate { background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(220, 38, 38, 0.3) 100%); }
    
    .safety-approved { background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%); border: 2px solid #10b981; border-radius: 16px; padding: 24px; text-align: center; }
    .safety-rejected { background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%); border: 2px solid #ef4444; border-radius: 16px; padding: 24px; }
    .safety-icon { font-size: 3rem; margin-bottom: 12px; }
    .safety-title { font-size: 1.25rem; font-weight: 700; }
    
    .ethics-pill { display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; padding: 8px 16px; border-radius: 100px; font-size: 0.85rem; font-weight: 500; margin: 4px; transition: all 0.2s ease; }
    .ethics-pill:hover { background: rgba(16, 185, 129, 0.3); transform: scale(1.02); }
    
    .section-header { display: flex; align-items: center; gap: 12px; margin: 32px 0 16px 0; }
    .section-number { width: 32px; height: 32px; background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem; }
    .section-title { font-size: 1.25rem; font-weight: 600; color: #f3f4f6; }
    
    .rationale-card { background: linear-gradient(135deg, rgba(234, 179, 8, 0.15) 0%, rgba(202, 138, 4, 0.1) 100%); border-radius: 16px; padding: 20px; border: 1px solid rgba(234, 179, 8, 0.3); }
    .rationale-label { font-size: 0.75rem; font-weight: 600; color: #fbbf24; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
    .rationale-text { color: #fcd34d; font-size: 0.95rem; line-height: 1.6; }
    
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%); }
    section[data-testid="stSidebar"] * { color: #e5e5e5 !important; }
    
    .stButton > button { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; border: none; border-radius: 12px; padding: 12px 32px; font-weight: 600; font-size: 1rem; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5); }
    
    [data-testid="stMetricValue"] { font-weight: 700; color: #f3f4f6 !important; }
    [data-testid="stMetricLabel"] { color: #9ca3af !important; }
    
    [data-testid="stExpander"] { background: rgba(30, 30, 50, 0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; }
    [data-testid="stExpander"] summary { color: #e5e5e5 !important; }
    
    .stDataFrame { background: rgba(30, 30, 50, 0.8) !important; }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in { animation: slideIn 0.5s ease forwards; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False
if "result" not in st.session_state:
    st.session_state.result = None
if "graph" not in st.session_state:
    st.session_state.graph = None


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0;">
        <h2 style="margin: 0; font-weight: 700; color: #f3f4f6;">✨ Orchestrator</h2>
        <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 4px;">Set up your analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("##### 📊 Where's the data coming from?")
    data_source = st.radio("Source type", ["Preset Scenario", "JSON File"], index=0, label_visibility="collapsed")
    
    if data_source == "Preset Scenario":
        scenario = st.selectbox(
            "Scenario",
            ["normal_week", "stressed_week", "recovery_week"],
            index=1,
            format_func=lambda x: {"normal_week": "🟢 A Normal Week", "stressed_week": "🔴 A Tough Week", "recovery_week": "🟡 Getting Better"}[x]
        )
        
        st.markdown("")
        st.markdown("##### 📅 How far back to look?")
        time_range = st.selectbox(
            "Time Range",
            ["last_week", "last_2_weeks", "last_month"],
            index=0,
            label_visibility="collapsed",
            format_func=lambda x: {"last_week": "📆 Last Week (7 days)", "last_2_weeks": "📅 Last 2 Weeks (14 days)", "last_month": "🗓️ Last Month (30 days)"}[x]
        )
    else:
        uploaded_file = st.file_uploader("Upload JSON", type=["json"])
        time_range = "last_week"  # Default for JSON uploads
    
    st.markdown("")
    st.markdown("##### 🎨 Pick a fun theme")
    theme = st.selectbox(
        "Theme",
        ["pet_care", "adventure", "garden", "space", "cozy"],
        label_visibility="collapsed",
        format_func=lambda x: {"pet_care": "🐾 Pet Care", "adventure": "⚔️ Adventure", "garden": "🌱 Garden", "space": "🚀 Space", "cozy": "🏠 Cozy"}[x]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        pet_name = st.text_input("Pet name", "Luna", label_visibility="collapsed", placeholder="Pet name")
    with col2:
        character_name = st.text_input("Character", "Star", label_visibility="collapsed", placeholder="Character")
    
    st.markdown("")
    st.markdown("##### 📋 Who can we reach out to?")
    school_opt_in = st.toggle("School counselor can help", value=False)
    pro_opt_in = st.toggle("Professional support available", value=False)
    parent_ack = st.toggle("Parent is in the loop", value=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.15); padding: 16px; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
        <p style="font-weight: 600; color: #34d399; margin: 0 0 8px 0; font-size: 0.85rem;">🛡️ Our Promise</p>
        <p style="color: #6ee7b7; margin: 0; font-size: 0.8rem; line-height: 1.5;">
            No diagnosing • No labels<br>No spying • Always positive
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Support Orchestrator</h1>
    <p class="hero-subtitle">See exactly how I think through helping young people thrive</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="phase-card planner"><div class="phase-icon">📋</div><p class="phase-title">Step 1: I Look at the Data</p><p class="phase-desc">What patterns am I noticing?</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="phase-card actor"><div class="phase-icon">⚡</div><p class="phase-title">Step 2: I Take Action</p><p class="phase-desc">What should I do about it?</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="phase-card critic"><div class="phase-icon">🛡️</div><p class="phase-title">Step 3: I Double-Check</p><p class="phase-desc">Is this safe and helpful?</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_button = st.button("🚀 Let's See What's Going On", type="primary", use_container_width=True)

if run_button:
    if not st.session_state.llm_initialized:
        with st.spinner("Warming up the brain..."):
            try:
                init_llm(model="gpt-4o-mini", temperature=0.7)
                st.session_state.llm_initialized = True
            except Exception as e:
                st.error(f"Oops! Couldn't start the AI: {e}")
                st.stop()
    
    if st.session_state.graph is None:
        with st.spinner("Setting things up..."):
            st.session_state.graph = build_graph()
    
    consent = ConsentSettings(school_opt_in=school_opt_in, pro_opt_in=pro_opt_in, parent_acknowledged=parent_ack)
    child_prefs = ChildPreferences(preferred_categories=["breathing", "movement", "sunlight"], theme=theme, pet_name=pet_name, character_name=character_name)
    
    if data_source == "Preset Scenario":
        metrics = generate_metrics(scenario, time_range=time_range)
    else:
        if uploaded_file:
            data = json.load(uploaded_file)
            metrics = [DailyMetrics(**d) for d in data.get("metrics", [])]
        else:
            st.warning("Please upload a JSON file")
            st.stop()
    
    initial_state = OrchestratorState(
        metrics=metrics,
        scenario=scenario if data_source == "Preset Scenario" else "custom",
        child_preferences=child_prefs,
        consent=consent
    )
    
    # Node descriptions for human-readable progress
    node_descriptions = {
        "signal_interpreter": ("🔍 Analyzing patterns...", 15),
        "planner": ("🤔 Deciding what to do...", 30),
        "hold_actor": ("⏸️ Preparing hold response...", 55),
        "nudge_actor": ("💬 Creating fun nudge...", 55),
        "brief_actor": ("📋 Writing parent briefing...", 55),
        "escalate_actor": ("🔔 Preparing escalation...", 55),
        "safety_critic": ("🛡️ Running safety checks...", 80),
        "revision_handler": ("✏️ Making revisions...", 85),
    }
    
    progress = st.progress(0, text="Starting up...")
    
    # Stream the graph execution to show real-time progress
    # Accumulate all state updates into final_state
    # Convert to dict for LangGraph compatibility
    initial_state_dict = initial_state.model_dump()
    final_state = initial_state_dict.copy()
    
    for event in st.session_state.graph.stream(initial_state_dict):
        # event is a dict with the node name as key and its output as value
        for node_name, node_output in event.items():
            if node_name in node_descriptions:
                desc, pct = node_descriptions[node_name]
                progress.progress(pct, text=desc)
            # Merge node output into final state
            if isinstance(node_output, dict):
                final_state.update(node_output)
    
    progress.progress(100, text="✅ All done!")
    time.sleep(0.3)
    progress.empty()
    st.session_state.result = final_state
    st.toast("All done! Take a look below 👇", icon="✅")


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

if st.session_state.result:
    result = st.session_state.result
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dynamic title based on number of days
    num_days = len(result.get("metrics", []))
    time_label = f"{num_days} days"
    if num_days == 7:
        time_label = "Last Week (7 days)"
    elif num_days == 14:
        time_label = "Last 2 Weeks (14 days)"
    elif num_days == 30:
        time_label = "Last Month (30 days)"
    
    with st.expander(f"📊 **The Data I Looked At** — {time_label}", expanded=False):
        if result.get("metrics"):
            import pandas as pd
            metrics = result["metrics"]
            
            # Row 1: Key wellness metrics
            st.markdown("##### 💤 Sleep & Wellness")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_sleep = sum(m.sleep_hours if hasattr(m, 'sleep_hours') else m.get('sleep_hours', 0) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_sleep:.1f}h</div><div class="stat-label">Avg Sleep</div></div>""", unsafe_allow_html=True)
            with col2:
                avg_activity = sum((m.physical_activity_minutes if hasattr(m, 'physical_activity_minutes') else m.get('physical_activity_minutes', 0)) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_activity:.0f}m</div><div class="stat-label">Avg Activity</div></div>""", unsafe_allow_html=True)
            with col3:
                avg_outdoor = sum((m.outdoor_time_minutes if hasattr(m, 'outdoor_time_minutes') else m.get('outdoor_time_minutes', 0)) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_outdoor:.0f}m</div><div class="stat-label">Outdoor Time</div></div>""", unsafe_allow_html=True)
            with col4:
                bedtime_rate = sum(1 for m in metrics if (m.bedtime_target_met if hasattr(m, 'bedtime_target_met') else m.get('bedtime_target_met', True))) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{bedtime_rate:.0%}</div><div class="stat-label">Bedtime Met</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Row 2: Screen time metrics
            st.markdown("##### 📱 Screen Time")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_screen = sum((m.screen_time_minutes if hasattr(m, 'screen_time_minutes') else m.get('screen_time_minutes', 0)) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_screen:.0f}m</div><div class="stat-label">Daily Avg</div></div>""", unsafe_allow_html=True)
            with col2:
                avg_late = sum((m.late_night_screen_minutes if hasattr(m, 'late_night_screen_minutes') else m.get('late_night_screen_minutes', 0)) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_late:.0f}m</div><div class="stat-label">Late Night</div></div>""", unsafe_allow_html=True)
            with col3:
                avg_pickups = sum((m.device_pickups if hasattr(m, 'device_pickups') else m.get('device_pickups', 0)) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_pickups:.0f}</div><div class="stat-label">Pickups/Day</div></div>""", unsafe_allow_html=True)
            with col4:
                avg_notifications = sum((m.notifications_received if hasattr(m, 'notifications_received') else m.get('notifications_received', 0)) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_notifications:.0f}</div><div class="stat-label">Notifications</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Row 3: App category breakdown
            st.markdown("##### 🎮 App Categories (Weekly Total)")
            total_social = sum((m.app_usage.social_media_minutes if hasattr(m, 'app_usage') else m.get('app_usage', {}).get('social_media_minutes', 0)) for m in metrics)
            total_games = sum((m.app_usage.games_minutes if hasattr(m, 'app_usage') else m.get('app_usage', {}).get('games_minutes', 0)) for m in metrics)
            total_entertainment = sum((m.app_usage.entertainment_minutes if hasattr(m, 'app_usage') else m.get('app_usage', {}).get('entertainment_minutes', 0)) for m in metrics)
            total_education = sum((m.app_usage.education_minutes if hasattr(m, 'app_usage') else m.get('app_usage', {}).get('education_minutes', 0)) for m in metrics)
            total_productivity = sum((m.app_usage.productivity_minutes if hasattr(m, 'app_usage') else m.get('app_usage', {}).get('productivity_minutes', 0)) for m in metrics)
            total_all = total_social + total_games + total_entertainment + total_education + total_productivity
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                pct = (total_social / total_all * 100) if total_all > 0 else 0
                st.markdown(f"""<div class="stat-card"><div class="stat-value" style="font-size: 1.2rem;">📱 {total_social}m</div><div class="stat-label">Social ({pct:.0f}%)</div></div>""", unsafe_allow_html=True)
            with col2:
                pct = (total_games / total_all * 100) if total_all > 0 else 0
                st.markdown(f"""<div class="stat-card"><div class="stat-value" style="font-size: 1.2rem;">🎮 {total_games}m</div><div class="stat-label">Games ({pct:.0f}%)</div></div>""", unsafe_allow_html=True)
            with col3:
                pct = (total_entertainment / total_all * 100) if total_all > 0 else 0
                st.markdown(f"""<div class="stat-card"><div class="stat-value" style="font-size: 1.2rem;">🎬 {total_entertainment}m</div><div class="stat-label">Videos ({pct:.0f}%)</div></div>""", unsafe_allow_html=True)
            with col4:
                pct = (total_education / total_all * 100) if total_all > 0 else 0
                st.markdown(f"""<div class="stat-card"><div class="stat-value" style="font-size: 1.2rem;">📚 {total_education}m</div><div class="stat-label">Learning ({pct:.0f}%)</div></div>""", unsafe_allow_html=True)
            with col5:
                pct = (total_productivity / total_all * 100) if total_all > 0 else 0
                st.markdown(f"""<div class="stat-card"><div class="stat-value" style="font-size: 1.2rem;">⚙️ {total_productivity}m</div><div class="stat-label">Productivity ({pct:.0f}%)</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Row 4: Engagement metrics
            st.markdown("##### 🎯 Engagement & Routine")
            col1, col2 = st.columns(2)
            with col1:
                avg_routine = sum((m.routine_consistency_score if hasattr(m, 'routine_consistency_score') else m.get('routine_consistency_score', 0)) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_routine:.0%}</div><div class="stat-label">Routine Consistency</div></div>""", unsafe_allow_html=True)
            with col2:
                avg_checkin = sum((m.checkin_completion if hasattr(m, 'checkin_completion') else m.get('checkin_completion', 0)) for m in metrics) / len(metrics)
                st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_checkin:.0%}</div><div class="stat-label">Check-in Rate</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show raw data table
            st.markdown("##### 📅 Daily Breakdown")
            # Flatten the data for display
            display_data = []
            for m in metrics:
                if hasattr(m, 'model_dump'):
                    d = m.model_dump()
                else:
                    d = m
                # Flatten app_usage
                app = d.pop('app_usage', {})
                d['social_min'] = app.get('social_media_minutes', 0)
                d['games_min'] = app.get('games_minutes', 0)
                d['edu_min'] = app.get('education_minutes', 0)
                display_data.append(d)
            
            df = pd.DataFrame(display_data)
            # Reorder columns for readability
            col_order = ['date', 'sleep_hours', 'screen_time_minutes', 'late_night_screen_minutes', 
                        'device_pickups', 'bedtime_target_met', 'routine_consistency_score', 
                        'checkin_completion', 'physical_activity_minutes', 'outdoor_time_minutes',
                        'social_min', 'games_min', 'edu_min']
            available_cols = [c for c in col_order if c in df.columns]
            df = df[available_cols]
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # SIGNAL ANALYSIS - More human-readable
    st.markdown("""<div class="section-header"><div class="section-number">1</div><span class="section-title">What I Noticed</span></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="thought-bubble"><div class="thought-label">What I'm Doing Here</div><div class="thought-content">I'm looking at the daily patterns to see if anything changed recently. I compare this week to what's typical — just observing, not judging.</div></div>""", unsafe_allow_html=True)
    
    if result.get("signal_output"):
        signal = result["signal_output"]
        findings = signal.findings if hasattr(signal, 'findings') else signal.get("findings", [])
        
        if findings:
            st.markdown(f"**I spotted {len(findings)} thing(s) worth noting:**")
            for finding in findings:
                f_data = finding.model_dump() if hasattr(finding, 'model_dump') else finding
                severity = f_data.get("severity", "low")
                confidence = f_data.get("confidence", 0)
                conf_color = "#10b981" if confidence > 0.8 else "#f59e0b" if confidence > 0.6 else "#ef4444"
                evidence_html = "".join([f'<div class="stat-card" style="flex: 1; min-width: 100px;"><div class="stat-value" style="font-size: 1rem;">{v}</div><div class="stat-label">{k.replace("_", " ").title()}</div></div>' for k, v in f_data.get('evidence', {}).items()])
                
                st.markdown(f"""
                <div class="finding-card finding-{severity}">
                    <div class="finding-header">
                        <span class="finding-title">{f_data.get('title', 'Unknown')}</span>
                        <span class="finding-id">{f_data.get('finding_id', '')}</span>
                    </div>
                    <div style="display: flex; gap: 16px; flex-wrap: wrap;">{evidence_html}</div>
                    <div class="confidence-meter"><div class="confidence-fill" style="width: {confidence*100}%; background: {conf_color};"></div></div>
                    <div style="text-align: right; font-size: 0.75rem; color: #9ca3af; margin-top: 4px;">How sure am I: {confidence:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Everything looks pretty normal — no big changes to flag! 🎉")
    
    st.markdown("""<div class="timeline-connector"><div class="timeline-line"></div><div class="timeline-dot"></div></div>""", unsafe_allow_html=True)
    
    # PLANNER DECISION - More human-readable
    st.markdown("""<div class="section-header"><div class="section-number">2</div><span class="section-title">What Should I Do About It?</span></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="thought-bubble"><div class="thought-label">What I'm Thinking</div><div class="thought-content">Based on what I noticed, I need to decide the best next step. Sometimes the right choice is to do nothing — that's totally valid! I always pick the gentlest option that makes sense.</div></div>""", unsafe_allow_html=True)
    
    if result.get("planner_output"):
        planner = result["planner_output"]
        p_data = planner.model_dump() if hasattr(planner, 'model_dump') else planner
        plan = p_data.get("plan", "UNKNOWN")
        confidence = p_data.get("confidence", 0)
        
        # Human-readable decision names
        plan_labels = {"HOLD": "Just Keep Watching", "NUDGE_CHILD": "Send a Fun Nudge", "BRIEF_PARENT": "Update the Parent", "ESCALATE": "Get Some Help"}
        plan_class = {"HOLD": "hold", "NUDGE_CHILD": "nudge", "BRIEF_PARENT": "brief", "ESCALATE": "escalate"}.get(plan, "hold")
        plan_icon = {"HOLD": "⏸️", "NUDGE_CHILD": "💬", "BRIEF_PARENT": "📋", "ESCALATE": "🔔"}.get(plan, "❓")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""<div style="text-align: center; padding: 24px;"><span class="decision-badge decision-{plan_class}">{plan_icon} {plan_labels.get(plan, plan)}</span></div>""", unsafe_allow_html=True)
        with col2:
            st.metric("How Sure", f"{confidence:.0%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="rationale-card"><div class="rationale-label">📝 Why I Chose This</div><div class="rationale-text">{p_data.get('rationale', 'N/A')}</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="rationale-card"><div class="rationale-label">⚖️ Why Not Something Bigger?</div><div class="rationale-text">{p_data.get('why_not_stronger', 'N/A')}</div></div>""", unsafe_allow_html=True)
        
        st.markdown(f"""<div class="thought-bubble" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(167, 139, 250, 0.1) 100%); border-color: rgba(167, 139, 250, 0.3);"><div class="thought-label" style="color: #a78bfa;">🔍 What the Data Told Me</div><div class="thought-content" style="color: #c4b5fd;">{p_data.get('signal_summary', 'N/A')}</div></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="timeline-connector"><div class="timeline-line"></div><div class="timeline-dot"></div></div>""", unsafe_allow_html=True)
    
    # ACTOR OUTPUT - More human-readable
    st.markdown("""<div class="section-header"><div class="section-number">3</div><span class="section-title">Here's What I Prepared</span></div>""", unsafe_allow_html=True)
    
    plan = ""
    if result.get("planner_output"):
        p = result["planner_output"]
        plan = p.plan if hasattr(p, 'plan') else p.get("plan", "")
    
    actor_tasks = {
        "HOLD": "Since things look okay, I'll explain why taking no action is actually the best move right now. Sometimes watchful waiting is the wisest choice!",
        "NUDGE_CHILD": f"I'm creating a friendly, fun message using the {theme} theme. It's designed to feel like a game, not like homework!",
        "BRIEF_PARENT": "I'm putting together a warm, supportive summary for the parent — helpful info without being alarming or invasive.",
        "ESCALATE": "I'm preparing clear notes so the right people can help. This is done carefully and respectfully."
    }
    st.markdown(f"""<div class="thought-bubble"><div class="thought-label">What I'm Creating</div><div class="thought-content">{actor_tasks.get(plan, 'Working on the response...')}</div></div>""", unsafe_allow_html=True)
    
    if plan == "HOLD" and result.get("hold_output"):
        hold = result["hold_output"]
        h_data = hold.model_dump() if hasattr(hold, 'model_dump') else hold
        st.markdown(f"""<div class="output-card"><div class="output-header"><div class="output-icon" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.3) 100%);">⏸️</div><div><strong style="font-size: 1.1rem; color: #f3f4f6;">Taking a Step Back</strong><p style="margin: 0; color: #9ca3af; font-size: 0.9rem;">Sometimes the best action is no action</p></div></div><p style="color: #e5e5e5;"><strong>Why I'm waiting:</strong> {h_data.get('rationale', 'N/A')}</p><p style="color: #e5e5e5;"><strong>What I observed:</strong> {h_data.get('signals_observed', 'N/A')}</p><p style="color: #34d399;"><strong>When I'll check again:</strong> {h_data.get('next_check_recommendation', 'N/A')}</p></div>""", unsafe_allow_html=True)
    
    elif plan == "NUDGE_CHILD" and result.get("youth_nudge_output"):
        nudge_out = result["youth_nudge_output"]
        n_data = nudge_out.model_dump() if hasattr(nudge_out, 'model_dump') else nudge_out
        nudge = n_data.get("nudge", {})
        quest = n_data.get("micro_quest", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="output-card"><div class="output-header"><div class="output-icon nudge">💬</div><div><strong style="font-size: 1.1rem; color: #f3f4f6;">{nudge.get('title', 'Nudge')}</strong><p style="margin: 0; color: #9ca3af; font-size: 0.85rem;">{nudge.get('category', 'general')}</p></div></div><p style="font-size: 1rem; line-height: 1.6; color: #e5e5e5;">{nudge.get('text', '')}</p><hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 16px 0;"><p style="font-size: 0.85rem; color: #9ca3af;"><strong>Why I'm showing this:</strong> {nudge.get('why_shown', 'N/A')}</p><p style="font-size: 0.8rem; color: #6b7280; font-style: italic;">{nudge.get('opt_out_hint', '')}</p></div>""", unsafe_allow_html=True)
        with col2:
            steps_html = "".join([f"<li style='margin: 8px 0; color: #e5e5e5;'>{step}</li>" for step in quest.get('steps', [])])
            st.markdown(f"""<div class="output-card"><div class="output-header"><div class="output-icon quest">🎯</div><div><strong style="font-size: 1.1rem; color: #f3f4f6;">{quest.get('title', 'Mini Quest')}</strong><p style="margin: 0; color: #9ca3af; font-size: 0.85rem;">⏱️ About {quest.get('duration_minutes', 5)} minutes</p></div></div><ol style="padding-left: 20px; margin: 0;">{steps_html}</ol></div>""", unsafe_allow_html=True)
        
        driven_by = n_data.get("driven_by_findings", [])
        if driven_by:
            st.markdown(f"**🔗 This was triggered by:** `{'`, `'.join(driven_by)}`")
    
    elif plan == "BRIEF_PARENT" and result.get("parent_briefing_output"):
        brief = result["parent_briefing_output"]
        b_data = brief.model_dump() if hasattr(brief, 'model_dump') else brief
        
        # Build changes, actions, and activities lists
        changes_list = b_data.get("top_changes", [])
        actions_list = b_data.get("suggested_actions", [])
        activities_list = b_data.get("suggested_activities", [])
        if not activities_list:
            activities_list = ["Take a walk together", "Cook their favorite meal"]
        
        # Pre-build HTML strings
        changes_html = "".join([f'<p style="margin: 6px 0; color: #e5e5e5;">• {c}</p>' for c in changes_list])
        actions_html = "".join([f'<p style="margin: 6px 0; color: #e5e5e5;">• {a}</p>' for a in actions_list])
        activities_html = "".join([f'<p style="margin: 6px 0; color: #e5e5e5;">• {a}</p>' for a in activities_list])
        
        weekly_summary = b_data.get('weekly_summary', 'N/A')
        conversation_starter = b_data.get('conversation_starter', 'N/A')
        
        html_content = f'''<div class="output-card"><div class="output-header"><div class="output-icon brief">📋</div><div><strong style="font-size: 1.1rem; color: #f3f4f6;">Message for Parent</strong><p style="margin: 0; color: #9ca3af; font-size: 0.85rem;">A friendly update</p></div></div><p style="font-size: 1rem; line-height: 1.6; color: #e5e5e5; margin-bottom: 16px;">{weekly_summary}</p><div style="background: rgba(245, 158, 11, 0.15); padding: 16px; border-radius: 12px; margin: 16px 0; border: 1px solid rgba(245, 158, 11, 0.3);"><p style="margin: 0; font-weight: 600; color: #fbbf24;">💬 A Way to Start the Conversation</p><p style="margin: 8px 0 0 0; color: #fcd34d; font-style: italic;">"{conversation_starter}"</p></div><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 20px;"><div><p style="font-weight: 600; color: #f3f4f6; margin-bottom: 8px;">📈 The Biggest Changes</p>{changes_html}</div><div><p style="font-weight: 600; color: #f3f4f6; margin-bottom: 8px;">💡 Some Ideas to Try</p>{actions_html}</div><div><p style="font-weight: 600; color: #f3f4f6; margin-bottom: 8px;">🎉 Fun Activities Together</p>{activities_html}</div></div></div>'''
        
        st.markdown(html_content, unsafe_allow_html=True)
    
    elif plan == "ESCALATE" and result.get("escalation_output"):
        esc = result["escalation_output"]
        e_data = esc.model_dump() if hasattr(esc, 'model_dump') else esc
        level = e_data.get("level", "parent_only")
        level_labels = {"parent_only": "Parent Only", "school_counselor": "School Counselor", "professional": "Professional Support"}
        st.markdown(f"""<div class="output-card"><div class="output-header"><div class="output-icon escalate">🔔</div><div><strong style="font-size: 1.1rem; color: #f3f4f6;">Time to Get Some Help</strong><p style="margin: 0; color: #9ca3af; font-size: 0.85rem;">Reaching out to: {level_labels.get(level, level)}</p></div></div><p style="color: #e5e5e5;"><strong>Should we escalate?</strong> {"✓ Yes" if e_data.get('escalate') else "✗ No"}</p><p style="color: #e5e5e5;"><strong>Here's why:</strong> {e_data.get('reason', 'N/A')}</p><p style="color: #e5e5e5;"><strong>Suggested next step:</strong> {e_data.get('recommended_next_step', 'N/A')}</p><p style="color: {'#f87171' if e_data.get('requires_parent_ack') else '#9ca3af'};"><strong>Parent needs to acknowledge:</strong> {"Yes" if e_data.get('requires_parent_ack') else "No"}</p></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="timeline-connector"><div class="timeline-line"></div><div class="timeline-dot"></div></div>""", unsafe_allow_html=True)
    
    # SAFETY CRITIC - More human-readable
    st.markdown("""<div class="section-header"><div class="section-number">4</div><span class="section-title">Safety Check</span></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="thought-bubble"><div class="thought-label">What I'm Checking For</div><div class="thought-content">Before anything goes out, I double-check that it's safe and helpful. No medical talk, no scary labels, no invasive stuff — just positive, supportive content.</div></div>""", unsafe_allow_html=True)
    
    # Display safety history if available
    if result.get("safety_history"):
        safety_history = result["safety_history"]
        if isinstance(safety_history, list) and len(safety_history) > 0:
            st.markdown(f"""<div style="background: rgba(99, 102, 241, 0.15); padding: 16px; border-radius: 12px; margin-bottom: 16px; border: 1px solid rgba(99, 102, 241, 0.3);"><p style="margin: 0; font-weight: 600; color: #818cf8; font-size: 0.9rem;">🔄 Critic Review History ({len(safety_history)} iteration{"s" if len(safety_history) > 1 else ""})</p></div>""", unsafe_allow_html=True)
            
            for idx, historical_safety in enumerate(safety_history):
                hist_data = historical_safety.model_dump() if hasattr(historical_safety, 'model_dump') else historical_safety
                iteration_num = hist_data.get("iteration", idx)
                approved = hist_data.get("approved", False)
                violations = hist_data.get("violations", [])
                
                iteration_label = "Initial Review" if iteration_num == 0 else f"Revision {iteration_num}"
                status_icon = "✅" if approved else "⚠️"
                status_color = "#34d399" if approved else "#f87171"
                
                with st.expander(f"{status_icon} {iteration_label} - {'Approved' if approved else f'{len(violations)} Issue(s) Found'}", expanded=(idx == len(safety_history) - 1)):
                    if approved:
                        st.markdown(f"""<div style="color: {status_color}; padding: 12px; border-radius: 8px; background: rgba(52, 211, 153, 0.1);">✓ This iteration passed all safety checks!</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Issues found in iteration {iteration_num}:**")
                        for v in violations:
                            st.markdown(f"- {v}")
                        if hist_data.get("revision_instructions"):
                            st.markdown(f"""<div style="background: rgba(245, 158, 11, 0.15); padding: 12px; border-radius: 8px; margin-top: 12px; border: 1px solid rgba(245, 158, 11, 0.3); color: #fcd34d;"><strong>Revision Instructions:</strong> {hist_data.get("revision_instructions")}</div>""", unsafe_allow_html=True)
    
    if result.get("safety_output"):
        safety = result["safety_output"]
        s_data = safety.model_dump() if hasattr(safety, 'model_dump') else safety
        approved = s_data.get("approved", False)
        violations = s_data.get("violations", [])
        iteration_num = s_data.get("iteration", 0)
        
        iteration_text = f" (Iteration {iteration_num})" if iteration_num > 0 else ""
        
        if approved:
            st.markdown(f"""<div class="safety-approved"><div class="safety-icon">✅</div><div class="safety-title" style="color: #34d399;">All Good!{iteration_text}</div><p style="color: #6ee7b7; margin: 8px 0 0 0;">Everything passed my safety and ethics review</p></div>""", unsafe_allow_html=True)
        else:
            violations_html = "".join([f'<p style="color: #fca5a5; margin: 8px 0;">• {v}</p>' for v in violations])
            revision_html = f'<p style="color: #fcd34d; background: rgba(245, 158, 11, 0.15); padding: 12px; border-radius: 8px; margin-top: 16px; border: 1px solid rgba(245, 158, 11, 0.3);"><strong>How to fix it:</strong> {s_data.get("revision_instructions")}</p>' if s_data.get("revision_instructions") else ''
            st.markdown(f"""<div class="safety-rejected"><div class="safety-icon">⚠️</div><div class="safety-title" style="color: #f87171;">Hmm, Found Some Issues{iteration_text}</div>{violations_html}{revision_html}</div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        ethics = ["No Diagnosing", "No Labels", "No Spying", "Strength-Based", "Privacy First"]
        pills_html = "".join([f'<span class="ethics-pill">✓ {e}</span>' for e in ethics])
        st.markdown(f'<div style="text-align: center;">{pills_html}</div>', unsafe_allow_html=True)
    
    # EXPLAINABILITY - More human-readable
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""<div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); padding: 32px; border-radius: 20px; color: white;"><h3 style="margin: 0 0 16px 0; color: white;">📖 Why Am I Seeing This?</h3>""", unsafe_allow_html=True)
    
    try:
        why_result = why_am_i_seeing_this(OrchestratorState(**result))
        st.markdown(f"""<p style="color: rgba(255,255,255,0.9); line-height: 1.8;">{why_result.get('explanation', 'No explanation available')}</p></div>""", unsafe_allow_html=True)
    except:
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.expander("🔧 **Technical Details** — The raw data (for developers)", expanded=False):
        json_result = {}
        for key, value in result.items():
            if value is None:
                json_result[key] = None
            elif hasattr(value, 'model_dump'):
                json_result[key] = value.model_dump()
            elif isinstance(value, list):
                json_result[key] = [v.model_dump() if hasattr(v, 'model_dump') else v for v in value]
            else:
                json_result[key] = value
        st.json(json_result)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 24px 0;">
    <p style="margin: 0;">Support Orchestrator v1.0 • Powered by GPT-4o-mini • Built with LangGraph + Streamlit</p>
    <p style="margin: 8px 0 0 0; font-size: 0.8rem; color: #9ca3af;">🛡️ No diagnosing • No labels • No spying • Always positive</p>
</div>
""", unsafe_allow_html=True)
