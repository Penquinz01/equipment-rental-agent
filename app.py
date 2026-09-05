import streamlit as st
import time

# ==============================================================================
# 1. BACKEND INTEGRATION CONTRACT (PERSON B)
# ==============================================================================
try:
    from agent import run_rental_agent
    BACKEND_CONNECTED = True
    BACKEND_ERROR_MSG = ""
except Exception as import_err:
    run_rental_agent = None
    BACKEND_CONNECTED = False
    BACKEND_ERROR_MSG = str(import_err)

# Person D Integration Contract
try:
    from data_loader import log_decision
    from utils import export_log_as_text
    LOGGING_CONNECTED = True
except Exception:
    log_decision = None
    export_log_as_text = None
    LOGGING_CONNECTED = False

# ==============================================================================
# 2. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Equipment Rental Decision Agent",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 3. CUSTOM CSS STYLING (Preserve Visual Design System)
# ==============================================================================
st.markdown("""
<style>
    /* Global Container Styles */
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* Header Component */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .header-title-box {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .header-icon {
        font-size: 2.2rem;
        background: rgba(37, 99, 235, 0.2);
        padding: 0.5rem 0.75rem;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 0.875rem;
        color: #94a3b8;
        margin: 0.2rem 0 0 0;
    }

    /* Response State Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.825rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-auto-quote {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .badge-request-info {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-manual-review {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-unrecognized {
        background-color: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }

    /* Decision Card Formatting */
    .decision-card {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .decision-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid #27272a;
        margin-bottom: 1rem;
    }
    .score-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .score-value {
        font-size: 1.25rem;
        font-weight: 800;
    }
    .score-label {
        font-size: 0.75rem;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .card-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .reason-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: #e4e4e7;
        margin-bottom: 0.35rem;
    }

    /* Quote Table & Info Containers */
    .quote-box {
        background: #09090b;
        border: 1px solid #27272a;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.85rem;
    }
    .quote-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        font-size: 0.9rem;
        color: #d4d4d8;
    }
    .quote-row.total {
        border-top: 1px solid #3f3f46;
        padding-top: 0.6rem;
        margin-top: 0.4rem;
        font-weight: 700;
        font-size: 1.1rem;
        color: #38bdf8;
    }

    .sidebar-card {
        background: #18181b;
        border: 1px solid #27272a;
        padding: 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        color: #a1a1aa;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "type": "welcome",
            "content": "👋 Welcome to the **Equipment Rental Decision Agent**!\n\nSubmit your rental inquiry to receive an instant automated quotation, missing information request, or underwriter review routing.\n\n*Try typing a request or select a quick test prompt from the sidebar.*"
        }
    ]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "session_context" not in st.session_state:
    st.session_state.session_context = {}

# ==============================================================================
# 5. RENDER FUNCTIONS FOR BACKEND CONTRACT RESULTS
# ==============================================================================
def _extract_score_num(val):
    if isinstance(val, dict):
        return val.get("score", 0)
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def render_scorecard(scorecard: dict):
    """Renders the 5 scoring factors and total qualification score."""
    if not isinstance(scorecard, dict) or not scorecard:
        return
    
    st.markdown("<div class='card-section-title'>📊 Qualification Scorecard</div>", unsafe_allow_html=True)
    
    avail = _extract_score_num(scorecard.get("availability"))
    comp = _extract_score_num(scorecard.get("completeness"))
    trust = _extract_score_num(scorecard.get("customer_trust"))
    risk = _extract_score_num(scorecard.get("liability_risk"))
    timing = _extract_score_num(scorecard.get("timing"))
    total = scorecard.get("total", 0)
    max_total = scorecard.get("max_total", 100)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Availability", f"{avail} / 30")
    with col2:
        st.metric("Completeness", f"{comp} / 20")
    with col3:
        st.metric("Customer Trust", f"{trust} / 25")
    with col4:
        st.metric("Liability Risk", f"{risk} / 15")
    with col5:
        st.metric("Timing", f"{timing} / 10")

    st.markdown(f"<div style='font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin-top: 0.4rem; margin-bottom: 0.5rem;'>Total Qualification Score: {total} / {max_total}</div>", unsafe_allow_html=True)


def render_parsed_fields(parsed: dict, equipment_name=None, contractor_name=None):
    """Displays extracted inquiry fields safely under 'Request Understanding'."""
    if not isinstance(parsed, dict):
        parsed = {}
        
    eq = equipment_name or parsed.get("equipment_name") or parsed.get("equipment")
    cont = contractor_name or parsed.get("contractor_name") or parsed.get("contractor")
    dur = parsed.get("duration_days") if parsed.get("duration_days") is not None else parsed.get("duration")
    s_date = parsed.get("start_date")
    site = parsed.get("site_info") or parsed.get("site")
    urg = parsed.get("urgency")

    items = []
    if eq: items.append(f"**Equipment:** {eq}")
    if cont: items.append(f"**Contractor:** {cont}")
    if dur is not None: items.append(f"**Duration:** {dur} days" if isinstance(dur, (int, float)) else f"**Duration:** {dur}")
    if s_date: items.append(f"**Start Date:** {s_date}")
    if site: items.append(f"**Site:** {site}")
    if urg: items.append(f"**Urgency:** {urg}")

    # Safely iterate over any additional fields in parsed_fields
    known_keys = {"equipment", "equipment_name", "contractor", "contractor_name", "duration", "duration_days", "start_date", "site", "site_info", "urgency"}
    for k, v in parsed.items():
        if k not in known_keys and v:
            label = k.replace("_", " ").title()
            items.append(f"**{label}:** {v}")

    if items:
        st.markdown("<div class='card-section-title'>🔍 Request Understanding</div>", unsafe_allow_html=True)
        st.markdown(" • " + " &nbsp;|&nbsp; • ".join(items), unsafe_allow_html=True)


def render_result(result: dict):
    """Renders the natural chat response in the conversation, placing technical scores in a collapsed log."""
    decision = result.get("decision", "UNRECOGNIZED")
    reasoning = result.get("reasoning", "")
    scorecard = result.get("scorecard", {})
    parsed_fields = result.get("parsed_fields", {})
    equipment_name = result.get("equipment_name")
    contractor_name = result.get("contractor_name")

    total_score = scorecard.get("total", "N/A") if scorecard else "N/A"
    max_score = scorecard.get("max_total", 100) if scorecard else 100

    # 1. Primary Conversational Agent Message
    agent_msg = result.get("message")
    if not agent_msg:
        if decision == "REQUEST_INFO":
            missing_info = result.get("missing_info", {})
            agent_msg = missing_info.get("message") if isinstance(missing_info, dict) else "We need a few additional details to proceed with your quote."
        elif decision == "MANUAL_REVIEW":
            agent_msg = "Thank you for the project details. Because this inquiry involves high-value equipment or specialized access conditions, our underwriting team is conducting a quick manual review."
        elif decision == "AUTO_QUOTE":
            agent_msg = "Great news! Your rental request has been approved. Here is your itemized quotation:"
        else:
            agent_msg = "How can I assist you with your equipment rental needs today?"

    st.markdown(agent_msg)

    # 2. Quotation Payload (Displayed when AUTO_QUOTE is approved)
    if decision == "AUTO_QUOTE":
        quote = result.get("quote", {})
        if quote:
            st.markdown(f"""
            <div class="quote-box">
                <div class="card-section-title" style="color: #38bdf8; margin-top: 0; margin-bottom: 0.6rem;">Itemized Quotation</div>
                <div class="quote-row"><span>Equipment:</span><strong>{quote.get('equipment', equipment_name or 'N/A')}</strong></div>
                <div class="quote-row"><span>Contractor:</span><span>{contractor_name or 'N/A'}</span></div>
                <div class="quote-row"><span>Duration:</span><span>{quote.get('duration_days', 'N/A')} days</span></div>
                <div class="quote-row"><span>Daily Rate:</span><span>${quote.get('daily_rate', 0):,.2f}</span></div>
                <div class="quote-row"><span>Subtotal:</span><span>${quote.get('subtotal', 0):,.2f}</span></div>
                <div class="quote-row"><span>Delivery Fee:</span><span>${quote.get('delivery_fee', 0):,.2f}</span></div>
                <div class="quote-row"><span>Damage Waiver:</span><span>${quote.get('damage_waiver', 0):,.2f}</span></div>
                <div class="quote-row"><span>Rush Surcharge:</span><span>${quote.get('rush_surcharge', 0):,.2f}</span></div>
                <div class="quote-row"><span>Loyalty Discount ({quote.get('tier', 'Standard')} Tier):</span><span style="color: #4ade80;">-${quote.get('loyalty_discount', 0):,.2f}</span></div>
                <div class="quote-row"><span>Pre-Tax Total:</span><span>${quote.get('pre_tax_total', 0):,.2f}</span></div>
                <div class="quote-row"><span>Tax:</span><span>${quote.get('tax', 0):,.2f}</span></div>
                <div class="quote-row total"><span>Final Total:</span><span>${quote.get('final_total', 0):,.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)

    elif decision == "MANUAL_REVIEW":
        review_ticket = result.get("review_ticket", {})
        priority = review_ticket.get("priority", "Standard") if isinstance(review_ticket, dict) else "Standard"
        st.info(f"📋 **Review Ticket Submitted** (Priority: {priority}) — Assigned for underwriter evaluation.")

    # 3. Technical Underwriting Log (Collapsed by default, preserving logs for judges/auditors)
    with st.expander("🛠️ Decision & Underwriting Log", expanded=False):
        st.markdown(f"**Decision:** `{decision}` &nbsp;|&nbsp; **Score:** `{total_score} / {max_score}`")
        if parsed_fields:
            render_parsed_fields(parsed_fields, equipment_name, contractor_name)
        matched = result.get("matched_equipment", [])
        if matched and len(matched) > 1:
            st.markdown(f"**All Matched Fleet Machines:** {', '.join(matched)}")
        unavail = result.get("unavailable_equipment", [])
        if unavail:
            st.markdown(f"**Unavailable Items (Not in Fleet):** {', '.join(unavail)}")
        alts = result.get("recommended_alternatives", {})
        if alts:
            st.markdown("**Recommended Fleet Alternatives:**")
            for k, v in alts.items():
                st.markdown(f"- *{k}*: {v}")
        if scorecard:
            render_scorecard(scorecard)
        if decision == "MANUAL_REVIEW":
            review_ticket = result.get("review_ticket", {})
            if isinstance(review_ticket, dict):
                triggers = review_ticket.get("triggers", [])
                if triggers:
                    st.markdown("**Risk Triggers:**")
                    for t in triggers:
                        st.markdown(f"- 🛑 {t}")
        if reasoning:
            st.markdown("**Chain-of-Thought Reasoning:**")
            st.markdown(reasoning)



# ==============================================================================
# 6. SIDEBAR COMPONENT
# ==============================================================================
with st.sidebar:
    st.markdown("### 🚜 Rental Decision Agent")
    st.markdown("<p style='color: #71717a; font-size: 0.8rem;'>Person C — Streamlit Frontend Interface</p>", unsafe_allow_html=True)
    st.divider()

    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "type": "welcome",
                "content": "👋 Welcome to the **Equipment Rental Decision Agent**!\n\nSubmit your rental inquiry to receive an instant automated quotation, missing information request, or underwriter review routing.\n\n*Try typing a request or select a quick test prompt from the sidebar.*"
            }
        ]
        st.session_state.session_context = {}
        st.rerun()

    # Active Multi-Turn Session State Inspector
    if st.session_state.get("session_context"):
        with st.expander("🧠 Active Session Memory", expanded=False):
            st.json(st.session_state.session_context)

    # Person D Export Chat Log Button
    if LOGGING_CONNECTED and export_log_as_text is not None:
        st.download_button(
            "📥 Export Chat Log",
            data=export_log_as_text(),
            file_name="rental_audit_log.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()

    # Quick Test Prompts (Exact Judge Test Prompts)
    st.markdown("#### 💡 Quick Test Prompts")
    
    if st.button("1. AUTO-QUOTE Prompt", use_container_width=True):
        st.session_state.pending_prompt = "Need the mini excavator for 5 days starting Sept 15, site is our usual compacted lot, our operator's HEO cert is on file. From Ferreira Builders LLC."
        st.rerun()
        
    if st.button("2. REQUEST_INFO Prompt", use_container_width=True):
        st.session_state.pending_prompt = "Need a scissor lift for a job next week, maybe 4-5 days, not sure on exact dates yet."
        st.rerun()
        
    if st.button("3. MANUAL_REVIEW Prompt", use_container_width=True):
        st.session_state.pending_prompt = "Need the 50-ton mobile crane for a demolition job starting tomorrow morning, 3 days, downtown site with tight access. This is Titan Demolition Inc."
        st.rerun()
        
    if st.button("4. UNRECOGNIZED Prompt", use_container_width=True):
        st.session_state.pending_prompt = "Do you rent bulldozers for residential lawn mowing?"
        st.rerun()

    st.divider()

    # Decision Thresholds Info Card
    st.markdown("""
    <div class="sidebar-card">
        <strong>Score Decision Thresholds:</strong>
        <ul style="padding-left: 1.2rem; margin-top: 0.4rem; margin-bottom: 0.4rem; font-size: 0.8rem;">
            <li><span style="color:#4ade80;">80+</span> → AUTO_QUOTE</li>
            <li><span style="color:#fbbf24;">50–79</span> → REQUEST_INFO</li>
            <li><span style="color:#f87171;">&lt;50</span> → MANUAL_REVIEW</li>
        </ul>
        <hr style="border-color: #27272a; margin: 0.6rem 0;"/>
        <strong>Backend Status:</strong> {status_text}<br/>
        <strong>Agent Backend:</strong> Person B's <code>run_rental_agent()</code>
    </div>
    """.format(
        status_text="<span style='color:#4ade80;'>Connected</span>" if BACKEND_CONNECTED else "<span style='color:#f87171;'>Not Loaded</span>"
    ), unsafe_allow_html=True)


# ==============================================================================
# 7. MAIN APP HEADER
# ==============================================================================
st.markdown("""
<div class="header-container">
    <div class="header-title-box">
        <div class="header-icon">🚜</div>
        <div>
            <h1 class="header-title">Equipment Rental Decision Agent</h1>
            <p class="header-subtitle">Automated Underwriting, Instant Quotation & Risk Assessment Portal</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not BACKEND_CONNECTED:
    st.warning(f"⚠️ **Backend Agent Notice**: Could not import `run_rental_agent` from `agent.py`. The interface is ready to connect once `agent.py` is present. Details: `{BACKEND_ERROR_MSG}`")


# ==============================================================================
# 8. CHAT MESSAGES DISPLAY (HISTORIC PERSISTENCE)
# ==============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            if msg.get("type") == "welcome":
                st.markdown(msg["content"])
            elif "result" in msg:
                render_result(msg["result"])
            else:
                st.write(msg.get("content", ""))


# ==============================================================================
# 9. CHAT INPUT & BACKEND INTEGRATION EXECUTION
# ==============================================================================
# Capture user input from st.chat_input or sidebar pending quick prompt
user_input = st.chat_input("Ask for an equipment rental quote or enter application details...")

if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input:
    # 1. Render and record user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2. Call Person B's backend agent
    with st.chat_message("assistant"):
        with st.spinner("Processing rental inquiry through Decision Agent..."):
            if BACKEND_CONNECTED and run_rental_agent is not None:
                try:
                    # Construct chat_history for multi-turn conversational context
                    chat_history = []
                    for m in st.session_state.messages:
                        if m.get("role") == "user" and m.get("content"):
                            chat_history.append({"role": "user", "content": m["content"]})
                        elif m.get("role") == "assistant":
                            res = m.get("result", {})
                            msg_text = res.get("message") or m.get("content", "")
                            if msg_text:
                                chat_history.append({"role": "assistant", "content": msg_text})

                    session_ctx = st.session_state.get("session_context", {})
                    result = run_rental_agent(
                        user_input,
                        chat_history=chat_history,
                        session_context=session_ctx,
                    )
                    if isinstance(result, dict):
                        # Persist accumulated multi-turn state
                        st.session_state.session_context = result.get("session_context", {})
                        render_result(result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "result": result
                        })
                        # Person D Decision Logging Integration
                        if LOGGING_CONNECTED and log_decision is not None:
                            try:
                                log_decision(user_input, result)
                            except Exception:
                                pass  # Ensure logging failure never blocks UI rendering
                    else:
                        st.error("Invalid response format received from backend agent.")
                except Exception as err:
                    st.error(f"Backend Agent execution error: {str(err)}")
            else:
                st.error("Backend agent is not connected. Please ensure `agent.py` with `run_rental_agent()` is available.")
