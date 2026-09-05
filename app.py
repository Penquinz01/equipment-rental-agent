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

# ==============================================================================
# 5. RENDER FUNCTIONS FOR BACKEND CONTRACT RESULTS
# ==============================================================================
def render_scorecard(scorecard: dict):
    """Renders the 5 scoring factors and total qualification score."""
    if not isinstance(scorecard, dict) or not scorecard:
        return
    
    st.markdown("<div class='card-section-title'>📊 Qualification Scorecard</div>", unsafe_allow_html=True)
    
    avail = scorecard.get("availability", 0)
    comp = scorecard.get("completeness", 0)
    trust = scorecard.get("customer_trust", 0)
    risk = scorecard.get("liability_risk", 0)
    timing = scorecard.get("timing", 0)
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
    """Renders the complete result dictionary returned by Person B's run_rental_agent()."""
    decision = result.get("decision", "UNRECOGNIZED")
    reasoning = result.get("reasoning", "")
    scorecard = result.get("scorecard", {})
    parsed_fields = result.get("parsed_fields", {})
    equipment_name = result.get("equipment_name")
    contractor_name = result.get("contractor_name")

    total_score = scorecard.get("total", "N/A") if scorecard else "N/A"
    max_score = scorecard.get("max_total", 100) if scorecard else 100

    # Header Badges & Colors
    if decision == "AUTO_QUOTE":
        badge_html = '<span class="badge badge-auto-quote">⚡ AUTO-QUOTE APPROVED</span>'
        score_color = "#4ade80"
    elif decision == "REQUEST_INFO":
        badge_html = '<span class="badge badge-request-info">📋 REQUEST MORE INFO</span>'
        score_color = "#fbbf24"
    elif decision == "MANUAL_REVIEW":
        badge_html = '<span class="badge badge-manual-review">⚠️ MANUAL REVIEW REQUIRED</span>'
        score_color = "#f87171"
    else: # UNRECOGNIZED
        badge_html = '<span class="badge badge-unrecognized">❓ CLARIFICATION NEEDED</span>'
        score_color = "#cbd5e1"

    # Decision Card Wrapper
    st.markdown(f"""
    <div class="decision-card">
        <div class="decision-header">
            <div>{badge_html}</div>
            <div class="score-container">
                <span class="score-label">Qualification Score:</span>
                <span class="score-value" style="color: {score_color};">{total_score} / {max_score}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Display Request Understanding (parsed_fields)
    render_parsed_fields(parsed_fields, equipment_name, contractor_name)

    # 2. Display Decision-Specific Payload
    if decision == "AUTO_QUOTE":
        quote = result.get("quote", {})
        st.success("✅ **Auto-Quote Approved** — Instant quotation calculated.")
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

    elif decision == "REQUEST_INFO":
        missing_info = result.get("missing_info", {})
        msg = missing_info.get("message", "Additional details are required to complete this quotation.") if isinstance(missing_info, dict) else "Additional details required."
        missing_fields = missing_info.get("missing_fields", []) if isinstance(missing_info, dict) else []

        st.warning(f"📋 **Request More Information** — {msg}")
        if missing_fields:
            st.markdown('<div class="quote-box" style="border-color: rgba(245, 158, 11, 0.4);"><div class="card-section-title" style="color: #fbbf24; margin-top: 0;">Missing Information Items</div>', unsafe_allow_html=True)
            if isinstance(missing_fields, list):
                for mf in missing_fields:
                    st.markdown(f'<div class="reason-item" style="color: #fbbf24;"><span>⚠️</span> <span>{mf}</span></div>', unsafe_allow_html=True)
            elif isinstance(missing_fields, dict):
                for k, v in missing_fields.items():
                    st.markdown(f'<div class="reason-item" style="color: #fbbf24;"><span>⚠️</span> <span><strong>{k}:</strong> {v}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="reason-item" style="color: #fbbf24;"><span>⚠️</span> <span>{missing_fields}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif decision == "MANUAL_REVIEW":
        review_ticket = result.get("review_ticket", {})
        if not isinstance(review_ticket, dict):
            review_ticket = {}
            
        priority = review_ticket.get("priority", "Standard")
        eq = review_ticket.get("equipment", equipment_name or "N/A")
        cont = review_ticket.get("contractor", contractor_name or "N/A")
        triggers = review_ticket.get("triggers", [])
        recommendation = review_ticket.get("recommendation", "Assigned for manual risk evaluation.")

        st.error("⚠️ **Manual Review Required** — Request escalated to underwriter.")
        st.markdown(f"""
        <div class="quote-box" style="border-color: rgba(239, 68, 68, 0.4);">
            <div class="card-section-title" style="color: #f87171; margin-top: 0;">Underwriter Review Ticket Details</div>
            <div class="quote-row"><span>Priority Level:</span><strong>{priority}</strong></div>
            <div class="quote-row"><span>Equipment Requested:</span><span>{eq}</span></div>
            <div class="quote-row"><span>Contractor:</span><span>{cont}</span></div>
            <div class="quote-row"><span>Recommended Action:</span><span>{recommendation}</span></div>
        """, unsafe_allow_html=True)

        if triggers:
            st.markdown("<div class='card-section-title' style='color: #fca5a5;'>Risk Triggers Identified</div>", unsafe_allow_html=True)
            if isinstance(triggers, list):
                for t in triggers:
                    st.markdown(f'<div class="reason-item" style="color: #fca5a5;"><span>🛑</span> <span>{t}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="reason-item" style="color: #fca5a5;"><span>🛑</span> <span>{triggers}</span></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    elif decision == "UNRECOGNIZED":
        msg = result.get("message", "I couldn't identify the equipment you're looking for. Could you specify the machine name?")
        st.info(f"❓ {msg}")

    # 3. Scorecard Display
    if scorecard:
        render_scorecard(scorecard)

    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Renamed Expander for Thought Process & Scorecard Explanation
    if reasoning:
        with st.expander("Decision Explanation & Scorecard", expanded=False):
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
        st.rerun()

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
                    result = run_rental_agent(user_input)
                    if isinstance(result, dict):
                        render_result(result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "result": result
                        })
                    else:
                        st.error("Invalid response format received from backend agent.")
                except Exception as err:
                    st.error(f"Backend Agent execution error: {str(err)}")
            else:
                st.error("Backend agent is not connected. Please ensure `agent.py` with `run_rental_agent()` is available.")
