from datetime import datetime, timezone
from textwrap import dedent
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from google.cloud import bigquery


st.set_page_config(page_title="Salee Agent Conversations", layout="wide")


@st.cache_data(ttl=300)
def load_conversation_data(limit: int = 50, product: Optional[str] = None) -> pd.DataFrame:
    """Load the latest conversations enriched with LinkedIn profile data."""
    client = bigquery.Client()
    query = """
        WITH ranked_conversations AS (
  SELECT
    chatId,
    participantLinkedinId,
    topicSummary,
    raw_excerpt,
    sentMessages,
    receivedMessages,
    reply_ratio,
    conversationDuration,
    primary_intent,
    intent_direction,
    primary_product_or_service,
    tone,
    relationship_stage,
    conversation_temperature,
    next_action,
    next_action_date,
    firstTopicMessageAt,
    lastTopicMessageAt,
    lastConversationMessageAt,
    firstConversationMessageAt,
    ROW_NUMBER() OVER (
      PARTITION BY chatId
      ORDER BY lastTopicMessageAt DESC
    ) AS row_number
  FROM `salee-chrome-extention.SaleeAgent.conversations_embedded`
)

SELECT
  rc.chatId,
  rc.participantLinkedinId,
  rc.topicSummary,
  rc.raw_excerpt,
  rc.sentMessages,
  rc.receivedMessages,
  rc.reply_ratio,
  rc.conversationDuration,
  rc.primary_intent,
  rc.intent_direction,
  rc.primary_product_or_service,
  rc.tone,
  rc.relationship_stage,
  rc.conversation_temperature,
  rc.next_action,
  rc.next_action_date,
  rc.firstTopicMessageAt,
  rc.lastTopicMessageAt,
  rc.lastConversationMessageAt,
  rc.firstConversationMessageAt,
  acc.firstName AS participantFirstName,
  acc.lastName AS participantLastName,
  acc.title AS participantTitle,
  acc.country AS participantCountry,
  acc.city AS participantCity,
  acc.url AS participantUrl
FROM ranked_conversations rc
LEFT JOIN `salee-chrome-extention.salee.linked_in_accounts` acc
  ON rc.participantLinkedinId = acc.id
WHERE rc.row_number = 1
  AND (
    @product IS NULL
    OR EXISTS (
      SELECT 1
      FROM `salee-chrome-extention.SaleeAgent.conversations_embedded` ce
      WHERE ce.chatId = rc.chatId
        AND ce.primary_product_or_service = @product
    )
  )
ORDER BY rc.lastTopicMessageAt DESC

        LIMIT @limit
    """
    

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("product", "STRING", product),
        ]
    )

    return client.query(query, job_config=job_config).to_dataframe()


@st.cache_data(ttl=300)
def load_topics_for_chat(chat_id: str) -> pd.DataFrame:
    """Load all topics for a specific chat."""
    client = bigquery.Client()
    query = """
        SELECT
            topicId,
            topicSummary,
            raw_excerpt,
            sentMessages,
            receivedMessages,
            reply_ratio,
            conversationDuration,
            primary_intent,
            intent_direction,
            primary_product_or_service,
            tone,
            relationship_stage,
            conversation_temperature,
            next_action,
            next_action_date,
            firstTopicMessageAt,
            lastTopicMessageAt,
            labels,
            topicKeywords
        FROM `salee-chrome-extention.SaleeAgent.conversations_embedded`
        WHERE chatId = @chat_id
        ORDER BY lastTopicMessageAt DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("chat_id", "STRING", chat_id)]
    )
    return client.query(query, job_config=job_config).to_dataframe()


def _format_relative_time(value: Optional[object]) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""

    now = datetime.now(timezone.utc)
    delta = now - timestamp

    if delta.days > 365:
        years = delta.days // 365
        return f"{years} yr{'s' if years != 1 else ''} ago"
    if delta.days > 30:
        months = delta.days // 30
        return f"{months} mo{'s' if months != 1 else ''} ago"
    if delta.days > 0:
        return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"

    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours} hr{'s' if hours != 1 else ''} ago"

    minutes = (delta.seconds % 3600) // 60
    if minutes > 0:
        return f"{minutes} min{'s' if minutes != 1 else ''} ago"

    return "Just now"


def _shorten_text(text: Optional[str], width: int = 290) -> str:
    if not text or pd.isna(text):
        return ""
    stripped = str(text).strip()
    if len(stripped) <= width:
        return stripped
    return stripped[: width - 1].rstrip() + "…"


def _build_sidebar() -> str:
    labels = [
        ("🔥", "Hot"),
        ("↑", "Need Follow Up"),
        ("$", "Investors"),
        ("👥", "Colleagues"),
        ("💼", "Hiring"),
        ("🚫", "Junk"),
        ("🌱", "Ron Mai Dagun"),
    ]

    items = []
    for icon, label in labels:
        items.append(
            f"<li><span class=\"label-icon\">{icon}</span><span class=\"label-text\">{label}</span></li>"
        )

    return "".join(items)


def _build_conversation_items(df: pd.DataFrame, selected_chat_id: Optional[str] = None) -> str:
    import html as html_module
    
    rows = []
    for _, row in df.iterrows():
        chat_id = str(row.get("chatId") or "")

        full_name = " ".join(
            filter(None, [row.get("participantFirstName"), row.get("participantLastName")])
        ).strip()
        display_title = html_module.escape(row.get("participantTitle") or "")
        display_name = html_module.escape(full_name or "Unknown contact")
        preview = html_module.escape(_shorten_text(row.get("topicSummary") or row.get("raw_excerpt")))
        relative_time = _format_relative_time(row.get("lastConversationMessageAt"))
        product = html_module.escape(str(row.get("primary_product_or_service") or ""))
        next_action = html_module.escape(str(row.get("next_action") or ""))

        selected_class = "selected" if chat_id == (selected_chat_id or "") else ""
        product_span = (
            f'<span class="product">Product: {product}</span>' if product else ""
        )
        next_action_span = (
            f'<span class="next_action">Next action: {next_action}</span>' if next_action else ""
        )

        if chat_id:
            name_html = f'<a data-chat-id="{chat_id}" href="?selected_chat_id={quote_plus(chat_id)}" target="_self"><span class="name">{display_name}</span></a>'
        else:
            name_html = f'<span class="name">{display_name}</span>'

        rows.append(
            f'<div class="conversation-item {selected_class}"><div class="avatar"></div><div class="conversation-content"><div class="top-row">{name_html}<span class="title">{display_title}</span></div><div class="preview">{preview or "No recent summary available."}</div><div class="bottom-row"><span class="time">{relative_time}</span>{product_span}{next_action_span}</div></div></div>'
        )
    return "".join(rows)


def _build_topics_panel(topics_df: pd.DataFrame) -> str:
    """Build the topics detail panel HTML."""
    import html
    
    if topics_df.empty:
        return "<div class='no-topics'>No topics found for this conversation.</div>"
    
    items = []
    for _, topic in topics_df.iterrows():
        summary = html.escape(_shorten_text(topic.get("topicSummary") or topic.get("raw_excerpt"), width=200))
        relative_time = _format_relative_time(topic.get("lastTopicMessageAt"))
        intent = html.escape(str(topic.get("primary_intent") or "N/A"))
        product = html.escape(str(topic.get("primary_product_or_service") or "N/A"))
        tone = html.escape(str(topic.get("tone") or "N/A"))
        relationship_stage = html.escape(str(topic.get("relationship_stage") or "N/A"))
        conversation_temperature = html.escape(str(topic.get("conversation_temperature") or "N/A"))
        next_action = html.escape(str(topic.get("next_action") or "N/A"))
        next_action_date = html.escape(str(topic.get("next_action_date") or "N/A"))
        # Safely normalize topicKeywords which may be a numpy array, list, JSON string, or empty
        tk = topic.get("topicKeywords")
        keywords_list = []
        if isinstance(tk, (list, tuple)):
            keywords_list = [str(x) for x in tk if str(x).strip()]
        else:
            try:
                import numpy as _np  # type: ignore
                if isinstance(tk, _np.ndarray):
                    keywords_list = [str(x) for x in tk.tolist() if str(x).strip()]
                elif isinstance(tk, str):
                    try:
                        import json as _json
                        data = _json.loads(tk)
                        if isinstance(data, list):
                            keywords_list = [str(x) for x in data if str(x).strip()]
                        else:
                            keywords_list = [s.strip() for s in tk.split(',') if s.strip()]
                    except Exception:
                        keywords_list = [s.strip() for s in tk.split(',') if s.strip()]
                elif tk is None or (isinstance(tk, float) and pd.isna(tk)):
                    keywords_list = []
                else:
                    keywords_list = [str(tk)]
            except Exception:
                keywords_list = [str(tk)] if tk not in (None, '') else []
        keywords_html = ''.join(f'<span class="topic-keyword">{html.escape(kw)}</span>' for kw in keywords_list) or '<span class="topic-keyword empty">No keywords</span>'
        
        items.append(f"""<div class="topic-item">
                <div class="topic-header">
                    <span class="topic-id">{html.escape(str(topic.get("topicId", "Unknown")))}</span>
                    <span class="topic-time">{relative_time}</span>
                </div>
                <div class="topic-summary">{summary}</div>
                <div class="topic-meta">
                    <span class="topic-intent">Intent: {intent}</span>
                    <span class="topic-product">Product: {product}</span>
                    <span class="topic-tone">Tone: {tone}</span>
                    <span class="topic-relationship_stage">Relationship stage: {relationship_stage}</span>
                </div>
                <div class="topic-meta">
                    <span class="topic-conversation_temperature">Conversation temperature: {conversation_temperature}</span>
                    <span class="topic-next_action">Next action: {next_action}</span>
                    <span class="topic-next_action_date">Next action date: {next_action_date}</span>
                </div>
                <div class="topic-keywords">{keywords_html}</div>
            </div>""")
    
    return "".join(items)


def _build_styles() -> str:
    return """
<style>
            body, .block-container {
                background-color: #f5f5f5;
                padding: 0;
            }
            .app-shell-wrapper {
                background-color: #ffffff;
                border-radius: 24px;
                box-shadow: 0 12px 36px rgba(15, 23, 42, 0.12);
                display: grid;
                grid-template-columns: 220px 0.5fr;
                overflow: hidden;
                border: 1px solid #e5e7eb;
            }
            .app-shell-wrapper.with-topics {
                grid-template-columns: 220px 0.5fr 400px;
            }
            .content-wrapper {
                padding: 36px 40px;
                background-color: #ffffff;
                overflow-y: auto;
            }
            .block-container {
                padding: 32px 40px;
            }
            .sidebar {
                background-color: #fafafa;
                padding: 36px 26px;
                border-right: 1px solid #e5e7eb;
            }
            .sidebar h2 {
                font-size: 16px;
                font-weight: 600;
                color: #111827;
                letter-spacing: 0.02em;
                margin-bottom: 24px;
            }
            .sidebar ul {
                list-style: none;
                padding: 0;
                margin: 0;
                display: flex;
                flex-direction: column;
                gap: 18px;
            }
            .sidebar li {
                display: flex;
                align-items: center;
                gap: 14px;
                font-size: 15px;
                color: #374151;
            }
            .label-icon {
    display: flex;
                align-items: center;
                justify-content: center;
                width: 34px;
                height: 34px;
                border-radius: 50%;
                background-color: #e5e7eb;
                color: #1f2937;
                font-size: 18px;
                line-height: 1;
            }
            .label-text {
                font-weight: 500;
                letter-spacing: 0.01em;
            }
            .sidebar.icons-only .label-text {
                display: none;
            }
            .sidebar.icons-only h2 {
                display: none;
            }
            .sidebar.icons-only {
                padding: 36px 16px;
            }
            .app-shell-wrapper:has(.sidebar.icons-only) {
                grid-template-columns: 80px 0.5fr;
            }
            .app-shell-wrapper.with-topics {
                grid-template-columns: 220px 0.5fr 400px;
            }
            .app-shell-wrapper.with-topics:has(.sidebar.icons-only) {
                grid-template-columns: 80px 0.5fr 400px;
            }
            .content {
                padding: 36px 40px;
                background-color: #ffffff;
            }
            .header {
    display: flex;
    align-items: center;
                gap: 16px;
                margin-bottom: 28px;
            }
            .header img {
                width: 48px;
                height: 48px;
                object-fit: contain;
            }
            .header h1 {
                margin: 0;
                font-size: 24px;
                font-weight: 700;
                color: #111827;
            }
            .header h1 span {
                font-weight: 400;
                color: #6b7280;
                margin-left: 8px;
                display: inline-block;
            }
            .conversation-list {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 20px;
                overflow-y: auto;
                max-height: calc(100vh - 200px);
                display: flex;
                flex-direction: column;
            }
.avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
                background-color: #d1d5db;
                flex-shrink: 0;
            }
            .conversation-list p {
                margin: 0;
                padding: 28px;
                text-align: center;
                color: #6b7280;
                font-size: 15px;
            }
            .conversation-item {
                display: flex;
                align-items: center;
                gap: 18px;
                padding: 18px 28px;
                border-bottom: 1px solid #e5e7eb;
                transition: background-color 0.2s;
                text-decoration: none;
                color: inherit;
            }
            .conversation-item:hover {
                background-color: #f9fafb;
            }
            .conversation-item.selected {
                background-color: #eff6ff;
                border-left: 3px solid #3b82f6;
            }
            .conversation-item:last-child {
                border-bottom: none;
            }
            .conversation-content {
    flex: 1;
                display: flex;
                flex-direction: column;
                gap: 6px;
            }
            .name {
    font-size: 16px;
    font-weight: 600;
                color: #111827;
            }
            .top-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
                align-items: baseline;
            }
            .preview {
                font-size: 15px;
                color: #4b5563;
                overflow: hidden;
                text-overflow: ellipsis;
                #white-space: nowrap;
                width: 50%;
            }
            .time {
                font-size: 13px;
                color: #9ca3af;
            }
            .title {
                font-size: 14px;
                color: rgb(86 88 100 / 38%);
            }
            .bottom-row {
                display: flex;
                gap: 12px;
                align-items: center;
                flex-wrap: wrap;
            }
            .product, .next_action {
                font-size: 13px;
                color: #6b7280;
            }
            .topics-panel {
                background-color: #f9fafb;
                padding: 24px;
                border-left: 1px solid #e5e7eb;
                overflow-y: auto;
            }
            .topics-panel h3 {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 16px;
                color: #111827;
            }
            .topic-item {
                background-color: #ffffff;
                padding: 16px;
                border-radius: 8px;
                margin-bottom: 12px;
                border: 1px solid #e5e7eb;
                transition: box-shadow 0.2s;
            }
            .topic-item:hover {
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .topic-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            .topic-id {
                font-weight: 600;
                color: #3b82f6;
                font-size: 14px;
            }
            .topic-time {
                font-size: 12px;
                color: #9ca3af;
            }
            .topic-summary {
    font-size: 14px;
                color: #374151;
                margin-bottom: 8px;
                line-height: 1.5;
            }
            .topic-meta {
                display: flex;
                gap: 12px;
                font-size: 12px;
            }
            .topic-intent, .topic-product, .topic-tone, .topic-relationship_stage, 
            .topic-conversation_temperature, .topic-next_action, .topic-next_action_date, .topic-labels {
                padding: 4px 8px;
                background-color: #f3f4f6;
                border-radius: 4px;
                color: #6b7280;
                font-size: 12px;
            }
            .topic-keywords {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
                margin-top: 8px;
            }
            .topic-keyword {
    padding: 4px 8px;
                background-color: #eef2ff;
                color: #3730a3;
                border-radius: 999px;
                font-size: 12px;
                line-height: 1;
            }
            .no-topics {
                text-align: center;
                color: #9ca3af;
                padding: 40px 20px;
            }
            .hidden-buttons {
                display: none;
            }
            @media (max-width: 960px) {
                .app-shell-wrapper {
                    grid-template-columns: 1fr;
                }
                .sidebar {
                    display: none;
                }
                .block-container {
                    padding: 16px;
                }
                .content-wrapper {
                    padding: 24px;
                }
}
</style>
    """


def main() -> None:
    # Initialize session state
    if 'show_label_text' not in st.session_state:
        st.session_state.show_label_text = True
    if 'selected_chat_id' not in st.session_state:
        st.session_state.selected_chat_id = None

    # Handle product filter first before loading data
    product_filter = st.query_params.get("product") if "product" in st.query_params else ""
    
    # Add product filter selectbox at the top
    st.markdown("### Filter by Product:")
    selected_product = st.selectbox(
        "Select a product to filter conversations",
        options=["All", "TalentScan Pro", "Salee"],
        index=0 if not product_filter else (1 if product_filter == "TalentScan Pro" else 2),
        key="product_filter_select"
    )
    
    # Update URL parameter when selection changes
    if selected_product == "All":
        if "product" in st.query_params:
            del st.query_params["product"]
            st.rerun()
    else:
        st.query_params["product"] = selected_product
        if selected_product != product_filter:
            st.rerun()
    
    query_params = st.query_params
    if "selected_chat_id" in query_params:
        selected_from_query = query_params["selected_chat_id"]
        if selected_from_query != st.session_state.selected_chat_id:
            st.session_state.selected_chat_id = selected_from_query
    elif st.session_state.selected_chat_id:
        st.query_params["selected_chat_id"] = st.session_state.selected_chat_id
    else:
        st.query_params.clear()
    
    try:
        # Product filter from query params ("product"), values: "TalentScan Pro", "Salee" or empty for all
        effective_product = product_filter if product_filter in ("TalentScan Pro", "Salee") else None
        conversations = load_conversation_data(product=effective_product)
    except Exception as exc:  # pragma: no cover - Streamlit error reporting
        st.error("Unable to load conversation data. Please verify your data source configuration.")
        st.stop()

    st.markdown(_build_styles(), unsafe_allow_html=True)
    
    # Add toggle button at the top
    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("👁️", help="Toggle Label Text", key="toggle_labels", use_container_width=True):
            st.session_state.show_label_text = not st.session_state.show_label_text
            st.rerun()
    
    sidebar_html = _build_sidebar()
    
    # Add CSS class based on toggle state
    label_class = "" if st.session_state.show_label_text else "icons-only"

    # For the UI list, just render the already-filtered conversations
    conversation_items_html = _build_conversation_items(
        conversations, st.session_state.selected_chat_id
    )

    # Load topics if a chat is selected
    topics_html = ""
    with_topics_class = ""
    if st.session_state.selected_chat_id:
        try:
            topics_df = load_topics_for_chat(st.session_state.selected_chat_id)
            topics_panel_content = _build_topics_panel(topics_df)
            topics_html = f'<aside class="topics-panel"><h3>Topics ({len(topics_df)})</h3>{topics_panel_content}</aside>'
            with_topics_class = "with-topics"
        except Exception as e:
            topics_html = f'<aside class="topics-panel"><div class="no-topics">Error loading topics: {str(e)}</div></aside>'
            with_topics_class = "with-topics"

    layout_html = dedent(
        f"""
        <div class="header">
            <img src="https://res2.weblium.site/res/63aad091b5bd9f000db82b0b/66d596109d1d3227fab0bfda_optimized_376.webp" alt="Salee Agent logo" />
            <h1>Salee Agent <span>&mdash; AI-Powered B2B Sales Assistant</span></h1>
			</div>
        <div class="app-shell-wrapper {with_topics_class}">
            <aside class="sidebar {label_class}">
                <h2>Labels</h2>
                <ul>{sidebar_html}</ul>
            </aside>
            <section class="content-wrapper">
                <div class="header">
                </div>
                <div class="conversation-list">
                    {conversation_items_html or '<p>No conversations available.</p>'}
			</div>
            </section>
            {topics_html}
		</div>
        """
    )

    st.markdown(layout_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
