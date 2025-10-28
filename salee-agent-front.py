import json
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st
from google.cloud import bigquery


st.set_page_config(page_title="Salee Agent Conversations", layout="wide")


@st.cache_data(ttl=300)
def load_conversation_data(limit: int = 30) -> pd.DataFrame:
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
                lastTopicMessageAt,
                conversationDuration,
                primary_intent,
                relationship_stage,
                labels,
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
            rc.lastTopicMessageAt,
            rc.conversationDuration,
            rc.primary_intent,
            rc.relationship_stage,
            rc.labels,
            acc.firstName,
            acc.lastName,
            acc.title,
            acc.country,
            acc.city,
            acc.url
        FROM ranked_conversations rc
        LEFT JOIN `salee-chrome-extention.salee.linked_in_accounts` acc
        ON rc.participantLinkedinId = acc.id
        WHERE rc.row_number = 1
        ORDER BY rc.lastTopicMessageAt DESC
        LIMIT @limit
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    )

    return client.query(query, job_config=job_config).to_dataframe()


def _ensure_iterable_labels(raw_labels: Optional[object]) -> List[str]:
    """Normalize the labels column into a list of strings."""
    if raw_labels is None or (isinstance(raw_labels, float) and pd.isna(raw_labels)):
        return []

    if isinstance(raw_labels, (list, tuple, set)):
        return [str(label).strip() for label in raw_labels if str(label).strip()]

    if hasattr(raw_labels, "tolist"):
        return [str(label).strip() for label in raw_labels.tolist() if str(label).strip()]

    if isinstance(raw_labels, str):
        try:
            data = json.loads(raw_labels)
            if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
                return [str(label).strip() for label in data if str(label).strip()]
        except json.JSONDecodeError:
            pass
        return [label.strip() for label in raw_labels.split(",") if label.strip()]

    return [str(raw_labels).strip()] if str(raw_labels).strip() else []


def _initials(first_name: Optional[str], last_name: Optional[str]) -> str:
    initials = "".join(
        part[0].upper()
        for part in [first_name or "", last_name or ""]
        if part
    )
    return initials or "?"


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


def _shorten_text(text: Optional[str], width: int = 90) -> str:
    if not text or pd.isna(text):
        return ""
    stripped = str(text).strip()
    if len(stripped) <= width:
        return stripped
    return stripped[: width - 1].rstrip() + "…"


def _build_sidebar(label_counts: dict) -> str:
    default_labels = [
        ("🔥", "Hot"),
        ("⭐", "New"),
        ("📈", "Investors"),
        ("💡", "Usecases"),
        ("🧑‍💼", "Hiring"),
        ("🚫", "Junk"),
    ]

    items = []
    if label_counts:
        sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
        for icon, (label, count) in zip(["🔥", "⭐", "📈", "💡", "🧑‍💼", "🚫"], sorted_labels):
            items.append(
                f"<li><span class=\"icon\">{icon}</span>{label}<span class=\"count\">{count}</span></li>"
            )
        if len(items) < len(default_labels):
            for icon, label in default_labels[len(items):]:
                items.append(
                    f"<li class=\"muted\"><span class=\"icon\">{icon}</span>{label}<span class=\"count\">0</span></li>"
                )
    else:
        for icon, label in default_labels:
            items.append(
                f"<li class=\"muted\"><span class=\"icon\">{icon}</span>{label}<span class=\"count\">0</span></li>"
            )

    return "".join(items)


def _build_conversation_items(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        full_name = " ".join(filter(None, [row.get("firstName"), row.get("lastName")])).strip()
        display_name = full_name or "Unknown contact"
        initials = _initials(row.get("firstName"), row.get("lastName"))
        role_parts = [
            row.get("title"),
            ", ".join(filter(None, [row.get("city"), row.get("country")])),
        ]
        role_text = " • ".join([part for part in role_parts if part])
        preview = _shorten_text(row.get("topicSummary") or row.get("raw_excerpt"))
        relative_time = _format_relative_time(row.get("lastTopicMessageAt"))
        total_messages = (row.get("sentMessages") or 0) + (row.get("receivedMessages") or 0)

        rows.append(
            f"""
            <div class=\"conversation-item\">
                <div class=\"avatar\">{initials}</div>
                <div class=\"conversation-body\">
                    <div class=\"conversation-header\">
                        <div>
                            <div class=\"name\">{display_name}</div>
                            <div class=\"meta\">{role_text}</div>
                        </div>
                        <div class=\"time\">{relative_time}</div>
                    </div>
                    <div class=\"message\">{preview or 'No recent summary available.'}</div>
                    <div class=\"stats\">
                        <span class=\"pill\">{total_messages} messages</span>
                        {f"<span class='pill neutral'>{row.get('primary_intent')}</span>" if row.get('primary_intent') else ''}
                        {f"<span class='pill outline'>{row.get('relationship_stage')}</span>" if row.get('relationship_stage') else ''}
                    </div>
                </div>
            </div>
            """
        )
    return "".join(rows)


def _build_styles() -> str:
    return """
        <style>
            body, .block-container {
                background-color: #f5f7fb;
                padding: 0;
            }
            .block-container {
                padding-top: 2rem;
                padding-left: 2.5rem;
                padding-right: 2.5rem;
            }
            .app-shell {
                background-color: #ffffff;
                border-radius: 24px;
                box-shadow: 0 20px 60px rgba(15, 23, 42, 0.12);
                display: grid;
                grid-template-columns: 220px 1fr;
                gap: 0;
                overflow: hidden;
                min-height: 80vh;
            }
            .sidebar {
                background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
                padding: 32px 24px;
                border-right: 1px solid #e5ecf6;
            }
            .sidebar h2 {
                font-size: 18px;
                margin-bottom: 16px;
                color: #1f2937;
            }
            .sidebar ul {
                list-style: none;
                padding: 0;
                margin: 0;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .sidebar li {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 10px 12px;
                border-radius: 12px;
                font-size: 14px;
                color: #1f2937;
                font-weight: 500;
                transition: all 0.2s ease;
                box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.2);
            }
            .sidebar li:hover {
                transform: translateX(4px);
                box-shadow: inset 0 0 0 1px #2563eb;
            }
            .sidebar li.muted {
                color: #94a3b8;
                box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.15);
            }
            .sidebar .icon {
                margin-right: 10px;
                font-size: 16px;
            }
            .sidebar .count {
                background-color: #2563eb;
                color: #ffffff;
                border-radius: 999px;
                padding: 2px 8px;
                font-size: 12px;
                font-weight: 600;
            }
            .sidebar li.muted .count {
                background-color: #e2e8f0;
                color: #64748b;
            }
            .content {
                padding: 32px 40px;
                display: flex;
                flex-direction: column;
                gap: 32px;
            }
            .header {
                display: flex;
                align-items: center;
                gap: 16px;
            }
            .header img {
                width: 48px;
                height: 48px;
                border-radius: 12px;
            }
            .header h1 {
                margin: 0;
                font-size: 28px;
                color: #111827;
            }
            .header p {
                margin: 0;
                color: #6b7280;
                font-size: 15px;
            }
            .conversation-list {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .conversation-item {
                background-color: #f8fbff;
                border-radius: 18px;
                padding: 18px 20px;
                display: flex;
                gap: 16px;
                align-items: flex-start;
                border: 1px solid #e5ecf6;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .conversation-item:hover {
                transform: translateY(-3px);
                box-shadow: 0 12px 32px rgba(37, 99, 235, 0.12);
                border-color: #bfdbfe;
            }
            .avatar {
                width: 52px;
                height: 52px;
                border-radius: 16px;
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #ffffff;
                font-weight: 700;
                font-size: 18px;
                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.25);
                flex-shrink: 0;
            }
            .conversation-body {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .conversation-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 12px;
            }
            .name {
                font-size: 18px;
                font-weight: 600;
                color: #0f172a;
            }
            .meta {
                font-size: 13px;
                color: #64748b;
                margin-top: 4px;
            }
            .time {
                font-size: 13px;
                color: #2563eb;
                font-weight: 600;
            }
            .message {
                font-size: 15px;
                color: #1f2937;
                line-height: 1.5;
            }
            .stats {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }
            .pill {
                background-color: rgba(37, 99, 235, 0.12);
                color: #1d4ed8;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 600;
            }
            .pill.neutral {
                background-color: rgba(16, 185, 129, 0.12);
                color: #047857;
            }
            .pill.outline {
                background-color: transparent;
                color: #475569;
                border: 1px solid #cbd5f5;
            }
            @media (max-width: 960px) {
                .app-shell {
                    grid-template-columns: 1fr;
                }
                .sidebar {
                    display: none;
                }
                .block-container {
                    padding: 0 1.25rem 2rem;
                }
            }
        </style>
    """


def main() -> None:
    try:
        conversations = load_conversation_data()
    except Exception as exc:  # pragma: no cover - Streamlit error reporting
        st.error("Unable to load conversation data. Please verify your data source configuration.")
        st.stop()

    conversations["label_list"] = conversations["labels"].apply(_ensure_iterable_labels)

    label_counts: dict = {}
    for labels in conversations["label_list"]:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    st.markdown(_build_styles(), unsafe_allow_html=True)

    sidebar_html = _build_sidebar(label_counts)
    conversation_items_html = _build_conversation_items(conversations)

    st.markdown(
        f"""
        <div class=\"app-shell\">
            <aside class=\"sidebar\">
                <h2>Labels</h2>
                <ul>{sidebar_html}</ul>
            </aside>
            <section class=\"content\">
                <div class=\"header\">
                    <img src=\"https://res2.weblium.site/res/63aad091b5bd9f000db82b0b/66d596109d1d3227fab0bfda_optimized_376.webp\" alt=\"Salee Agent logo\" />
                    <div>
                        <h1>Salee Agent</h1>
                        <p>AI-Powered B2B Sales Assistant</p>
                    </div>
                </div>
                <div class=\"conversation-list\">
                    {conversation_items_html or '<p>No conversations available.</p>'}
                </div>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
