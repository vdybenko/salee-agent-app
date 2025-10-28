import streamlit as st
import pandas as pd
from google.cloud import bigquery
import json

st.set_page_config(layout="wide")

client = bigquery.Client()

query = """
SELECT
  userId,
  userLinkedinId,
  participantLinkedinId,
  chatId,
  topicId,
  conversationType,
  labels,
  topicSummary,
  topicKeywords,
  topicEmbeddingVector,
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
  firstConversationMessageAt,
  lastConversationMessageAt,
  processedAt,
  raw_excerpt
FROM `salee-chrome-extention.SaleeAgent.conversations_embedded`
ORDER BY lastTopicMessageAt DESC
"""

df = client.query(query).to_dataframe()

# Normalize topicKeywords to list[str]
if 'topicKeywords' in df.columns:
	def _normalize_keywords(v):
		if v is None:
			return []
		if isinstance(v, (list, tuple)):
			return [str(x) for x in v]
		# numpy array
		try:
			import numpy as _np
			if isinstance(v, _np.ndarray):
				return [str(x) for x in v.tolist()]
		except Exception:
			pass
		# try JSON string
		if isinstance(v, str):
			try:
				data = json.loads(v)
				if isinstance(data, list):
					return [str(x) for x in data]
			except Exception:
				# fallback: comma-separated
				return [s.strip() for s in v.split(',') if s.strip()]
		return [str(v)]
	
	df['topicKeywords'] = df['topicKeywords'].apply(_normalize_keywords)

# Initialize session state for selected topic
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = None

# Custom CSS for LinkedIn-style interface
st.markdown("""
<style>
.main-container {
    display: flex;
    height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.conversation-list {
    width: 35%;
    border-right: 1px solid #e0e0e0;
    overflow-y: auto;
    background-color: #f8f9fa;
}

.conversation-item {
    padding: 16px;
    border-bottom: 1px solid #e0e0e0;
    cursor: pointer;
    transition: background-color 0.2s;
    display: flex;
    align-items: center;
}

.conversation-item:hover {
    background-color: #e8f4fd;
}

.conversation-item.selected {
    background-color: #0073b1;
    color: white;
}

.avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background-color: #0073b1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    margin-right: 12px;
    font-size: 18px;
}

.conversation-info {
    flex: 1;
}

.conversation-title {
    font-weight: 600;
    margin-bottom: 4px;
    font-size: 14px;
}

.conversation-preview {
    color: #666;
    font-size: 12px;
    margin-bottom: 4px;
}

.conversation-meta {
    font-size: 11px;
    color: #999;
}

.conversation-detail {
    width: 65%;
    padding: 24px;
    overflow-y: auto;
    background-color: white;
}

.detail-header {
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 16px;
    margin-bottom: 24px;
}

.detail-title {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 8px;
}

.detail-meta {
    color: #666;
    font-size: 14px;
}

.detail-section {
    margin-bottom: 24px;
}

.detail-section h3 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #333;
}

.detail-content {
    background-color: #f8f9fa;
    padding: 16px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.5;
}

.keywords {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.keyword-tag {
    background-color: #e3f2fd;
    color: #1976d2;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}

.keyword-tag.product-feedback {
    background-color: #fff3e0;
    color: #f57c00;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 16px;
}

.stat-card {
    background-color: #f8f9fa;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
}

.stat-value {
    font-size: 24px;
    font-weight: 600;
    color: #0073b1;
}

.stat-label {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}

.linkedin-button {
    background-color: #0073b1;
    color: white;
    padding: 12px 24px;
    border: none;
    border-radius: 6px;
    text-decoration: none;
    display: inline-block;
    font-weight: 500;
    margin-top: 16px;
}

.linkedin-button:hover {
    background-color: #005885;
    color: white;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# Create the main interface
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Left panel - Topic list
st.markdown('<div class="conversation-list">', unsafe_allow_html=True)
st.markdown('<h2 style="padding: 16px; margin: 0; background-color: #0073b1; color: white;">Topics</h2>', unsafe_allow_html=True)

for idx, row in df.iterrows():
    chat_id = row['chatId']
    topic_id = row['topicId']
    summary = row['topicSummary'][:100] + "..." if len(str(row['topicSummary'])) > 100 else row['topicSummary']
    date_col = row['lastTopicMessageAt'] if pd.notna(row.get('lastTopicMessageAt', None)) else row['lastConversationMessageAt']
    date = pd.to_datetime(date_col).strftime('%b %d, %Y') if pd.notna(date_col) else ''
    
    # Create avatar with first letter of topic or chat
    avatar_text = str(topic_id or chat_id)[0].upper() if (topic_id or chat_id) else "?"
    
    # Check if this topic is selected
    is_selected = st.session_state.selected_topic == topic_id
    item_class = "conversation-item selected" if is_selected else "conversation-item"
    
    # Create clickable topic item
    if st.button(" ", key=f"topic_{topic_id}_{idx}", help=f"Click to view topic {topic_id}"):
        st.session_state.selected_topic = topic_id
        st.rerun()
    
    st.markdown(f"""
    <div class="{item_class}">
        <div class="avatar">{avatar_text}</div>
        <div class="conversation-info">
            <div class="conversation-title">Topic {topic_id} · Chat {chat_id}</div>
            <div class="conversation-preview">{summary}</div>
            <div class="conversation-meta">{date}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Right panel - Topic details
st.markdown('<div class="conversation-detail">', unsafe_allow_html=True)

if st.session_state.selected_topic:
	# Get the selected topic data
	selected_row = df[df['topicId'] == st.session_state.selected_topic].iloc[0]
	last_dt = selected_row['lastTopicMessageAt'] if pd.notna(selected_row.get('lastTopicMessageAt', None)) else selected_row['lastConversationMessageAt']
	st.markdown(f"""
	<div class="detail-header">
		<div class="detail-title">Topic {st.session_state.selected_topic} · Chat {selected_row['chatId']}</div>
		<div class="detail-meta">Last updated: {pd.to_datetime(last_dt).strftime('%B %d, %Y at %I:%M %p') if pd.notna(last_dt) else ''}</div>
	</div>
	""", unsafe_allow_html=True)
	
	# Conversation summary
	st.markdown('<div class="detail-section"><h3>Summary</h3><div class="detail-content">' + str(selected_row['topicSummary']) + '</div></div>', unsafe_allow_html=True)
	
	# Keywords
	keywords = selected_row['topicKeywords'] if isinstance(selected_row['topicKeywords'], (list, tuple)) else []
	if keywords:
		keyword_html_items = []
		for kw in keywords:
			kw_text = str(kw).strip()
			cls = 'keyword-tag product-feedback' if kw_text == 'product-feedback' else 'keyword-tag'
			keyword_html_items.append(f'<span class="{cls}">{kw_text}</span>')
		keyword_html = ''.join(keyword_html_items)
		st.markdown(f'<div class="detail-section"><h3>Keywords</h3><div class="keywords">{keyword_html}</div></div>', unsafe_allow_html=True)
	
	# Statistics
	st.markdown(f"""
	<div class="detail-section">
		<h3>Statistics</h3>
		<div class="stats-grid">
			<div class="stat-card">
				<div class="stat-value">{selected_row['sentMessages']}</div>
				<div class="stat-label">Messages sent</div>
			</div>
			<div class="stat-card">
				<div class="stat-value">{selected_row['receivedMessages']}</div>
				<div class="stat-label">Messages received</div>
			</div>
			<div class="stat-card">
				<div class="stat-value">{round(selected_row['conversationDuration'] / 3600, 1) if selected_row['conversationDuration'] > 0 else 0}</div>
				<div class="stat-label">Duration (hours)</div>
			</div>
            <div class="stat-card">
                <div class="stat-value">{round(selected_row['reply_ratio'], 2) if pd.notna(selected_row.get('reply_ratio', None)) else '-'}</div>
                <div class="stat-label">Reply ratio</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{round(selected_row['conversation_temperature'], 2) if pd.notna(selected_row.get('conversation_temperature', None)) else '-'}</div>
                <div class="stat-label">Temperature</div>
            </div>
		</div>
	</div>
	""", unsafe_allow_html=True)
	
	# Attributes
	attr_items = []
	def _add_attr(label, value):
		if pd.notna(value) and str(value) != '':
			attr_items.append(f"<div><b>{label}:</b> {value}</div>")
	_add_attr("Primary intent", selected_row.get('primary_intent', ''))
	_add_attr("Intent direction", selected_row.get('intent_direction', ''))
	_add_attr("Primary product/service", selected_row.get('primary_product_or_service', ''))
	_add_attr("Tone", selected_row.get('tone', ''))
	_add_attr("Relationship stage", selected_row.get('relationship_stage', ''))
	_add_attr("Next action", selected_row.get('next_action', ''))
	_add_attr("Next action date", selected_row.get('next_action_date', ''))
	_add_attr("First topic message at", selected_row.get('firstTopicMessageAt', ''))
	_add_attr("Last topic message at", selected_row.get('lastTopicMessageAt', ''))
	_add_attr("First conversation message at", selected_row.get('firstConversationMessageAt', ''))
	_add_attr("Last conversation message at", selected_row.get('lastConversationMessageAt', ''))
	if attr_items:
		st.markdown('<div class="detail-section"><h3>Attributes</h3><div class="detail-content">' + ''.join(attr_items) + '</div></div>', unsafe_allow_html=True)

	# LinkedIn link (chat scope)
	linkedin_url = f"https://www.linkedin.com/messaging/thread/{selected_row['chatId']}/"
	st.markdown(f'<a href="{linkedin_url}" target="_blank" class="linkedin-button">Open in LinkedIn</a>', unsafe_allow_html=True)
	
else:
	st.markdown("""
	<div style="text-align: center; padding: 60px 20px; color: #666;">
		<h2>Select a topic to view details</h2>
		<p>Choose a topic from the list on the left to see its details here.</p>
	</div>
	""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# JavaScript for conversation selection (placeholder)
st.markdown("""
<script>
function selectConversation(chatId) {
    // handled via Streamlit session state buttons
}
</script>
""", unsafe_allow_html=True)

# Summary statistics at the bottom
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
	st.metric("Total Conversations", len(df))
with col2:
	st.metric("Total Messages", int(df['sentMessages'].fillna(0).sum()) + int(df['receivedMessages'].fillna(0).sum()))
with col3:
	avg_dur = df['conversationDuration'].fillna(0).mean()
	st.metric("Avg Duration (hours)", round(avg_dur / 3600, 1) if avg_dur > 0 else 0)