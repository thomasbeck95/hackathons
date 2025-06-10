

import requests
import json
import nltk
import re
import pandas as pd
from collections import defaultdict
from transformers import pipeline
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# API endpoint from the newly deployed service

API_URL = "https://zfgp45ih7i.execute-api.eu-west-1.amazonaws.com/sandbox/api/search"
API_KEY = "XYZ38746G38B7RB46GBER"

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

query_text = "what is happening with riots in Los Angeles?"

# Edit the below to get different data
payload = {
  "query_text": query_text,
  "result_size": 100,
  "include_highlights":True,
  "include_smart_tags":True,

}

response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
json_response = response.json()

json_response

grouped_results = defaultdict(list)

for item in json_response['results']:

  if not item["timestamp"]:
    continue
  else:
    date = item["timestamp"].split("T")[0]

  grouped_results[date].append(item["title"])

grouped_results = dict(grouped_results)

sorted_group_counts = {date: len(grouped_results[date]) for date in sorted(grouped_results)}

print(json.dumps(grouped_results, indent=4))



pipe = pipeline("summarization", model="facebook/bart-large-cnn", max_length=30)

daily_summaries = []

query_results = json.dumps(grouped_results, indent=4)
#print(grouped_results)

for day in grouped_results:
  to_summarize = ""
  summaries = grouped_results[day] #['summary1', 'summary2'...]
  for summary in summaries:
    to_summarize += summary
  print(to_summarize)


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

for day, summaries in grouped_results.items():
    print(day)
    to_summarize = " ".join(summaries)[:2024]  # Limit input size
    summary = summarizer(to_summarize, max_length=100, num_beams=2, do_sample=False)
    print(summary[0]["summary_text"])



# tony code
df1 = pd.json_normalize(grouped_results['2025-06-04'])
df2 = pd.json_normalize(grouped_results['2025-06-05'])
df3 = pd.json_normalize(grouped_results['2025-06-06'])
df4 = pd.json_normalize(grouped_results['2025-06-07'])
df5 = pd.json_normalize(grouped_results['2025-06-08'])
df6 = pd.json_normalize(grouped_results['2025-06-09'])
df7 = pd.json_normalize(grouped_results['2025-06-10'])

def clean_text(text):
    """
    - Convert to lowercase
    - Remove URLs
    - Remove punctuation / non-alpha
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs (very basic)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Lowercase
    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

df1['clean_summary'] = df1['summary'].apply(clean_text)
df2['clean_summary'] = df2['summary'].apply(clean_text)
df3['clean_summary'] = df3['summary'].apply(clean_text)
df4['clean_summary'] = df4['summary'].apply(clean_text)
df5['clean_summary'] = df5['summary'].apply(clean_text)
df6['clean_summary'] = df6['summary'].apply(clean_text)
df7['clean_summary'] = df7['summary'].apply(clean_text)

# Split into separate columns if you like
df1['sent_neg'] = df1['sentiment'].apply(lambda d: d['neg'])
df1['sent_neu'] = df1['sentiment'].apply(lambda d: d['neu'])
df1['sent_pos'] = df1['sentiment'].apply(lambda d: d['pos'])
df1['sent_compound'] = df1['sentiment'].apply(lambda d: d['compound'])

df2['sent_neg'] = df2['sentiment'].apply(lambda d: d['neg'])
df2['sent_neu'] = df2['sentiment'].apply(lambda d: d['neu'])
df2['sent_pos'] = df2['sentiment'].apply(lambda d: d['pos'])
df2['sent_compound'] = df2['sentiment'].apply(lambda d: d['compound'])

df3['sent_neg'] = df3['sentiment'].apply(lambda d: d['neg'])
df3['sent_neu'] = df3['sentiment'].apply(lambda d: d['neu'])
df3['sent_pos'] = df3['sentiment'].apply(lambda d: d['pos'])
df3['sent_compound'] = df3['sentiment'].apply(lambda d: d['compound'])

df4['sent_neg'] = df4['sentiment'].apply(lambda d: d['neg'])
df4['sent_neu'] = df4['sentiment'].apply(lambda d: d['neu'])
df4['sent_pos'] = df4['sentiment'].apply(lambda d: d['pos'])
df4['sent_compound'] = df4['sentiment'].apply(lambda d: d['compound'])

df5['sent_neg'] = df5['sentiment'].apply(lambda d: d['neg'])
df5['sent_neu'] = df5['sentiment'].apply(lambda d: d['neu'])
df5['sent_pos'] = df5['sentiment'].apply(lambda d: d['pos'])
df5['sent_compound'] = df5['sentiment'].apply(lambda d: d['compound'])

df6['sent_neg'] = df6['sentiment'].apply(lambda d: d['neg'])
df6['sent_neu'] = df6['sentiment'].apply(lambda d: d['neu'])
df6['sent_pos'] = df6['sentiment'].apply(lambda d: d['pos'])
df6['sent_compound'] = df6['sentiment'].apply(lambda d: d['compound'])


df7['sent_neg'] = df7['sentiment'].apply(lambda d: d['neg'])
df7['sent_neu'] = df7['sentiment'].apply(lambda d: d['neu'])
df7['sent_pos'] = df7['sentiment'].apply(lambda d: d['pos'])
df7['sent_compound'] = df7['sentiment'].apply(lambda d: d['compound'])

# Quick look at top 5 compound scores
print(df1[['timestamp', 'clean_summary', 'sent_compound']].sort_values(by='sent_compound', ascending=False).head())
print(df1[['timestamp', 'clean_summary', 'sent_compound']].sort_values(by='sent_compound').head())

def todate(ts):
  return datetime.strptime(ts, "%Y-%m-%d-%T").strftime("%d-%b")

dfs = [df1, df2, df3, df4, df5, df6, df7]

x_data = [
    todate(df1['timestamp'][0][0:19]),
    todate(df2['timestamp'][0][0:19]),
    todate(df3['timestamp'][0][0:19]),
    todate(df4['timestamp'][0][0:19]),
    todate(df5['timestamp'][0][0:19]),
    todate(df6['timestamp'][0][0:19]),
    todate(df7['timestamp'][0][0:19])
    ]
data = []
y_data = [df['sent_compound'].tolist() for df in [df1, df2, df3, df4, df5, df6]]
colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)', 'rgba(255, 99, 71, 0.5)']

# Create figure
fig = go.Figure()

# Add traces for each day
for xd, yd, cls in zip(x_data, y_data, colors):
    fig.add_trace(go.Box(
        y=yd,
        name=xd,
        boxpoints='all',  # Show all points
        jitter=0.5,  # Spread points
        whiskerwidth=0.2,
        fillcolor=cls,
        marker_size=2,
        line_width=1
    ))

# Update layout
fig.update_layout(
    title=dict(text=query_text),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        zerolinecolor='green',
        zerolinewidth=3,  # Increase thickness for better visibility
        dtick=0.2,  # Adjust tick spacing for sentiment scores
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
    ),
    margin=dict(l=40, r=30, b=80, t=100),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

# Show plot
fig.show()



# Initialize lists to store results
max_scores = []
min_scores = []
max_summaries = []
min_summaries = []

# Loop through each dataframe
for i, df in enumerate(dfs, start=1):
    max_idx = df['sent_compound'].idxmax()
    min_idx = df['sent_compound'].idxmin()

    max_score = df.loc[max_idx, 'sent_compound']
    min_score = df.loc[min_idx, 'sent_compound']
    max_summary = df.loc[max_idx, 'summary']
    min_summary = df.loc[min_idx, 'summary']

    max_scores.append(max_score)
    min_scores.append(min_score)
    max_summaries.append(max_summary)
    min_summaries.append(min_summary)

    print(f"Day {i} - Most positive summary (compound = {max_score:.3f}):\n{max_summary}\n")
    print(f"Day {i} - Most negative summary (compound = {min_score:.3f}):\n{min_summary}\n")
    print("-" * 80)