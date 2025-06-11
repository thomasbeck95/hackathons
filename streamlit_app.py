# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown] id="J5PoPAgwXukD"
# ## Cardiff NLP Hackathon 2025 - Starter Code
#
# Welcome to Cardiff NLP's second hackathon! Below is some code to get started on the AMPLYFI API and look at some data.
#
# ====================
#
# Note: the API is a real time resource so extra points to projects that can treat it as a continual data stream rather than a one-off data source!
#
# Another thing to note about this is that it will affect Amplyfi's servers if you download a silly amount of data. We ask that you only request 100 results per request, but if you have the data you need, try to download it or store it as a variable rather than requesting the exact same data over and over again.

# %% colab={"base_uri": "https://localhost:8080/"} id="lAyXKlRkXfT1" outputId="d08ec716-5ad7-4202-f970-9b5b762d1bff"
# Import some libraries

import requests
import json
import nltk
import re
import pandas as pd
import streamlit as st
from collections import defaultdict, Counter
from transformers import pipeline
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# %% [markdown] id="3nWf0GTmkF1-"
# Amplyfi have provided some limits and explanations of what you can query the API for below:
#
# `query_text` anything
#
# `result_size` <=100
#
# `include_highlights` boolean (if True, you get sentences matching keyphrases in the query)
#
# `include_smart_tags` boolean (if True, you get back metadata from our "smart" market intel classifiers - signals and industries)
#
# `ai_answer` can only accept "basic", this will take the 5 most relevant docs and answer the query_text based on them
#
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="pB9UJXhho0XV" outputId="f7ec73a6-1f8a-45bd-bb70-44f349392e43"
# API endpoint from the newly deployed service

API_URL = "https://zfgp45ih7i.execute-api.eu-west-1.amazonaws.com/sandbox/api/search"
API_KEY = "XYZ38746G38B7RB46GBER"

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

stop_words = set(stopwords.words('english'))

# logo 

st.image("logo.png", use_container_width=True)

# User Input Field
query_text = st.text_input("Enter a search query:")
# Button to trigger API call
if st.button("Analyze"):
    # Prepare API payload

    # Edit the below to get different data
    payload = {
    "query_text": query_text,
    "result_size": 100,
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    json_response = response.json()
    grouped_results = defaultdict(list)
    grouped_results2 = defaultdict(list)

    for item in json_response['results']:
        if not item["timestamp"]:
            continue
        else:
            date = item["timestamp"].split("T")[0]
            grouped_results[date].append(item["summary"])
            grouped_results2[date].append(item)
    
    grouped_results = dict(grouped_results)
    grouped_results2 = dict(grouped_results2)

    #sorted_group_counts = {date: len(grouped_results[date]) for date in sorted(grouped_results)}

    #print(json.dumps(grouped_results, indent=4))

 

    # tony code
    df = pd.json_normalize(json_response['results'])
    daily_dfs = {}
    for date, articles in grouped_results2.items():
        daily_dfs[date] = pd.json_normalize(articles)



    # %% id="sN-TP0mPqEmu"
    ## Clean data

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

    df['clean_summary'] = df['summary'].apply(clean_text)
    for daily_df in daily_dfs.values():
        daily_df['clean_summary'] = daily_df['summary'].apply(clean_text)


    # %% colab={"base_uri": "https://localhost:8080/"} id="Le9c5zKPpSQh" outputId="fdf22455-3610-4156-82f2-45591e3abbdd"
    ## Sentiment analysis example

    sia = SentimentIntensityAnalyzer()

    def get_sentiment_scores(text):
        """
        Returns a dict with these keys:
        - neg: negative sentiment score
        - neu: neutral score
        - pos: positive score
        - compound: normalized, weighted composite (-1 to +1)
        """
        return sia.polarity_scores(text)
    
    def remove_stopword(text, q):
        if not isinstance(text, str):
            return ""
        custom_stopwords = stop_words.copy()  
        custom_stopwords.update(q.lower().split())  
        words = text.lower().split()
        filtered = [word for word in words if word not in custom_stopwords]
        return " ".join(filtered)


    df['clean_summary'] = df['summary'].apply(clean_text)
    df['published_date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    df['sentiment'] = df['clean_summary'].apply(get_sentiment_scores)
    df['sent_compound'] = df['sentiment'].apply(lambda d: d['compound'])
    df['clean_text'] = df['clean_summary'].apply(lambda x: remove_stopword(x, query_text))



    daily_best = df.groupby('published_date').apply(lambda g: g.loc[g['sent_compound'].idxmax()]).reset_index(drop=True)
    daily_worst = df.groupby('published_date').apply(lambda g: g.loc[g['sent_compound'].idxmin()]).reset_index(drop=True)

    # Apply to each summary
    for daily_df in daily_dfs.values():
        daily_df['sentiment'] = daily_df['clean_summary'].apply(get_sentiment_scores)

    # Split into separate columns if you like
    for daily_df in daily_dfs.values():
        daily_df['sent_neg'] = daily_df['sentiment'].apply(lambda d: d['neg'])
        daily_df['sent_neu'] = daily_df['sentiment'].apply(lambda d: d['neu'])
        daily_df['sent_pos'] = daily_df['sentiment'].apply(lambda d: d['pos'])
        daily_df['sent_compound'] = daily_df['sentiment'].apply(lambda d: d['compound'])
    

    # Quick look at top 5 compound scores
    #print(df1[['clean_summary', 'sent_compound', 'timestamp']].sort_values(by='sent_compound', ascending=False).head())
    #print(df1[['clean_summary', 'sent_compound', 'timestamp']].sort_values(by='sent_compound').head())

    def todate(ts):
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d")

    dfs = list(daily_dfs.values())
    sorted_dates = sorted(daily_dfs.keys())
    dfs = [daily_dfs[date] for date in sorted_dates]




    x_data = sorted_dates
    y_data = [df['sent_compound'].tolist() for df in dfs]
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
    st.plotly_chart(fig)

    # Sentiment over time



    #Word Cloud
    # Basic word frequency from summaries
    words = " ".join(df['clean_text']).split()
    word_freq = dict(Counter(words).most_common(50))

    # Word cloud (requires matplotlib backend)
    wc = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(word_freq)
    st.subheader("🧠 Top Words in Summaries")
    st.image(wc.to_image())


    #Laura's code
    #pipe = pipeline("summarization", model="facebook/bart-large-cnn", max_length=30)
    
    #Maybe a faster summariser?
    pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", max_length=30, do_sample=True)

    # daily_summaries = []

    query_results = json.dumps(grouped_results, indent=4)
    # print(grouped_results)

    for day in grouped_results:
        to_summarize = ""
        summaries = grouped_results[day] #['summary1', 'summary2'...]
        for summary in summaries:
            to_summarize += summary
        # print(to_summarize)

    sorted_by_day = sorted(grouped_results.keys())

    #summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    for day in sorted_by_day:
        summaries = grouped_results[day]
        st.write(day)
        to_summarize = " ".join(summaries)[:2024]  # Limit input size
        summary = pipe(to_summarize, max_length=100, num_beams=2, do_sample=False)
        st.write(summary[0]["summary_text"])




    # %% colab={"base_uri": "https://localhost:8080/"} id="_9SPSX-rqYFd" outputId="a56961a0-22fd-42d1-8845-df085b7e319f"
    # Find index of the most positive (max compound) and most negative (min compound) summaries
    st.write("\n📅 Best & Worst News Articles Each Day:\n")

    for date in sorted(df['published_date'].dropna().unique()):
        st.write(f"\n📅 Date: {date}")
        
        best = daily_best[daily_best['published_date'] == date].iloc[0]
        st.write(f"\n✅ Most Positive Article:")
        st.write(f"🔹 Title: {best['title']}")
        st.write(f"📝 Summary: {best['summary'][:300]}{'...' if len(best['summary']) > 300 else ''}")
        st.write(f"🙂 Sentiment Score: {round(best['sent_compound'], 3)}")
        st.write(f"🔗 URL: {best['url']}")

        worst = daily_worst[daily_worst['published_date'] == date].iloc[0]
        st.write(f"\n❌ Most Negative Article:")
        st.write(f"🔹 Title: {worst['title']}")
        st.write(f"📝 Summary: {worst['summary'][:300]}{'...' if len(worst['summary']) > 300 else ''}")
        st.write(f"🙁 Sentiment Score: {round(worst['sent_compound'], 3)}")
        st.write(f"🔗 URL: {worst['url']}")
        
        st.write("\n" + "=" * 80 + "\n")
    # %% [markdown] id="ODtKNUO-jXV1"
    # ***My code:***
    
    
    # %% colab={"base_uri": "https://localhost:8080/", "height": 397} id="6VwDtN1IlyX6" outputId="f1c807a0-73aa-4bd2-8cf1-8c5ebb409d93"
    #import matplotlib.pyplot as plt

    # Create a scatter plot of timestamp vs sentiment compound score
    #plt.figure(figsize=(10, 6)) # Set the figure size
    #plt.scatter(df['timestamp'], df['sent_compound'])
    #plt.xlabel('Timestamp')
    #plt.ylabel('Sentiment Compound Score')
    #plt.title('Sentiment Over Time')
    #plt.grid(True)
    #plt.show()
    #st.pyplot(plt.gcf())

    # %% colab={"base_uri": "https://localhost:8080/", "height": 450, "referenced_widgets": ["111dc26ebdb346faa89eb162a5e18698", "9ac9af1a07084b3eba8b3217de9b6618", "e9182d69d08b4ea69d888af90a75af29", "dff5b7b0fba34bab844924918dfb4520", "ce7dd561fada4a7e95892ba65e4eddfe", "5aa310653b144962a4add36fdbf34ba2", "f7b43bf50df343a0a7b5bee4d778bf72"]} id="GU6Q2iY3mrIu" outputId="827eb35c-0d18-4c44-cd54-58202022074c"
    #import ipywidgets as widgets
    #from IPython.display import display
    #import matplotlib.pyplot as plt
    #import pandas as pd # Make sure pandas is imported

    # Assuming 'df' is your DataFrame containing 'timestamp' and 'sent_compound'

    # Convert timestamp to datetime with error handling
    #df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce')

    # Drop rows where the timestamp conversion failed (resulted in NaT)
    #df.dropna(subset=['timestamp'], inplace=True)

    # Sort by timestamp to ensure slider works correctly
    #df = df.sort_values(by='timestamp').reset_index(drop=True)

    # ... rest of your code for update_plot and slider


    #def update_plot(index):
        #""Updates the plot to highlight the data point at the given index.""
        #plt.figure(figsize=(10, 6))
        #plt.scatter(df['timestamp'], df['sent_compound'], label='All Results') # Plot all points
        #if 0 <= index < len(df):
            #plt.scatter(df['timestamp'].iloc[index], df['sent_compound'].iloc[index], color='red', label='Selected Result') # Highlight selected point
            #plt.title(f"Sentiment Over Time (Selected: {df['timestamp'].iloc[index].strftime('%Y-%m-%d %H:%M:%S')})")
        #else:
            #plt.title("Sentiment Over Time")

        #plt.xlabel('Timestamp')
        #plt.ylabel('Sentiment Compound Score')
        #plt.grid(True)
        #plt.legend()
        #plt.show()
        #st.pyplot(plt.gcf())

    # Create a slider widget based on the DataFrame index
    #timestamp_slider = widgets.IntSlider(
        #min=0,
        #max=len(df) - 1,
        #step=1,
        #description='Result Index:',
        #continuous_update=False,
        #orientation='horizontal',
        #readout=True,
        #readout_format='d'
    #)

    # Use interactive to link the slider to the update_plot function
    #widgets.interactive(update_plot, index=timestamp_slider)

    # %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="Z4eZtBeCruhS" outputId="4869432f-70ae-4643-bf56-c2806b8b8892"
    #import matplotlib.pyplot as plt

    # Create a box plot of the 'sent_compound' column
    #plt.figure(figsize=(8, 6)) # Set the figure size
    #plt.boxplot(df['sent_compound'])
    #plt.ylabel('Sentiment Compound Score')
    #plt.title('Distribution of Sentiment Compound Scores')
    #plt.grid(True)
    #plt.show()
    #st.pyplot(plt.gcf())

    # %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="EuPrlfHatYrf" outputId="724fac3f-4df0-4492-b517-c0084d5b7ff6"
    #import matplotlib.pyplot as plt
    #import numpy as np # Import numpy
    #import pandas as pd # Ensure pandas is imported if you haven't already

    # Make sure 'timestamp' is in datetime format
    #df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract the date
    #df['date'] = df['timestamp'].dt.date

    # Group by date and get the sentiment scores for each day
    #sentiment_by_day = df.groupby('date')['sent_compound'].apply(list)

    # Prepare data for boxplot
    # Matplotlib's boxplot expects a list of arrays or lists, where each represents a group
    #boxplot_data = [np.array(scores) for scores in sentiment_by_day.values]

    # Get dates as labels and format them
    #labels = [date.strftime('%Y-%m-%d') for date in sentiment_by_day.index]

    #plt.figure(figsize=(12, 8)) # Adjust figure size for multiple box plots
    #plt.boxplot(boxplot_data, labels=labels, patch_artist=True) # Use patch_artist for colored boxes
    #plt.xlabel('Date')
    #plt.ylabel('Sentiment Compound Score')
    #plt.title('Sentiment Distribution by Day')
    #plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
    #plt.tight_layout() # Adjust layout to prevent labels from overlapping
    #plt.grid(True, axis='y') # Add grid lines on the y-axis
    #plt.show()
    #st.pyplot(plt.gcf())
    

    # %% [markdown] id="P3uEBnGrqxlA"
    # ## Project Ideas
    #
    # Feel free to use this code to start your own project, and here are some (Chat-GPT generated 😬) ideas for projects:
    #
    # * Real-Time Sentiment Pulse: Visualize sentiment trends over the past 24-48 hours for any keyword.
    #
    # * One-Click News Brief: Generate a 3-sentence summary of today's top articles on a given topic.
    #
    # * Bias/Slant Detector: Compare headlines from multiple outlets on the same event and label their bias.
    #
    # * Event Timeline Generator: Autofill a chronological list of key dates and summaries for any query.
    #
    # * Breaking News Alert Bot: Push a short alert whenever article volume spikes or sentiment turns extreme.
    #
    # * Multilingual Hashtag Trend Mapper: Show related hashtags and translations across different languages.
    #
    # * Rumor vs. Fact Checker: Verify a user-provided statement against recent reputable sources.
    #
    # * “What's Changed?” Comparator: Highlight how coverage of a topic has shifted from last month to last week.
    #
    # * Geo-Mood Map: Color-code countries by average sentiment or topic intensity on a query.
    #
    # * Voice-Activated News Q&A: Let users speak a question and hear back a 2–3 sentence summary of current events.

    # %% [markdown] id="ZfH0xonVrTwz"
    # ## Dashboard libraries for Python
    #
    # https://shiny.posit.co/py/
    #
    # https://dash.plotly.com/
    #
    # https://streamlit.io/
    #
    #

    # %% id="j2lU_lu5pxR5"
