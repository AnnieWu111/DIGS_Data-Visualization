import dash
from dash import dcc, html, Input, Output, State
import nltk.downloader
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import Counter
from textblob import TextBlob
import re
from wordcloud import WordCloud
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import datetime

# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger_eng')

# with open('chapter_45.txt', 'w', encoding='utf-8') as chap45:
#     for sentence in sentences:
#         chap45.write(sentence.strip() + '\n')

# Initialize the Dash app
app = dash.Dash(__name__, title="Text Analysis")

# Define the layout
block_style = {
    "padding": 20,
    "backgroundColor": "#f9f9f9",
    "borderRadius": 10,
    "marginBottom": 20,
    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
}
upload_style = {
    "width": "100%",
    "height": "60px",
    "lineHeight": "60px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "5px",
    "textAlign": "center",
    "margin": "10px 0",
    "cursor": "pointer",
}

# layout
app.layout = html.Div(
    [
        html.H1(
            "Text Analysis for Dream of the Red Chamber",
            style={"textAlign": "center", "marginBottom": 30},
        ),
        # text input
        html.Div(
            [
                html.H3("text input"),
                # file upload
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(
                        [
                            "choose txt.file to upload",
                            html.Small("(click here to upload)"),
                        ]
                    ),
                    style=upload_style,
                    multiple=False,
                    accept=".txt",
                ),
                dcc.Textarea(
                    id="text-input",
                    placeholder="...",
                    style={"width": "100%", "height": 200, "marginTop": 10},
                    value="",
                ),
                # buttons
                html.Div(
                    [
                        html.Button(
                            "Analyze",
                            id="analyze-button",
                            n_clicks=0,
                            style={
                                "marginRight": 10,
                                "backgroundColor": "#4CAF50",
                                "color": "white",
                                "border": "none",
                                "padding": "10px 20px",
                            },
                        ),
                        html.Button(
                            "Download",
                            id="download-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#2196F3",
                                "color": "white",
                                "border": "none",
                                "padding": "10px 20px",
                            },
                        ),
                        dcc.Download(id="download-text"),
                    ],
                    style={"marginTop": 10},
                ),
            ],
            style=block_style,
        ),
        # Text Statistics Summary
        html.Div(
            [
                html.H3("Statistics Summary", style={"textAlign": "center"}),
                html.Div(
                    id="summary-stats",
                    style={
                        "display": "flex",
                        "justifyContent": "space-around",
                        "flexWrap": "wrap",
                    },
                ),
            ],
            style=block_style,
        ),
        # Word Frequency
        html.Div(
            [
                html.H3("Top Words", style={"textAlign": "center"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Top 15 Words by Frequency"),
                                dcc.Graph(id="word-freq-chart"),
                            ],
                            style={"width": "60%", "padding": 10},
                        ),
                        html.Div(
                            [
                                html.H4("Word Cloud"),
                                html.Img(id="wordcloud-image", style={"width": "100%"}),
                            ],
                            style={"width": "40%", "padding": 10},
                        ),
                    ],
                    style={"display": "flex"},
                ),
            ],
            style=block_style,
        ),
        # Sentiment Analysis
        html.Div(
            [
                html.H3("Sentiment Analysis", style={"textAlign": "center"}),
                # sentiment percentage
                html.Div(
                    [
                        html.H4(
                            "Sentiment in Percentage",
                            style={"textAlign": "center", "width": "100%"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "negative",
                                            style={
                                                "padding": 10,
                                                "backgroundColor": "#f0f0f0",
                                                "textAlign": "center",
                                            },
                                        ),
                                        html.Div(
                                            id="negative-percent",
                                            style={
                                                "fontSize": 24,
                                                "color": "white",
                                                "backgroundColor": "mediumpurple",
                                                "padding": 10,
                                                "borderRadius": "0 10px 10px 0",
                                                "textAlign": "center",
                                            },
                                        ),
                                    ],
                                    style={"width": "33%"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "neutral",
                                            style={
                                                "padding": 10,
                                                "backgroundColor": "#f0f0f0",
                                                "textAlign": "center",
                                            },
                                        ),
                                        html.Div(
                                            id="neutral-percent",
                                            style={
                                                "fontSize": 24,
                                                "color": "white",
                                                "backgroundColor": "lightgray",
                                                "padding": 10,
                                                "textAlign": "center",
                                            },
                                        ),
                                    ],
                                    style={"width": "34%"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "positive",
                                            style={
                                                "padding": 10,
                                                "backgroundColor": "#f0f0f0",
                                                "textAlign": "center",
                                            },
                                        ),
                                        html.Div(
                                            id="positive-percent",
                                            style={
                                                "fontSize": 24,
                                                "color": "white",
                                                "backgroundColor": "sandybrown",
                                                "padding": 10,
                                                "borderRadius": "10px 0 0 10px",
                                                "textAlign": "center",
                                            },
                                        ),
                                    ],
                                    style={"width": "33%"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "marginTop": 20,
                                "borderRadius": 10,
                                "overflow": "hidden",
                            },
                        ),
                    ],
                    style={
                        "width": "100%",
                        "padding": 10,
                        "backgroundColor": "white",
                        "borderRadius": 10,
                        "marginTop": 20,
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Sentiment Score"),
                                dcc.Graph(id="sentiment-gauge"),
                            ],
                            style={"width": "50%", "padding": 10},
                        ),
                        html.Div(
                            [
                                html.H4("Polarity & Subjectivity"),
                                dcc.Graph(id="polarity-subjectivity"),
                            ],
                            style={"width": "50%", "padding": 10},
                        ),
                    ],
                    style={"display": "flex"},
                ),
            ],
            style=block_style,
        ),
        # Parts of Speech
        html.Div(
            [
                html.H3("Parts of Speech Distribution", style={"textAlign": "center"}),
                dcc.Graph(id="pos-chart"),
            ],
            style=block_style,
        ),
        # Sentence Analysis
        html.Div(
            [
                html.H3("Sentence Analysis", style={"textAlign": "center"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Sentence Length Distribution"),
                                dcc.Graph(id="sentence-length"),
                            ],
                            style={"width": "50%", "padding": 10},
                        ),
                        html.Div(
                            [
                                html.H4("Level of Complexity"),
                                dcc.Graph(id="complexity-chart"),
                            ],
                            style={"width": "50%", "padding": 10},
                        ),
                    ],
                    style={"display": "flex"},
                ),
            ],
            style=block_style,
        ),
    ]
)


@app.callback(
    [
        Output("summary-stats", "children"),
        Output("word-freq-chart", "figure"),
        Output("wordcloud-image", "src"),
        Output("sentiment-gauge", "figure"),
        Output("polarity-subjectivity", "figure"),
        Output("pos-chart", "figure"),
        Output("sentence-length", "figure"),
        Output("complexity-chart", "figure"),
        Output("negative-percent", "children"),
        Output("neutral-percent", "children"),
        Output("positive-percent", "children"),
    ],
    [Input("analyze-button", "n_clicks")],
    [State("text-input", "value")],
)
def analyze_text(n_clicks, text):
    if not text:
        return ["No text to analyze"], {}, None, {}, {}, {}, {}, {}

    # Basic text cleaning
    text = re.sub(r"[^\w\s]", "", text.lower())
    # Tokenization
    words = word_tokenize(text)
    sentences = [s.strip() for s in text.split("\n") if s.strip()]
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in words if word not in stop_words and len(word) > 1
    ]
    # Word frequency
    word_freq = FreqDist(filtered_words)
    top_words = word_freq.most_common(15)
    # Parts of speech tagging
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter([tag for word, tag in pos_tags])
    # Sentiment analysis with TextBlob
    blob = TextBlob(text)
    # import pdb;pdb.set_trace()
    sentiment = blob.sentiment

    sent_lengths = [len(word_tokenize(sentence)) for sentence in sentences]

    avg_word_length = (
        sum(len(word) for word in filtered_words) / len(filtered_words)
        if filtered_words
        else 0
    )

    sentiment_score = sentiment.polarity

    # Sentiment percentage calculation
    def calculate_sentiment_percentages(sentiment_score):
        positive = 0
        neutral = 0
        negative = 0
        if sentiment_score > 1e-2:
            positive = round(sentiment_score * 100, 2)
            neutral = 100 - (positive + negative)
        if sentiment_score < -1e-2:
            negative = round(abs(sentiment_score) * 100, 2)
            neutral = 100 - (positive + negative)
        if sentiment_score >= -1e-2 and sentiment_score <= 1e-2:
            neutral = 100 - (positive + negative)

        return positive, neutral, negative

    positive_pct, neutral_pct, negative_pct = calculate_sentiment_percentages(
        sentiment_score
    )

    # Create summary statistics cards
    summary_stats = [
        html.Div(
            [html.H4(f"{len(words)}"), html.P("Words")],
            style={
                "textAlign": "center",
                "padding": "10px",
                "backgroundColor": "#e3f2fd",
                "borderRadius": "5px",
                "margin": "5px",
                "width": "120px",
            },
        ),
        html.Div(
            [html.H4(f"{len(sentences)}"), html.P("Sentences")],
            style={
                "textAlign": "center",
                "padding": "10px",
                "backgroundColor": "#e8f5e9",
                "borderRadius": "5px",
                "margin": "5px",
                "width": "120px",
            },
        ),
        html.Div(
            [html.H4(f"{len(set(words))}"), html.P("Unique Words")],
            style={
                "textAlign": "center",
                "padding": "10px",
                "backgroundColor": "#fff3e0",
                "borderRadius": "5px",
                "margin": "5px",
                "width": "120px",
            },
        ),
        html.Div(
            [
                html.H4(f"{round(sum(sent_lengths)/len(sentences), 1)}"),
                html.P("Avg Sentence Length"),
            ],
            style={
                "textAlign": "center",
                "padding": "10px",
                "backgroundColor": "#f3e5f5",
                "borderRadius": "5px",
                "margin": "5px",
                "width": "120px",
            },
        ),
        html.Div(
            [html.H4(f"{round(avg_word_length, 1)}"), html.P("Avg Word Length")],
            style={
                "textAlign": "center",
                "padding": "10px",
                "backgroundColor": "#e0f7fa",
                "borderRadius": "5px",
                "margin": "5px",
                "width": "120px",
            },
        ),
    ]

    # Create word frequency chart
    word_freq_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
    word_freq_fig = px.bar(
        word_freq_df,
        x="Frequency",
        y="Word",
        orientation="h",
        title="Top 15 Words by Frequency",
        color="Frequency",
        color_continuous_scale=["lightpink", "hotpink", "maroon"],
    )
    word_freq_fig.update_layout(yaxis={"categoryorder": "total ascending"})

    # Create WordCloud
    wordcloud = WordCloud(
        width=600, height=700, background_color="white", colormap="Reds"
    ).generate_from_frequencies(dict(word_freq))

    # Convert wordcloud to image
    img = BytesIO()
    plt.figure(figsize=(6, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    wordcloud_src = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"

    # Create sentiment gauge
    sentiment_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sentiment.polarity,
            title={"text": "Sentiment Polarity"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [-1, -0.5], "color": "mediumpurple"},
                    {"range": [-0.5, 0], "color": "thistle"},
                    {"range": [0, 0.5], "color": "peachpuff"},
                    {"range": [0.5, 1], "color": "sandybrown"},
                ],
            },
        )
    )

    # Create polarity vs subjectivity chart
    polarity_subj_fig = go.Figure()
    polarity_subj_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="markers",
            marker=dict(size=1, color="white"),
            showlegend=False,
        )
    )
    polarity_subj_fig.add_trace(
        go.Scatter(
            x=[sentiment.subjectivity],
            y=[sentiment.polarity],
            mode="markers",
            marker=dict(size=20, color="blue"),
            name="Current Text",
        )
    )
    polarity_subj_fig.update_layout(
        title="Polarity vs Subjectivity",
        xaxis_title="Subjectivity (0=Objective, 1=Subjective)",
        yaxis_title="Polarity (-1=Negative, 1=Positive)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[-1, 1]),
    )

    # Create parts of speech chart
    pos_mapping = {
        "NN": "Noun",
        "NNS": "Noun",
        "NNP": "Proper Noun",
        "NNPS": "Proper Noun",
        "VB": "Verb",
        "VBD": "Verb",
        "VBG": "Verb",
        "VBN": "Verb",
        "VBP": "Verb",
        "VBZ": "Verb",
        "JJ": "Adjective",
        "JJR": "Adjective",
        "JJS": "Adjective",
        "RB": "Adverb",
        "RBR": "Adverb",
        "RBS": "Adverb",
        "DT": "Determiner",
        "IN": "Preposition",
        "CC": "Conjunction",
    }

    pos_simplified = {}
    for tag, count in pos_counts.items():
        category = pos_mapping.get(tag, "Other")
        pos_simplified[category] = pos_simplified.get(category, 0) + count

    pos_df = pd.DataFrame(list(pos_simplified.items()), columns=["POS", "Count"])
    pos_df = pos_df.sort_values("Count", ascending=False).head(10)

    pos_fig = px.pie(
        pos_df,
        values="Count",
        names="POS",
        title="Parts of Speech Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    pos_fig.update_traces(textposition="inside", textinfo="percent+label")

    # Create sentence length distribution
    sent_length_fig = px.histogram(
        x=sent_lengths,
        nbins=20,
        title="Sentence Length Distribution",
        labels={"x": "Words per Sentence", "y": "Frequency"},
        color_discrete_sequence=["lightblue"],
    )
    sent_length_fig.update_traces(marker=dict(line=dict(color="lightgray", width=1)))

    # Create complexity analysis chart
    # Using Flesch Reading Ease score approximation
    # Higher score = easier to read (0-100 scale)
    word_count = len(words)
    sentence_count = len(sentences)
    syllable_count = sum(len(re.findall(r"[aeiouy]+", word, re.I)) for word in words)

    if sentence_count > 0 and word_count > 0:
        flesch_score = (
            206.835
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (syllable_count / word_count)
        )
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
    else:
        flesch_score = 50  # Default value

    complexity_categories = [
        "Very Easy",
        "Easy",
        "Fairly Easy",
        "Standard",
        "Fairly Difficult",
        "Difficult",
        "Very Difficult",
    ]
    complexity_ranges = [90, 80, 70, 60, 50, 30, 0]

    # Find the appropriate category
    category_index = next(
        (i for i, score in enumerate(complexity_ranges) if flesch_score >= score),
        len(complexity_ranges) - 1,
    )
    category = complexity_categories[category_index]

    complexity_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=flesch_score,
            title={"text": f"Reading Ease: {category}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "mediumpurple"},  # Light Purple
                    {"range": [30, 50], "color": "plum"},  # Soft Purple-Pink
                    {"range": [50, 70], "color": "khaki"},  # Soft Yellow
                    {"range": [70, 90], "color": "peachpuff"},  # Light Orange
                    {
                        "range": [90, 100],
                        "color": "sandybrown",
                    },  # Warm Medium-Light Orange
                ],
            },
        )
    )
    return (
        summary_stats,
        word_freq_fig,
        wordcloud_src,
        sentiment_fig,
        polarity_subj_fig,
        pos_fig,
        sent_length_fig,
        complexity_fig,
        f"{negative_pct}%",
        f"{neutral_pct}%",
        f"{positive_pct}%",
    )


@app.callback(
    Output("text-input", "value"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_textarea(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string).decode("utf-8")
        return decoded
    return ""


# file callback
@app.callback(
    Output("download-text", "data"),
    Input("download-button", "n_clicks"),
    State("text-input", "value"),
    prevent_initial_call=True,
)
def download_text(n_clicks, text):
    if text:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return dict(content=text, filename=f"text_analysis_{timestamp}.txt")
    return None


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)



# using nltk
with open('chapter_45.txt', 'r', encoding='utf-8') as chap45:
    lines = chap45.readlines()

# Remove newline characters and create a DataFrame
lines = [line.strip() for line in lines]
df = pd.DataFrame(lines, columns=['Sentence'])

# Save to CSV
df.to_csv('chap45.csv', index=False)
csv_45 = "chap45.csv"
chap45_data = pd.read_csv(csv_45)

chap45_data[['polarity','subjectivity']] = chap45_data['Sentence'].apply(lambda Text : pd.Series(TextBlob(Text).sentiment))

from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
for index, row in chap45_data['Sentence'].items():
    # compute a score
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    # Assign score categories to variables
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    
    # If negative score (neg) is greater than positive score (pos), then the text should be categorized as "negative"
    if neg> pos:
        chap45_data.loc[index,"sentiment"] = 'negative'
    # If positive score (pos) is greater than the negative score (neg), then the text should be categorized as "positive"
    elif pos > neg:
        chap45_data.loc[index,"sentiment"] = "positive"
    # Otherwise 
    else:
        chap45_data.loc[index,"sentiment"] = "neutral"
        chap45_data.loc[index,'neg'] = neg
        chap45_data.loc[index,'pos'] = pos
        chap45_data.loc[index,'neu'] = neu
        chap45_data.loc[index,'compound'] = comp

# Let's take a look at how many are labelled positive, negative or neutral
chap45_negative = chap45_data[chap45_data['sentiment']=='negative']
chap45_positive = chap45_data[chap45_data['sentiment']=='positive']
chap45_neutral = chap45_data[chap45_data['sentiment']=='neutral']

# Let's count how many of these values belong to each category. We will define a function to count values.
def count_values_in_column(data,feature):
    
    total = data.loc[:,feature].value_counts(dropna=False)
    percentage = round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    
    return pd.concat([total,percentage],axis=1, keys=['Total', 'Percentage'])

# Values for sentiment
pc = count_values_in_column(chap45_data, "sentiment")

pc