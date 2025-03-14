README: Sentiment and Network Analysis of Dream of the Red Chamber

Project Overview

This project visualizes sentiment analysis and network analysis of the classical Chinese text Dream of the Red Chamber, an 18th-century Qing Dynasty novel. The analysis is based on the English translation Story of the Stone by David Hawkes, published in the 1970s.

Sentiment Analysis

The sentiment analysis focuses on the direct interactions between two female characters, Bao-chai and Dai-yu. After reviewing the text, I found that only a few chapters contain direct verbal interactions between them, even though they appear together in many scenes. Therefore, I extracted their direct interactions from the first eight chapters into TXT files for analysis.

Visualization Enhancements

Text Input Dropbox: Users can upload TXT files directly instead of copying and pasting text, reducing content loss. Only TXT files are supported.

Download Button: Allows users to save text as a TXT file with a timestamp if they need to convert scraped text.

Sentence Separation: Sentences in the TXT file must be separated by "\n" to be properly processed.

Improved Word Cloud: Adjusted color scheme to red (matching the book's title) and resized the word cloud for better proportion with the top 15 words graph.

Enhanced Sentiment Visualization: Updated color schemes and edge lines for better readability. Also, added a percentage breakdown of different sentiments using TextBlob.

Network Analysis

Initially, I planned to manually input character relations or use generative AI to build a CSV file, but handling 242 characters within a limited timeframe was impractical. Instead, I utilized preexisting research data on character relations in Chinese, translating it via Google Translate in Python and refining it with OpenRefine.

Visualization Enhancements

Color-Coded Relationships:

Male kinships (e.g., brothers and fathers) are in blue.

Servants are in light green.

Friends are in orange.

Edge Labels: Every connection is labeled with its relation category.

Interactive Nodes:

Each node represents a character.

Clicking a node displays the character's name, personality description, social status, and number of connections.

The network can be resized and reshaped based on the selected character.

Reflections and Future Improvements

Python provides a basic yet effective way to visualize character relationships and sentiments, but deeper coding and literary analysis skills are needed for refined textual analysis. Some findings include:

The word cloud and frequency analysis effectively highlight Bao-chai and Dai-yu’s presence but do not precisely interpret the sentiment of their verbal interactions.

Sentiment analysis results indicate 93% neutral and 6% positive interactions, but further testing with different sentiment tools (e.g., NLTK and SentimentIntensityAnalyzer) could yield different insights.

Potential Enhancements

Network Analysis:

Adding a dashboard or button to filter different edge types.

Implementing this feature requires JavaScript in standalone Bokeh output, which I am currently unfamiliar with.

Sentiment Analysis:

Converting TXT files into CSV for alternative sentiment testing.

Dashboard Layout:

The original w9_textex code had a single-page dashboard for all analyses. I modified it into separate blocks for a more detailed viewing experience.

This project lays the groundwork for further literary analysis and data visualization improvements in pre-modern Chinese literature studies.

