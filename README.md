This project is an AI-powered **financial sentiment intelligence platform** designed to analyse real-time market news from both **global and Indian premium news sources**. The system collects finance-related headlines from trusted platforms such as Reuters, BBC, Bloomberg, Mint, Economic Times, and The Hindu, and transforms them into meaningful market sentiment insights using **FinBERT**, a finance-specific NLP model.

The main objective of this project is to help investors, researchers, and finance enthusiasts quickly understand how current news sentiment may be influencing market behaviour across different asset classes such as **stocks, commodities, indices, forex, bonds, sectors, and trade-related themes**.

The workflow begins by taking a market-related query from the user. The system intelligently identifies the asset class, applies an appropriate freshness window, and gathers relevant headlines from multiple verified sources. To improve data quality, the collected headlines pass through spam filtering, finance keyword validation, and a deduplication process that removes repetitive or low-value content.

Once the news data is cleaned, the project uses **ProsusAI FinBERT** to classify each headline into **positive, negative, or neutral sentiment**, along with a confidence score. These sentiment scores are then aggregated into a **confidence-weighted sentiment index ranging from 0 to 100**, which provides an easy way to interpret whether the market mood is bullish, bearish, or neutral.

A major strength of this project is its ability to convert raw sentiment outputs into **actionable market intelligence**. It automatically generates a human-readable market summary that highlights the dominant sentiment, strongest bullish and bearish signals, source coverage, and relevance window. This makes the output more useful than a standard machine learning model by translating technical results into finance-friendly insights.

The project also includes a professionally designed **8-panel visualization dashboard** built using Matplotlib. The dashboard presents sentiment distribution, article counts, confidence scores, source coverage, regional breakdown, confidence scatter plots, rolling sentiment trends, and an overall sentiment gauge. This makes the analysis highly visual and easier to interpret for decision-making and presentations.

What makes this project stand out is that it combines **web scraping, financial NLP, quality filtering, sentiment engineering, business interpretation, and dashboard design into one end-to-end market intelligence pipeline**. Rather than being just a sentiment classifier, it behaves like a lightweight institutional research assistant for understanding how news flow may affect market direction.



