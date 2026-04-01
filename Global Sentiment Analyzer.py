"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          GLOBAL SENTIMENT INTELLIGENCE SYSTEM                                ║
║          Powered by FinBERT AI | Multi-Source | Real-Time Analysis           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Covers: Stocks · Commodities · Securities · Sectors · Indices · Trade · Forex
Sources: WSJ · NYT · BBC · Reuters · The Economist · Washington Post ·
         The Hindu · Mint · MoneyControl · Economic Times · Business Standard ·
         Business Line · Bloomberg · Financial Times

Author: Swastik Sharma
"""

import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import warnings
import time
import re
import sys
import os
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

ASSET_TYPES = {
    'STOCK':       ['stock', 'share', 'equity', 'listed', 'ipo', 'dividend', 'earnings'],
    'COMMODITY':   ['gold', 'silver', 'oil', 'crude', 'copper', 'wheat', 'corn', 'commodity',
                    'natural gas', 'platinum', 'palladium', 'iron ore', 'coal', 'rubber'],
    'INDEX':       ['nifty', 'sensex', 'dow', 's&p', 'nasdaq', 'ftse', 'dax', 'nikkei',
                    'hang seng', 'cac', 'bse', 'index', 'benchmark'],
    'SECTOR':      ['sector', 'industry', 'banking', 'pharma', 'it sector', 'fmcg', 'realty',
                    'auto', 'metal', 'energy', 'healthcare', 'telecom', 'defence'],
    'FOREX':       ['rupee', 'dollar', 'euro', 'pound', 'yen', 'currency', 'forex', 'exchange rate'],
    'BOND':        ['bond', 'yield', 'treasury', 'gilt', 'debenture', 'fixed income'],
    'CRYPTO':      ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'token', 'defi'],
    'TRADE':       ['trade', 'export', 'import', 'tariff', 'wto', 'fta', 'sanctions', 'supply chain'],
}

# Freshness windows — how old is "relevant"?
FRESHNESS_CONFIG = {
    'STOCK':     {'hours': 24,  'label': '24 hours'},
    'COMMODITY': {'hours': 24,  'label': '24 hours'},
    'INDEX':     {'hours': 12,  'label': '12 hours'},
    'SECTOR':    {'hours': 48,  'label': '48 hours'},
    'FOREX':     {'hours': 12,  'label': '12 hours'},
    'BOND':      {'hours': 48,  'label': '48 hours'},
    'CRYPTO':    {'hours': 6,   'label': '6 hours'},
    'TRADE':     {'hours': 72,  'label': '72 hours'},
    'GENERAL':   {'hours': 24,  'label': '24 hours'},
}

NEWS_SOURCES = {
    # ── GLOBAL ──────────────────────────────────────────────────────────────
    'Reuters': {
        'url': 'https://www.reuters.com/search/news?blob={query}',
        'region': 'GLOBAL',
        'tier': 1,
        'selectors': ['h3.article-heading', 'h3', 'a.text__text'],
        'fallback_tags': ['h3', 'h2'],
    },
    'BBC News': {
        'url': 'https://www.bbc.com/search?q={query}&filter=news',
        'region': 'GLOBAL',
        'tier': 1,
        'selectors': ['h3.ssrcss-1b5j0yt', 'h3'],
        'fallback_tags': ['h3', 'h2'],
    },
    'Wall Street Journal': {
        'url': 'https://www.wsj.com/search?query={query}&mod=searchresults_viewallresults',
        'region': 'GLOBAL',
        'tier': 1,
        'selectors': ['h3.WSJTheme--headline--unZqjb45', 'h3', 'h2'],
        'fallback_tags': ['h3', 'h2'],
    },
    'New York Times': {
        'url': 'https://www.nytimes.com/search?query={query}&sort=newest',
        'region': 'GLOBAL',
        'tier': 1,
        'selectors': ['h4', 'h3', 'p.css-1i5dk7s'],
        'fallback_tags': ['h4', 'h3'],
    },
    'Washington Post': {
        'url': 'https://www.washingtonpost.com/search/?query={query}&sort=relevance',
        'region': 'GLOBAL',
        'tier': 1,
        'selectors': ['h3', 'span.font--headline'],
        'fallback_tags': ['h3', 'h2'],
    },
    'The Economist': {
        'url': 'https://www.economist.com/search?q={query}',
        'region': 'GLOBAL',
        'tier': 1,
        'selectors': ['h3.headline-link', 'h3'],
        'fallback_tags': ['h3', 'h2'],
    },
    'Financial Times': {
        'url': 'https://www.ft.com/search?q={query}&sort=date&dateTo=&dateFrom=',
        'region': 'GLOBAL',
        'tier': 1,
        'selectors': ['a.js-teaser-heading-link', 'h3', 'div.o-teaser__heading'],
        'fallback_tags': ['h3', 'h2'],
    },
    'Bloomberg': {
        'url': 'https://www.bloomberg.com/search?query={query}&sort=time:desc',
        'region': 'GLOBAL',
        'tier': 1,
        'selectors': ['a.headline__link', 'h3', 'h2'],
        'fallback_tags': ['h3', 'h2'],
    },
    # ── INDIA ───────────────────────────────────────────────────────────────
    'Economic Times': {
        'url': 'https://economictimes.indiatimes.com/topic/{query_dash}',
        'region': 'INDIA',
        'tier': 1,
        'selectors': ['h2', 'h3', 'h4'],
        'fallback_tags': ['h2', 'h3'],
    },
    'MoneyControl': {
        'url': 'https://www.moneycontrol.com/news/tags/{query_dash}.html',
        'region': 'INDIA',
        'tier': 1,
        'selectors': ['h2', 'h3'],
        'fallback_tags': ['h2', 'h3'],
    },
    'Mint': {
        'url': 'https://www.livemint.com/search?q={query}',
        'region': 'INDIA',
        'tier': 1,
        'selectors': ['h2', 'h3', 'a'],
        'fallback_tags': ['h2', 'h3'],
    },
    'Business Standard': {
        'url': 'https://www.business-standard.com/search?q={query}',
        'region': 'INDIA',
        'tier': 1,
        'selectors': ['h2', 'h3'],
        'fallback_tags': ['h2', 'h3'],
    },
    'Business Line': {
        'url': 'https://www.thehindubusinessline.com/search/?q={query}',
        'region': 'INDIA',
        'tier': 1,
        'selectors': ['h3.title', 'h3', 'h2'],
        'fallback_tags': ['h3', 'h2'],
    },
    'The Hindu': {
        'url': 'https://www.thehindu.com/search/?q={query}',
        'region': 'INDIA',
        'tier': 1,
        'selectors': ['h3.title', 'h3', 'h2'],
        'fallback_tags': ['h3', 'h2'],
    },
}

HEADERS_LIST = [
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    },
    {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
                      '(KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-GB,en;q=0.9',
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — AI MODEL
# ──────────────────────────────────────────────────────────────────────────────

_sentiment_model = None


def load_model():
    global _sentiment_model
    if _sentiment_model:
        return _sentiment_model

    print_banner()
    print("  🤖 Loading FinBERT Financial AI Model...")
    print("  ⏳ First run: ~1-2 min (downloading 500MB).  Next: Instant.\n")

    try:
        _sentiment_model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
        print("  ✅ FinBERT Ready!\n")
        return _sentiment_model
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        print("  💡 Fix: pip install --upgrade transformers torch")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — ASSET TYPE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_asset_type(query: str) -> str:
    q = query.lower()
    scores = defaultdict(int)
    for asset, keywords in ASSET_TYPES.items():
        for kw in keywords:
            if kw in q:
                scores[asset] += 1
    if scores:
        return max(scores, key=scores.get)
    return 'GENERAL'


def get_freshness(asset_type: str) -> dict:
    return FRESHNESS_CONFIG.get(asset_type, FRESHNESS_CONFIG['GENERAL'])


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — QUALITY FILTERING
# ──────────────────────────────────────────────────────────────────────────────

SPAM_PATTERNS = [
    r'subscribe', r'click here', r'advertisement', r'sponsored',
    r'top \d+', r'must buy', r'guaranteed', r'multibagger',
    r'sign up', r'log in', r'register', r'newsletter',
    r'cookies?', r'privacy policy', r'terms of (use|service)',
    r'follow us', r'share this', r'read more', r'load more',
    r'breaking[\s!]*$', r'^\s*$', r'\.{3,}',
]
_spam_re = re.compile('|'.join(SPAM_PATTERNS), re.IGNORECASE)

FINANCE_KEYWORDS = [
    'stock', 'share', 'market', 'price', 'earnings', 'profit', 'revenue',
    'quarter', 'percent', '%', 'crore', 'lakh', 'billion', 'million',
    'company', 'business', 'investor', 'trading', 'sensex', 'nifty',
    'dow', 'nasdaq', 's&p', 'fed', 'rbi', 'sebi', 'interest rate',
    'gdp', 'inflation', 'currency', 'forex', 'commodity', 'gold', 'oil',
    'crude', 'bond', 'yield', 'ipo', 'merger', 'acquisition', 'sector',
    'trade', 'export', 'import', 'tariff', 'growth', 'decline', 'rally',
    'slump', 'surge', 'fall', 'rise', 'gain', 'loss', 'fund', 'index',
    'portfolio', 'dividend', 'buyback', 'valuation', 'forecast',
]


def is_quality_headline(text: str, query: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) < 20 or len(text) > 400:
        return False
    if _spam_re.search(text):
        return False
    if text.count('!') > 1:
        return False

    tl = text.lower()
    ql = query.lower().replace('.ns', '').replace('.bo', '')
    query_words = [w for w in ql.split() if len(w) > 2]
    if not any(w in tl for w in query_words):
        return False

    if not any(fw in tl for fw in FINANCE_KEYWORDS):
        # Relax for global macro queries
        if not any(w in tl for w in ['report', 'data', 'rate', 'policy', 'decision']):
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — FETCH FROM SOURCES
# ──────────────────────────────────────────────────────────────────────────────

def build_url(template: str, query: str) -> str:
    q_enc = requests.utils.quote(query)
    q_dash = query.replace(' ', '-').replace('.NS', '').replace('.BO', '').lower()
    return template.replace('{query}', q_enc).replace('{query_dash}', q_dash)


def fetch_source(name: str, config: dict, query: str, max_articles: int = 20) -> list:
    articles = []
    url = build_url(config['url'], query)

    for headers in HEADERS_LIST:
        try:
            resp = requests.get(url, headers=headers, timeout=18,
                                allow_redirects=True)
            if resp.status_code == 200:
                break
            if resp.status_code in (403, 429, 503):
                time.sleep(1.5)
        except Exception:
            continue
    else:
        return []

    try:
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Try specific CSS selectors first
        found = []
        for sel in config.get('selectors', []):
            found = soup.select(sel)
            if len(found) >= 3:
                break

        # Fallback to generic tags
        if len(found) < 3:
            for tag in config.get('fallback_tags', ['h3', 'h2']):
                found += soup.find_all(tag, limit=60)

        seen_texts = set()
        for el in found:
            text = el.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            if text in seen_texts:
                continue
            if is_quality_headline(text, query):
                seen_texts.add(text)
                articles.append({
                    'title': text,
                    'source': name,
                    'region': config['region'],
                    'tier': config['tier'],
                    'timestamp': datetime.now(),
                    'url': url,
                })
            if len(articles) >= max_articles:
                break

    except Exception:
        pass

    return articles


def fetch_all_sources(query: str, region_filter: str = 'BOTH') -> list:
    print(f"\n  {'─'*60}")
    print(f"  📡 FETCHING NEWS FROM VERIFIED SOURCES")
    print(f"  {'─'*60}")

    all_articles = []
    for name, config in NEWS_SOURCES.items():
        if region_filter == 'INDIA' and config['region'] == 'GLOBAL':
            continue
        if region_filter == 'GLOBAL' and config['region'] == 'INDIA':
            continue

        print(f"  📰 {name:<22}", end='', flush=True)
        arts = fetch_source(name, config, query)
        print(f"→ {len(arts):2d} articles", end='')

        if arts:
            all_articles.extend(arts)
            print(f"  ✓")
        else:
            print(f"  (no results)")

        time.sleep(0.8)

    return all_articles


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — DEDUPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def deduplicate(articles: list) -> list:
    unique = []
    seen_sigs = []

    for art in articles:
        words = set(re.sub(r'[^\w\s]', '', art['title'].lower()).split())
        words = {w for w in words if len(w) > 3}
        sig = frozenset(sorted(words)[:10])

        is_dup = any(
            len(sig & s) / max(len(sig), 1) >= 0.65
            for s in seen_sigs
        )
        if not is_dup:
            unique.append(art)
            seen_sigs.append(sig)

    return unique


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — SENTIMENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    global _sentiment_model
    try:
        t = text[:512]
        result = _sentiment_model(t)[0]
        return {'label': result['label'].lower(), 'score': float(result['score'])}
    except Exception:
        return {'label': 'neutral', 'score': 0.5}


def analyze_batch(articles: list) -> list:
    print(f"\n  {'─'*60}")
    print(f"  🧠 RUNNING AI SENTIMENT ANALYSIS ({len(articles)} articles)")
    print(f"  {'─'*60}")
    print("  Progress: ", end='', flush=True)

    results = []
    for i, art in enumerate(articles, 1):
        sent = analyze_sentiment(art['title'])
        results.append({**art, **sent})
        if i % 10 == 0:
            print(f"▓", end='', flush=True)
        elif i % 2 == 0:
            print(f"░", end='', flush=True)

    print(f" ✅\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list, asset_type: str) -> dict:
    if not results:
        return None

    total = len(results)
    pos = [r for r in results if r['label'] == 'positive']
    neg = [r for r in results if r['label'] == 'negative']
    neu = [r for r in results if r['label'] == 'neutral']

    # Weighted sentiment score (confidence-weighted)
    raw_score = sum(r['score'] for r in pos) - sum(r['score'] for r in neg)
    normalized = raw_score / total  # range roughly -1..+1

    # Sentiment index 0–100
    sentiment_index = int(50 + normalized * 50)
    sentiment_index = max(0, min(100, sentiment_index))

    # Market label
    if normalized > 0.40:
        label, color, emoji = 'STRONGLY BULLISH', '#00c853', '🚀'
    elif normalized > 0.12:
        label, color, emoji = 'BULLISH', '#69f0ae', '📈'
    elif normalized < -0.40:
        label, color, emoji = 'STRONGLY BEARISH', '#d50000', '📉'
    elif normalized < -0.12:
        label, color, emoji = 'BEARISH', '#ff5252', '⬇️'
    else:
        label, color, emoji = 'NEUTRAL', '#90a4ae', '➡️'

    # Source breakdown
    by_source = Counter(r['source'] for r in results)
    by_region = Counter(r['region'] for r in results)

    # Top articles
    top_pos = sorted(pos, key=lambda x: x['score'], reverse=True)[:5]
    top_neg = sorted(neg, key=lambda x: x['score'], reverse=True)[:5]

    # Confidence averages
    avg_conf = lambda lst: np.mean([r['score'] for r in lst]) if lst else 0.0

    freshness = get_freshness(asset_type)

    return {
        'total': total,
        'pos': pos, 'neg': neg, 'neu': neu,
        'pos_n': len(pos), 'neg_n': len(neg), 'neu_n': len(neu),
        'pos_pct': len(pos) / total * 100,
        'neg_pct': len(neg) / total * 100,
        'neu_pct': len(neu) / total * 100,
        'raw_score': raw_score,
        'normalized': normalized,
        'sentiment_index': sentiment_index,
        'label': label, 'color': color, 'emoji': emoji,
        'by_source': by_source,
        'by_region': by_region,
        'top_pos': top_pos,
        'top_neg': top_neg,
        'avg_pos_conf': avg_conf(pos),
        'avg_neg_conf': avg_conf(neg),
        'avg_neu_conf': avg_conf(neu),
        'asset_type': asset_type,
        'freshness': freshness,
        'generated_at': datetime.now(),
        'all_results': results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — AI MARKET SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def generate_summary(m: dict, query: str) -> str:
    """Generate a human-readable AI market summary from metrics."""
    n = m['normalized']
    asset = m['asset_type'].lower()
    total = m['total']
    pos_pct = m['pos_pct']
    neg_pct = m['neg_pct']
    neu_pct = m['neu_pct']
    freshness = m['freshness']['label']
    label = m['label']
    idx = m['sentiment_index']
    sources_n = len(m['by_source'])
    regions = list(m['by_region'].keys())
    region_str = ' & '.join(regions) if regions else 'Global'

    # Build key themes from top headlines
    top_words = Counter()
    all_headlines = [r['title'] for r in m['all_results']]
    for h in all_headlines:
        for w in re.sub(r'[^\w\s]', '', h.lower()).split():
            if len(w) > 4 and w not in ('about', 'after', 'their', 'which', 'would',
                                         'could', 'should', 'these', 'those', 'there',
                                         'where', 'while', 'since', 'under', 'above',
                                         'below', 'stock', 'share', 'market', 'news'):
                top_words[w] += 1
    themes = [w for w, _ in top_words.most_common(6)]
    theme_str = ', '.join(themes) if themes else 'market developments'

    # Write summary paragraphs
    lines = []
    lines.append(
        f"📋 MARKET INTELLIGENCE SUMMARY — {query.upper()}"
    )
    lines.append(f"   Asset Class: {m['asset_type']}  |  Coverage: {region_str}  |  "
                 f"Freshness Window: Past {freshness}")
    lines.append("")

    # Overall verdict
    lines.append(f"   ▶ OVERALL VERDICT: {label} {m['emoji']}")
    lines.append(
        f"   The Sentiment Intelligence Index for {query} stands at {idx}/100, "
        f"derived from {total} verified news articles across {sources_n} premium sources."
    )
    lines.append("")

    # Sentiment breakdown paragraph
    dominant = 'positive' if pos_pct > neg_pct and pos_pct > neu_pct else \
               ('negative' if neg_pct > pos_pct and neg_pct > neu_pct else 'neutral')

    if dominant == 'positive':
        lines.append(
            f"   Market coverage is predominantly optimistic, with {pos_pct:.1f}% of headlines "
            f"carrying a positive tone vs. {neg_pct:.1f}% negative and {neu_pct:.1f}% neutral. "
            f"Key topics driving coverage include: {theme_str}."
        )
    elif dominant == 'negative':
        lines.append(
            f"   Market coverage leans cautious-to-bearish, with {neg_pct:.1f}% of headlines "
            f"negative vs. {pos_pct:.1f}% positive and {neu_pct:.1f}% neutral. "
            f"Headline themes reflect concerns around: {theme_str}."
        )
    else:
        lines.append(
            f"   Market sentiment is mixed and indecisive. {neu_pct:.1f}% of coverage is neutral, "
            f"while positive ({pos_pct:.1f}%) and negative ({neg_pct:.1f}%) views are roughly balanced. "
            f"Key topics in coverage: {theme_str}."
        )
    lines.append("")

    # Freshness note
    lines.append(
        f"   ⏱  RELEVANCE WINDOW: Articles analysed are from the past {freshness}. "
        f"For {m['asset_type']} assets, sentiment beyond this window may not reflect "
        f"current market conditions and should be used with caution."
    )

    # Top signals
    if m['top_pos']:
        lines.append("")
        lines.append("   📌 STRONGEST BULLISH SIGNAL:")
        title_pos = m['top_pos'][0]['title'][:95]
        src_pos   = m['top_pos'][0]['source']
        conf_pos  = m['top_pos'][0]['score'] * 100
        lines.append(f'      \u201c{title_pos}\u201d')
        lines.append(f"      \u2014 {src_pos}  |  AI Confidence: {conf_pos:.1f}%")

    if m['top_neg']:
        lines.append("")
        lines.append("   \U0001f4cc STRONGEST BEARISH SIGNAL:")
        title_neg = m['top_neg'][0]['title'][:95]
        src_neg   = m['top_neg'][0]['source']
        conf_neg  = m['top_neg'][0]['score'] * 100
        lines.append(f'      \u201c{title_neg}\u201d')
        lines.append(f"      \u2014 {src_neg}  |  AI Confidence: {conf_neg:.1f}%")

    lines.append("")
    lines.append(
        "   ⚠  DISCLAIMER: AI-generated analysis for informational purposes only. "
        "Not financial advice. Always consult a registered financial advisor."
    )

    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — TERMINAL REPORT
# ──────────────────────────────────────────────────────────────────────────────

def bar_str(pct: float, width: int = 40) -> str:
    filled = int(pct / 100 * width)
    return '█' * filled + '░' * (width - filled)


def print_report(m: dict, query: str, summary: str):
    W = 90
    sep = '═' * W

    print(f"\n{sep}")
    print(f"{'  📊 GLOBAL FINANCIAL SENTIMENT INTELLIGENCE':^{W}}")
    print(f"  Query: {query.upper():<{W-10}}")
    print(sep)

    # Sentiment index gauge
    idx = m['sentiment_index']
    gauge_w = 60
    pos_filled = int(idx / 100 * gauge_w)
    gauge = '▰' * pos_filled + '▱' * (gauge_w - pos_filled)
    print(f"\n  SENTIMENT INDEX:  [{gauge}]  {idx}/100")
    print(f"  MARKET FEELING:   {m['emoji']} {m['label']}")
    print(f"  RAW SCORE:        {m['raw_score']:+.3f}  (Normalised: {m['normalized']:+.3f})")
    print(f"  ARTICLES:         {m['total']} from {len(m['by_source'])} verified sources")
    print(f"  FRESHNESS:        Past {m['freshness']['label']} only\n")

    print(f"  {'─'*W}")
    print(f"  SENTIMENT BREAKDOWN")
    print(f"  {'─'*W}")
    for lbl, key_n, key_pct, key_conf, sym in [
        ('POSITIVE', 'pos_n', 'pos_pct', 'avg_pos_conf', '✅'),
        ('NEGATIVE', 'neg_n', 'neg_pct', 'avg_neg_conf', '❌'),
        ('NEUTRAL',  'neu_n', 'neu_pct', 'avg_neu_conf', '⚪'),
    ]:
        print(f"\n  {sym} {lbl}: {m[key_n]:3d} articles  ({m[key_pct]:5.1f}%)")
        print(f"     {bar_str(m[key_pct])}")
        print(f"     Avg AI Confidence: {m[key_conf]*100:.1f}%")

    print(f"\n  {'─'*W}")
    print(f"  NEWS SOURCE COVERAGE")
    print(f"  {'─'*W}")
    for src, cnt in sorted(m['by_source'].items(), key=lambda x: -x[1]):
        pct = cnt / m['total'] * 100
        print(f"  {src:<28} {bar_str(pct, 25)}  {cnt:2d} ({pct:5.1f}%)")

    print(f"\n  {'─'*W}")
    print(f"  REGIONAL COVERAGE")
    print(f"  {'─'*W}")
    for reg, cnt in m['by_region'].items():
        pct = cnt / m['total'] * 100
        print(f"  {reg:<12} {bar_str(pct, 30)}  {cnt:2d} ({pct:5.1f}%)")

    print(f"\n  {'─'*W}")
    print(f"  TOP HEADLINES\n  {'─'*W}")
    for cat, lst, sym in [('POSITIVE', m['top_pos'], '🟢'), ('NEGATIVE', m['top_neg'], '🔴')]:
        print(f"\n  {sym} MOST {cat}:")
        for i, a in enumerate(lst[:3], 1):
            print(f"\n  {i}. [{a['source']}]")
            print(f"     {a['title'][:82]}")
            print(f"     Confidence: {a['score']*100:.1f}%")

    print(f"\n{'═'*W}")
    print(f"\n{summary}\n")
    print(f"{'═'*W}\n")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 10 — DASHBOARD VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def create_dashboard(m: dict, query: str, filename: str):
    try:
        results = m['all_results']
        BG   = '#0d1117'
        CARD = '#161b22'
        GRID = '#21262d'
        TXT  = '#e6edf3'
        POS  = '#3fb950'
        NEG  = '#f85149'
        NEU  = '#8b949e'
        ACC  = '#58a6ff'

        fig = plt.figure(figsize=(22, 13), facecolor=BG)
        fig.patch.set_facecolor(BG)

        gs = gridspec.GridSpec(
            3, 4,
            figure=fig,
            hspace=0.48, wspace=0.35,
            left=0.05, right=0.97,
            top=0.90, bottom=0.06
        )

        # ── Title ──────────────────────────────────────────────────────────
        fig.text(
            0.5, 0.96,
            f'FINANCIAL SENTIMENT INTELLIGENCE  ·  {query.upper()}',
            ha='center', va='top', fontsize=18, fontweight='bold',
            color=TXT, fontfamily='monospace'
        )
        fig.text(
            0.5, 0.925,
            f"{m['emoji']}  {m['label']}   ·   Sentiment Index: {m['sentiment_index']}/100"
            f"   ·   {m['total']} Articles   ·   {len(m['by_source'])} Sources"
            f"   ·   Past {m['freshness']['label']}",
            ha='center', va='top', fontsize=11, color=ACC, fontfamily='monospace'
        )

        def styled_ax(ax):
            ax.set_facecolor(CARD)
            ax.tick_params(colors=TXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)
            ax.xaxis.label.set_color(TXT)
            ax.yaxis.label.set_color(TXT)
            ax.title.set_color(TXT)
            return ax

        # ── 1. Donut ──────────────────────────────────────────────────────
        ax1 = styled_ax(fig.add_subplot(gs[0, 0]))
        sizes = [m['pos_n'], m['neg_n'], m['neu_n']]
        clrs  = [POS, NEG, NEU]
        lbls  = ['Positive', 'Negative', 'Neutral']
        wedges, texts, autos = ax1.pie(
            sizes, labels=None, colors=clrs, autopct='%1.1f%%',
            startangle=90, wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
            pctdistance=0.75
        )
        for at in autos:
            at.set_color(BG)
            at.set_fontsize(8)
            at.set_fontweight('bold')
        ax1.set_title('Sentiment Mix', fontsize=10, fontweight='bold', pad=10)
        patches = [mpatches.Patch(color=c, label=l) for c, l in zip(clrs, lbls)]
        ax1.legend(handles=patches, loc='lower center', ncol=3,
                   fontsize=7, framealpha=0, labelcolor=TXT,
                   bbox_to_anchor=(0.5, -0.12))

        # ── 2. Bar — counts ───────────────────────────────────────────────
        ax2 = styled_ax(fig.add_subplot(gs[0, 1]))
        cats = ['Positive', 'Negative', 'Neutral']
        vals = [m['pos_n'], m['neg_n'], m['neu_n']]
        bars = ax2.bar(cats, vals, color=clrs, edgecolor=BG, linewidth=1.5, width=0.55)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(v), ha='center', va='bottom', fontsize=9,
                     fontweight='bold', color=TXT)
        ax2.set_title('Article Counts', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=8)
        ax2.grid(axis='y', alpha=0.2, color=GRID)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(cats, fontsize=8)

        # ── 3. Gauge / Sentiment Index ────────────────────────────────────
        ax3 = styled_ax(fig.add_subplot(gs[0, 2]))
        ax3.axis('off')
        idx_val = m['sentiment_index']
        theta = np.linspace(0, np.pi, 200)
        r_outer, r_inner = 1.0, 0.6

        # Background arc
        ax3.add_patch(mpatches.Wedge(
            (0, 0), r_outer, 0, 180,
            width=r_outer - r_inner,
            facecolor=GRID, edgecolor='none'
        ))
        # Colored fill based on index
        fill_deg = idx_val / 100 * 180
        fill_color = POS if idx_val > 55 else (NEG if idx_val < 45 else NEU)
        ax3.add_patch(mpatches.Wedge(
            (0, 0), r_outer, 0, fill_deg,
            width=r_outer - r_inner,
            facecolor=fill_color, edgecolor='none', alpha=0.9
        ))
        # Needle
        angle_rad = np.radians(fill_deg)
        nx = 0.72 * np.cos(angle_rad)
        ny = 0.72 * np.sin(angle_rad)
        ax3.annotate('', xy=(nx, ny), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=TXT, lw=2))

        ax3.text(0, -0.15, f'{idx_val}', ha='center', fontsize=20,
                 fontweight='bold', color=fill_color)
        ax3.text(0, -0.35, '/100', ha='center', fontsize=10, color=NEU)
        ax3.text(0, 0.6, 'Sentiment\nIndex', ha='center', fontsize=9,
                 color=TXT, va='center')
        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-0.5, 1.1)
        ax3.set_title('Market Gauge', fontsize=10, fontweight='bold')

        # ── 4. Confidence bars ────────────────────────────────────────────
        ax4 = styled_ax(fig.add_subplot(gs[0, 3]))
        conf_vals = [
            m['avg_pos_conf'] * 100,
            m['avg_neg_conf'] * 100,
            m['avg_neu_conf'] * 100,
        ]
        brs = ax4.bar(cats, conf_vals, color=clrs, edgecolor=BG, linewidth=1.5, width=0.55)
        for bar, v in zip(brs, conf_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=8,
                     fontweight='bold', color=TXT)
        ax4.set_ylim(0, 105)
        ax4.set_title('AI Confidence', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Avg Confidence (%)', fontsize=8)
        ax4.grid(axis='y', alpha=0.2, color=GRID)
        ax4.set_xticks(range(3))
        ax4.set_xticklabels(cats, fontsize=8)

        # ── 5. Source distribution (horizontal bar) ───────────────────────
        ax5 = styled_ax(fig.add_subplot(gs[1, :2]))
        srcs  = list(m['by_source'].keys())
        cnts  = list(m['by_source'].values())
        order = sorted(range(len(cnts)), key=lambda i: cnts[i])
        srcs  = [srcs[i] for i in order]
        cnts  = [cnts[i] for i in order]
        colors_src = [ACC] * len(srcs)
        bars_h = ax5.barh(srcs, cnts, color=colors_src, edgecolor=BG, linewidth=1, height=0.6)
        for bar in bars_h:
            w = bar.get_width()
            ax5.text(w + 0.1, bar.get_y() + bar.get_height()/2,
                     str(int(w)), va='center', fontsize=8, color=TXT)
        ax5.set_title('Articles by Source', fontsize=10, fontweight='bold')
        ax5.set_xlabel('Article Count', fontsize=8)
        ax5.grid(axis='x', alpha=0.2, color=GRID)
        ax5.tick_params(axis='y', labelsize=7)

        # ── 6. Regional pie ───────────────────────────────────────────────
        ax6 = styled_ax(fig.add_subplot(gs[1, 2]))
        reg_lbls = list(m['by_region'].keys())
        reg_vals = list(m['by_region'].values())
        reg_clrs = ['#58a6ff', '#ffa657', '#3fb950', '#bc8cff', '#ff7b72'][:len(reg_lbls)]
        wed, txt2, au2 = ax6.pie(
            reg_vals, labels=reg_lbls, colors=reg_clrs,
            autopct='%1.0f%%', startangle=90,
            wedgeprops=dict(edgecolor=BG, linewidth=1.5),
            textprops={'color': TXT, 'fontsize': 8},
        )
        for au in au2:
            au.set_color(BG)
            au.set_fontsize(7)
        ax6.set_title('Regional Coverage', fontsize=10, fontweight='bold')

        # ── 7. Scatter — conf vs label ────────────────────────────────────
        ax7 = styled_ax(fig.add_subplot(gs[1, 3]))
        for lbl2, lst2, col2 in [('positive', m['pos'], POS),
                                   ('negative', m['neg'], NEG),
                                   ('neutral',  m['neu'], NEU)]:
            xs = list(range(len(lst2)))
            ys = [r['score'] for r in lst2]
            ax7.scatter(xs, ys, c=col2, alpha=0.6, s=18, label=lbl2.capitalize())
        ax7.set_title('Confidence Distribution', fontsize=10, fontweight='bold')
        ax7.set_xlabel('Article #', fontsize=8)
        ax7.set_ylabel('Confidence Score', fontsize=8)
        ax7.legend(fontsize=7, framealpha=0, labelcolor=TXT)
        ax7.grid(alpha=0.15, color=GRID)

        # ── 8. Trend line (rolling MA) ────────────────────────────────────
        ax8 = styled_ax(fig.add_subplot(gs[2, :]))
        values = []
        for r in results:
            if r['label'] == 'positive':
                values.append(r['score'])
            elif r['label'] == 'negative':
                values.append(-r['score'])
            else:
                values.append(0)

        x = np.arange(len(values))
        window = max(5, len(values) // 8)
        ma = pd.Series(values).rolling(window=window, min_periods=1).mean().values

        ax8.fill_between(x, 0, values,
                         where=[v >= 0 for v in values], color=POS, alpha=0.25, interpolate=True)
        ax8.fill_between(x, 0, values,
                         where=[v <  0 for v in values], color=NEG, alpha=0.25, interpolate=True)
        ax8.plot(x, values, color=NEU,   alpha=0.35, linewidth=0.8, label='Individual')
        ax8.plot(x, ma,     color=ACC,   linewidth=2.5,  label=f'{window}-article MA')
        ax8.axhline(0, color=GRID, linewidth=1.2, linestyle='--', alpha=0.8)

        ax8.set_title('Sentiment Trend  (Individual articles + Rolling Average)',
                      fontsize=10, fontweight='bold')
        ax8.set_xlabel('Article Index (chronological order within source)', fontsize=8)
        ax8.set_ylabel('Sentiment Score (Positive ↑ / Negative ↓)', fontsize=8)
        ax8.legend(fontsize=8, framealpha=0.2, labelcolor=TXT,
                   facecolor=CARD, edgecolor=GRID)
        ax8.grid(alpha=0.15, color=GRID)

        # Timestamp
        fig.text(
            0.99, 0.01,
            f"Generated: {m['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}  ·  "
            f"FinBERT AI  ·  Not financial advice",
            ha='right', fontsize=7, color=NEU, style='italic'
        )

        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor=BG)
        print(f"  💾 Dashboard saved: {filename}")
        plt.show()

    except Exception as e:
        print(f"  ❌ Visualisation error: {e}")
        import traceback; traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def print_banner():
    print("\n" + "╔" + "═"*76 + "╗")
    print("║" + " "*76 + "║")
    print("║  🌐  GLOBAL FINANCIAL SENTIMENT INTELLIGENCE SYSTEM  v3.0            ║")
    print("║      FinBERT AI  ·  16 Verified Sources  ·  Real-Time Analysis       ║")
    print("║      Stocks · Commodities · Indices · Forex · Bonds · Sectors        ║")
    print("║" + " "*76 + "║")
    print("╚" + "═"*76 + "╝\n")


def print_asset_menu():
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  WHAT CAN YOU ANALYSE?                                          │")
    print("  │                                                                 │")
    print("  │  📈 STOCKS    : Reliance, TCS, Apple, Tesla, HDFC Bank          │")
    print("  │  🏅 COMMODITIES: Gold, Silver, Crude Oil, Natural Gas           │")
    print("  │  📊 INDICES   : Nifty 50, Sensex, S&P 500, NASDAQ, Dow Jones    │")
    print("  │  🏭 SECTORS   : IT Sector, Banking, Pharma, FMCG, Auto          │")
    print("  │  💱 FOREX     : USD/INR, EUR/USD, Rupee, Dollar, Yen            │")
    print("  │  🔗 BONDS     : US Treasury, India Bond Yield, G-Sec            │")
    print("  │  🌍 TRADE     : US-China Trade, Tariff, WTO, Export             │")
    print("  └─────────────────────────────────────────────────────────────────┘\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    load_model()
    print_asset_menu()

    query = input("  📝 Enter your query: ").strip()
    if not query:
        print("  ❌ No query entered. Exiting.")
        return

    print("\n  Which news region do you prefer?")
    print("  [1] Both India + Global  [2] India Only  [3] Global Only")
    choice = input("  → ").strip()
    region = 'INDIA' if choice == '2' else ('GLOBAL' if choice == '3' else 'BOTH')

    asset_type = detect_asset_type(query)
    freshness  = get_freshness(asset_type)
    print(f"\n  ✅ Detected asset class : {asset_type}")
    print(f"  ✅ Relevance window     : Past {freshness['label']}")
    print(f"  ✅ Region filter        : {region}")

    # Fetch
    raw_articles = fetch_all_sources(query, region)
    print(f"\n  📦 Total fetched : {len(raw_articles)} articles")

    unique = deduplicate(raw_articles)
    print(f"  🧹 After dedup   : {len(unique)} unique articles")

    if len(unique) < 5:
        print(f"\n  ⚠  Only {len(unique)} articles found — results may be limited.")
        go = input("  Continue anyway? (y/n): ").strip().lower()
        if go != 'y':
            return

    # Analyse
    results = analyze_batch(unique)

    # Metrics
    m = compute_metrics(results, asset_type)
    if not m:
        print("  ❌ Could not compute metrics.")
        return

    # Summary
    summary = generate_summary(m, query)

    # Report
    print_report(m, query, summary)

    # Dashboard
    safe_q = re.sub(r'[^\w]', '_', query.lower())[:30]
    fname  = f"sentiment_{safe_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    print(f"  📊 Creating 8-panel dashboard...\n")
    create_dashboard(m, query, fname)

    print(f"\n  {'═'*70}")
    print(f"  ✅ ANALYSIS COMPLETE")
    print(f"  {'═'*70}\n")


if __name__ == '__main__':
    main()