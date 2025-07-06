# EYE OF EARTH - GLOBAL LEADER TRANSPARENCY SYSTEM
# VERSION: 3.1 - ENHANCED, AI-ASSISTED, MULTIMODAL
# AUTHOR: AI Command Unit (w/ Grok + Claude Optimizations)

import asyncio
import aiohttp
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
from transformers import pipeline
from datetime import datetime, timedelta
import json
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import logging
import hashlib
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import time

# === CONFIGURATION ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@st.cache_data(ttl=timedelta(hours=12))
def fetch_countries_with_regions():
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = """
    SELECT ?country ?countryLabel ?countryCode ?continent ?continentLabel WHERE {
      ?country wdt:P31 wd:Q3624078; 
               wdt:P297 ?countryCode;
               wdt:P30 ?continent.
      SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }
    }
    LIMIT 100
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        countries = {}
        for r in results["results"]["bindings"]:
            code = r["countryCode"]["value"]
            countries[code] = {
                "name": r["countryLabel"]["value"],
                "continent": r.get("continentLabel", {}).get("value", "Unknown")
            }
        return countries
    except Exception as e:
        logger.error(f"Error fetching countries: {e}")
        return {}

COUNTRIES = fetch_countries_with_regions()

@st.cache_resource
def load_models():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        return summarizer, sentiment, emotion
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return None, None, None

summarizer, sentiment, emotion = load_models()

async def fetch_news(session, leader, country):
    await asyncio.sleep(0.2)
    return f"{leader} of {country} gave a speech about growth and security."

def analyze_news(leader, title, text, country):
    try:
        summary = summarizer(text, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
        mood = sentiment(summary)[0]
        feeling = emotion(summary)[0]['label'].lower() if emotion else 'neutral'
        score = round(mood['score'] * 100)
        crisis = any(k in summary.lower() for k in ["protest", "corruption", "crisis", "attack"])
        trust = 100 if mood['label'].startswith("4") or mood['label'].startswith("5") else 50 if mood['label'].startswith("3") else 20
        emoji = "🚨" if crisis else ("😊" if trust > 70 else "😐" if trust > 40 else "😡")
        return {
            "name": leader,
            "title": title,
            "summary": summary,
            "sentiment": mood['label'],
            "trust": trust,
            "emotion": feeling,
            "truth_index": int((trust * 0.6) + (score * 0.4)),
            "emoji": emoji,
            "crisis": crisis
        }
    except:
        return {
            "name": leader,
            "title": title,
            "summary": "Analysis failed.",
            "sentiment": "UNKNOWN",
            "trust": 0,
            "truth_index": 0,
            "emotion": "neutral",
            "emoji": "❓",
            "crisis": False
        }

async def get_leaders(session, code, country):
    query = f"""
    SELECT ?leader ?leaderLabel ?positionLabel WHERE {{
      ?country wdt:P297 \"{code}\".
      {{ ?country wdt:P35 ?leader. }} UNION {{ ?country wdt:P6 ?leader. }}
      ?leader wdt:P39 ?position.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}
    }}
    LIMIT 2
    """
    try:
        url = "https://query.wikidata.org/sparql"
        async with session.get(url, params={'query': query, 'format': 'json'}) as resp:
            data = await resp.json()
            return [{
                "name": b['leaderLabel']['value'],
                "title": b['positionLabel']['value']
            } for b in data['results']['bindings']]
    except:
        return []

async def pipeline():
    results = []
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        for code, info in COUNTRIES.items():
            leaders = await get_leaders(session, code, info['name'])
            leader_blocks = []
            for l in leaders:
                text = await fetch_news(session, l['name'], info['name'])
                analysis = analyze_news(l['name'], l['title'], text, info['name'])
                leader_blocks.append(analysis)
            if leader_blocks:
                results.append({"country": info['name'], "code": code, "continent": info['continent'], "leaders": leader_blocks})
            await asyncio.sleep(0.5)
    return results

def launch_dashboard(data):
    st.set_page_config(page_title="Eye of Earth", layout="wide")
    st.title("🌍 Eye of Earth: AI Global Leader Monitor")
    for nation in data:
        st.header(f"{nation['country']} ({nation['continent']})")
        for l in nation['leaders']:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"## {l['emoji']}")
                st.metric("Trust", f"{l['trust']}/100")
                st.metric("Truth Index", f"{l['truth_index']}/100")
            with col2:
                st.markdown(f"**{l['title']}: {l['name']}**")
                st.markdown(f"📊 Sentiment: {l['sentiment']}")
                st.markdown(f"🧠 Emotion: {l['emotion']}")
                if l['crisis']:
                    st.error("🚨 Crisis Detected")
                st.markdown(f"📝 {l['summary']}")
            st.markdown("---")

if __name__ == "__main__":
    st.info("🔄 Launching Eye of Earth 3.1...")
    data = asyncio.run(pipeline())
    launch_dashboard(data)
