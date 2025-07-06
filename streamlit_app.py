# EYE OF EARTH - GLOBAL LEADER TRANSPARENCY SYSTEM
# VERSION: 1.4 MULTI-LEADER BREAKDOWN
# AUTHOR: AI Command Unit

import asyncio
import requests
import aiohttp
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from langdetect import detect
import socket
import os
import random
import sqlite3
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# === MODULE: GLOBAL SCRAPER ===
COUNTRIES = [
    "un.org", "whitehouse.gov", "gov.uk", "china.org.cn", "india.gov.in",
    "europa.eu", "russia.ru", "gov.za", "japan.go.jp", "brazil.gov.br"
]

LEADER_LOOKUP = {
    "whitehouse.gov": [
        {"name": "Donald J. Trump", "title": "President"},
        {"name": "JD Vance", "title": "Vice President"}
    ],
    "gov.uk": [
        {"name": "Keir Starmer", "title": "Prime Minister"},
        {"name": "Angela Rayner", "title": "Deputy Prime Minister"}
    ],
    "un.org": [
        {"name": "António Guterres", "title": "Secretary-General"}
    ],
    "china.org.cn": [
        {"name": "Xi Jinping", "title": "President"},
        {"name": "Li Qiang", "title": "Premier"}
    ],
    "india.gov.in": [
        {"name": "Narendra Modi", "title": "Prime Minister"} 
    ],
    # Add more as needed
}

HEADERS = {
    'User-Agent': 'EyeOfEarth/1.0 (Global Transparency Bot)'
}

async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            return await response.text()
    except Exception as e:
        return f"ERROR: {e}"

async def scrape_all():
    results = {}
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = [fetch(session, f"https://{domain}") for domain in COUNTRIES]
        pages = await asyncio.gather(*tasks)
        for i, page in enumerate(pages):
            results[COUNTRIES[i]] = page[:2000]
    return results

# === MODULE: MULTI-LEADER ANALYSIS ===
CRISIS_KEYWORDS = ["protest", "corruption", "blackout", "violence", "collapse", "scandal", "emergency"]

def analyze_leaders_by_name(domain, text):
    leaders = LEADER_LOOKUP.get(domain, [])
    results = []
    for leader in leaders:
        context = f"{leader['title']} {leader['name']} — {text[:1000]}"
        try:
            summary = summarizer(context, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
            mood = sentiment(summary)[0]
            emoji = "😊" if mood['label'] == "POSITIVE" else "😐" if mood['label'] == "NEUTRAL" else "😡"
            score = round(mood['score'] * 100)
            trust_score = 100 if mood['label'] == "POSITIVE" else 50 if mood['label'] == "NEUTRAL" else 20
            crisis_alert = any(keyword in summary.lower() for keyword in CRISIS_KEYWORDS)
            truth_index = int((trust_score * 0.6) + (score * 0.4))
            results.append({
                "name": leader['name'],
                "title": leader['title'],
                "summary": summary,
                "sentiment": mood['label'],
                "trust": trust_score,
                "truth_index": truth_index,
                "emoji": emoji,
                "crisis": crisis_alert
            })
        except Exception:
            results.append({
                "name": leader['name'],
                "title": leader['title'],
                "summary": "Analysis failed.",
                "sentiment": "UNKNOWN",
                "trust": 0,
                "truth_index": 0,
                "emoji": "❓",
                "crisis": False
            })
    return results

# === DATABASE, EMAIL, and DASHBOARD modules are unchanged and should be appended next ===
# Only update launch_dashboard(data) to use the new data structure

def launch_dashboard(data):
    st.set_page_config(page_title="Eye of Earth", layout="wide")
    st.title("🌍 EYE OF EARTH: GLOBAL LEADER TRANSPARENCY")

    for nation in data:
        st.header(f"🌐 {nation['country']} ({nation['domain']})")
        for leader in nation['leaders']:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"## {leader['emoji']}")
                st.metric(label="Trust Score", value=f"{leader['trust']}/100")
                st.metric(label="Truth Index", value=f"{leader['truth_index']}/100")
            with col2:
                st.markdown(f"**🧑‍⚖️ {leader['title']}: {leader['name']}**")
                st.markdown(f"**📊 Sentiment:** {leader['sentiment']}")
                if leader['crisis']:
                    st.error("🚨 Crisis Detected")
                st.markdown(f"**🧠 Summary:** {leader['summary']}")
        st.markdown("---")

# === MAIN ===
if __name__ == "__main__":
    st.warning("🔄 Scraping and analyzing global leaders... Please wait.")
    raw = asyncio.run(scrape_all())
    processed = []
    for domain, text in raw.items():
        if text.startswith("ERROR"): continue
        leader_data = analyze_leaders_by_name(domain, text)
        processed.append({"domain": domain, "country": domain.split(".")[0].upper(), "leaders": leader_data})
    launch_dashboard(processed)
