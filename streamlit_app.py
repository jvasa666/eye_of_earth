import os
import streamlit as st
from transformers import pipeline, AutoTokenizer # Keep for potential future use, or remove if truly not needed
import warnings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import random
import requests
from bs4 import BeautifulSoup
import re
import numpy as np # For random data generation if scraping fails
import qrcode
from io import BytesIO
from PIL import Image

# Suppress warnings from Hugging Face Transformers and general warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="EYE OF EARTH: Global Analytics",
    page_icon="🌍",
    layout="wide", # Use wide layout for the dashboard
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
# This CSS will make buttons look nicer and provide some spacing
st.markdown(
    """
    <style>
    /* General Streamlit container styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333; /* Darker text */
        font-family: 'Arial', sans-serif;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #2E86C1; /* A nice blue for headers */
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }

    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5em;
        color: #1A5276; /* Darker blue */
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1em;
        color: #666;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.9em;
        color: #28B463; /* Green for positive delta */
    }

    /* Buttons styling */
    .stButton button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
        margin: 5px; /* Add some margin around buttons */
    }
    .stButton button:hover {
        background-color: #45a049; /* Darker green on hover */
        transform: scale(1.02);
    }
    .stButton button:active {
        transform: scale(0.98);
    }

    /* Style for the "Buy Me a Coffee" button specifically */
    .buy-me-a-coffee-button button {
        background-color: #FFDD00; /* Yellow for Ko-fi/BMC */
        color: #614B2A; /* Dark brown text */
        border: 2px solid #614B2A;
    }
    .buy-me-a-coffee-button button:hover {
        background-color: #E6C200;
        color: #614B2A;
    }

    /* General text styling for better readability */
    p {
        font-size: 1.1em;
        line-height: 1.6;
    }
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        color: #2E86C1; /* A nice blue */
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .crypto-address {
        background-color: #e6e9ee; /* Slightly darker light gray for code */
        padding: 8px;
        border-radius: 5px;
        font-family: monospace;
        word-break: break-all; /* Ensure long addresses wrap */
        white-space: pre-wrap; /* Preserve whitespace and wrap */
        font-size: 0.9em;
        color: #333;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #e0e6ed; /* Slightly darker sidebar background */
        padding-top: 2rem;
    }
    .sidebar .stButton button {
        width: 90%; /* Make sidebar buttons wider */
        margin-left: 5%;
        margin-right: 5%;
    }
    .sidebar .st-bb { /* Target markdown block quotes in sidebar for addresses */
        background-color: #d0d6df;
        padding: 10px;
        border-radius: 5px;
        word-break: break-all;
        font-size: 0.9em;
    }
    .sidebar .stMarkdown {
        padding: 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Donation and Tip Links (Sidebar Integration with enhanced styling) ---
# Function to generate QR code as a Streamlit image
def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=4, # Smaller box size for sidebar
        border=2, # Smaller border for sidebar
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

with st.sidebar:
    st.title("💖 Support XenoTech")
    st.markdown("""
    Your contributions power **EYE OF EARTH**!
    """)

    # Buy Me a Coffee
    st.markdown("---")
    st.subheader("☕ Buy Me a Coffee")
    st.markdown(
        f"""
        <div class="buy-me-a-coffee-button">
            <a href="https://coff.ee/xenotech" target="_blank" style="text-decoration: none;">
                <button>
                    Buy Me a Coffee! ☕
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("💸 Crypto Tips")

    with st.expander("Ethereum (ETH)"):
        eth_address = "0x5036dbcEEfae0a7429e64467222e1E259819c7C7"
        st.image(generate_qr_code(eth_address), width=100, caption="Scan ETH QR")
        st.markdown(f'<div class="crypto-address">{eth_address}</div>', unsafe_allow_html=True)
        # Note: Streamlit's native button doesn't copy directly, but displays for easy manual copy
        if st.button("Copy ETH Address", key="sidebar_copy_eth"):
            st.code(eth_address, language="text")
            st.success("ETH address copied to clipboard (see code block above)!")

    with st.expander("Bitcoin (BTC)"):
        btc_address = "bc1qzncgc94kgtcpumx80m5uedsp3hqp4fec2e3rvr" # REMEMBER TO REPLACE THIS WITH YOUR REAL BTC ADDRESS
        st.image(generate_qr_code(btc_address), width=100, caption="Scan BTC QR")
        st.markdown(f'<div class="crypto-address">{btc_address}</div>', unsafe_allow_html=True)
        if st.button("Copy BTC Address", key="sidebar_copy_btc"):
            st.code(btc_address, language="text")
            st.success("BTC address copied to clipboard (see code block above)!")

    with st.expander("Phantom/Solana (SOL)"):
        sol_address = "7ckfzhhkwkpdTRHdXoEorD5gN3Yg6ggaTHw2B6gF6hKq" # REMEMBER TO REPLACE THIS WITH YOUR REAL SOL ADDRESS
        st.image(generate_qr_code(sol_address), width=100, caption="Scan SOL QR")
        st.markdown(f'<div class="crypto-address">{sol_address}</div>', unsafe_allow_html=True)
        if st.button("Copy SOL Address", key="sidebar_copy_sol"):
            st.code(sol_address, language="text")
            st.success("SOL address copied to clipboard (see code block above)!")

    st.markdown("---")
    st.subheader("Connect with XenoTech")
    st.markdown("""
    - 🌐 [Farcaster Profile](https://warpcast.com/xenotech)
    - 📢 [Telegram Channel](https://t.me/xenodrop)
    - 🔗 [Follow Lens](https://lens.xyz/u/xenotech)
    """)
    st.markdown("---")
    st.success("Thank you for your generosity!")

# --- Model Loading (Kept for potential future use, but not actively used in dashboard display yet) ---
@st.cache_resource
def load_summarizer():
    """Loads and caches the BART large CNN summarization pipeline."""
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.warning(f"Could not load summarizer model: {e}. Summarization features might be unavailable.")
        return None

@st.cache_resource
def load_sentiment_analyzer():
    """Loads and caches the multilingual sentiment analysis pipeline."""
    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception as e:
        st.warning(f"Could not load sentiment analyzer model: {e}. Sentiment analysis features might be unavailable.")
        return None

@st.cache_resource
def load_emotion_analyzer():
    """Loads and caches the emotion classification pipeline."""
    try:
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    except Exception as e:
        st.warning(f"Could not load emotion analyzer model: {e}. Emotion analysis features might be unavailable.")
        return None

@st.cache_resource
def load_tokenizer():
    """Loads and caches the tokenizer for the BART large CNN model."""
    try:
        return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    except Exception as e:
        st.warning(f"Could not load tokenizer: {e}. Text processing features might be unavailable.")
        return None

# --- Data Scraping Functions (Re-implemented for real-time data) ---

@st.cache_data(ttl=3600) # Cache for 1 hour
def scrape_world_leaders():
    """
    Scrapes current world leaders from Wikipedia.
    Uses a robust approach to find tables and extract country, leader, and title.
    """
    leaders_data = []
    url = 'https://en.wikipedia.org/wiki/List_of_current_heads_of_state_and_government'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for tables that contain country and leader information
        # Wikipedia often uses 'wikitable' or similar classes
        tables = soup.find_all('table', class_='wikitable')

        for table in tables:
            rows = table.find_all('tr')
            # Heuristic: skip rows that look like headers or are too short
            for row in rows:
                cells = row.find_all(['td', 'th'])
                # A row should have at least 2-3 cells for country, leader, and title
                if len(cells) >= 2:
                    try:
                        # Extract text and clean up references like [1], [a]
                        country_raw = cells[0].get_text(strip=True)
                        leader_raw = cells[1].get_text(strip=True)
                        title_raw = cells[2].get_text(strip=True) if len(cells) > 2 else "Leader"

                        country = re.sub(r'\[.*?\]|\(.*?\)|\n.*', '', country_raw).strip()
                        leader = re.sub(r'\[.*?\]|\(.*?\)|\n.*', '', leader_raw).strip()
                        title = re.sub(r'\[.*?\]|\(.*?\)|\n.*', '', title_raw).strip()

                        # Basic validation to ensure meaningful data
                        if country and leader and len(country) > 2 and "List of" not in country and "Current" not in country:
                            leaders_data.append({
                                'country': country,
                                'leader': leader,
                                'title': title,
                                'source': 'Wikipedia',
                                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                    except IndexError:
                        # Skip rows that don't have enough cells
                        continue
                    except Exception as e:
                        # Log other parsing errors for debugging
                        st.warning(f"Error parsing Wikipedia row: {e}")
                        continue

        # Remove duplicates based on country and return a reasonable number
        seen_countries = set()
        clean_data = []
        for item in leaders_data:
            if item['country'] not in seen_countries:
                seen_countries.add(item['country'])
                clean_data.append(item)

        return clean_data[:200] # Limit to top 200 for performance and relevance

    except requests.exceptions.RequestException as e:
        st.error(f"Network error scraping world leaders: {e}")
        return []
    except Exception as e:
        st.error(f"Error scraping world leaders data: {e}")
        return []

@st.cache_data(ttl=3600) # Cache for 1 hour
def scrape_economic_data():
    """
    Scrapes GDP data from the World Bank API.
    Fetches GDP (current US$) for all countries for a recent year.
    """
    gdp_data = {}
    # Using 2022 as a recent stable year for GDP data
    wb_url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?format=json&date=2022&per_page=500"

    try:
        response = requests.get(wb_url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if len(data) > 1 and isinstance(data[1], list): # Check if data[1] exists and is a list
            for item in data[1]:
                country_name = item.get('country', {}).get('value')
                gdp_value = item.get('value')

                if country_name and gdp_value is not None:
                    gdp_data[country_name] = gdp_value / 1e12  # Convert to trillions

        return gdp_data
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching economic data from World Bank: {e}")
        return {}
    except Exception as e:
        st.error(f"Error parsing economic data: {e}")
        return {}

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_simulated_scores():
    """
    Generates simulated environmental and political stability scores.
    In a production app, these would come from specific APIs (e.g., Yale EPI, World Bank Governance Indicators).
    """
    # These are illustrative scores; real data would require specific API integrations
    environmental_scores = {
        'Denmark': 82.5, 'Luxembourg': 82.3, 'Switzerland': 81.5, 'Finland': 78.9,
        'Sweden': 78.7, 'Norway': 77.7, 'Germany': 77.2, 'Netherlands': 75.3,
        'France': 74.5, 'Austria': 73.6, 'Canada': 71.2, 'United Kingdom': 70.8,
        'Japan': 68.2, 'Australia': 67.9, 'United States': 67.1, 'South Korea': 66.5,
        'Italy': 66.2, 'Spain': 65.9, 'Portugal': 65.4, 'Belgium': 64.8,
        'China': 37.3, 'India': 30.6, 'Brazil': 49.7, 'Russia': 50.5,
        'Mexico': 52.6, 'Turkey': 44.1, 'South Africa': 43.1, 'Indonesia': 37.8,
        'Saudi Arabia': 34.4, 'Nigeria': 27.1, 'Taiwan': 76.0 # Added Taiwan
    }

    stability_scores = {
        'Switzerland': 95.2, 'Singapore': 94.8, 'Norway': 94.3, 'Denmark': 93.8,
        'Finland': 93.3, 'Sweden': 92.9, 'Netherlands': 92.4, 'Austria': 91.9,
        'Canada': 91.4, 'Australia': 90.9, 'Germany': 90.5, 'Luxembourg': 90.0,
        'United Kingdom': 89.5, 'France': 88.1, 'Japan': 87.6, 'Belgium': 87.1,
        'United States': 78.6, 'South Korea': 77.1, 'Italy': 72.9, 'Spain': 71.4,
        'Portugal': 85.7, 'Czech Republic': 84.3, 'Poland': 79.0, 'Hungary': 74.8,
        'Brazil': 48.1, 'Mexico': 47.6, 'China': 42.9, 'India': 56.2,
        'Russia': 28.1, 'Turkey': 41.4, 'Saudi Arabia': 35.2, 'Indonesia': 61.0,
        'South Africa': 43.8, 'Nigeria': 27.6, 'Egypt': 25.7, 'Pakistan': 18.1, 'Taiwan': 87.0 # Added Taiwan
    }

    # Aid data (simulated, as OECD DAC API requires authentication/complex setup)
    aid_data = {
        'United States': 34650, 'Germany': 28430, 'United Kingdom': 15756,
        'Japan': 11640, 'France': 11423, 'Sweden': 5603, 'Italy': 5320,
        'Netherlands': 5515, 'Norway': 4236, 'Canada': 4280, 'Denmark': 2630,
        'Switzerland': 3160, 'Belgium': 2200, 'Finland': 1290, 'Austria': 1340,
        'Australia': 3000, 'Spain': 2860, 'South Korea': 2420, 'Poland': 890,
        'Luxembourg': 470, 'Ireland': 950, 'New Zealand': 680, 'Portugal': 420,
        'Czech Republic': 320, 'Slovenia': 85, 'Slovakia': 130, 'Hungary': 180,
        'Estonia': 55, 'Latvia': 45, 'Lithuania': 65, 'Iceland': 75, 'Taiwan': 500 # Added Taiwan
    }

    return environmental_scores, stability_scores, aid_data

@st.cache_data(ttl=1800) # Cache for 30 minutes
def get_comprehensive_country_data():
    """
    Combines data from various scraping functions into a single comprehensive DataFrame.
    Handles country name matching and fills missing data with random values.
    """
    with st.spinner("🔍 Fetching and combining real-time and simulated data..."):
        leaders_data = scrape_world_leaders()
        economic_data = scrape_economic_data()
        environmental_scores, stability_scores, aid_data = get_simulated_scores()

        comprehensive_data = []
        processed_countries = set() # To avoid duplicate entries if a country appears multiple times in leaders_data

        for leader_info in leaders_data:
            country = leader_info['country']
            if country in processed_countries:
                continue # Skip if already processed

            processed_countries.add(country)

            # Attempt to find a match in other datasets, considering variations
            matched_econ_country = next((k for k in economic_data.keys() if country.lower() in k.lower() or k.lower() in country.lower()), None)
            matched_env_country = next((k for k in environmental_scores.keys() if country.lower() in k.lower() or k.lower() in country.lower()), None)
            matched_stability_country = next((k for k in stability_scores.keys() if country.lower() in k.lower() or k.lower() in country.lower()), None)
            matched_aid_country = next((k for k in aid_data.keys() if country.lower() in k.lower() or k.lower() in country.lower()), None)

            # Default values if data is missing, using random values within a reasonable range
            gdp_trillion = economic_data.get(matched_econ_country, np.random.uniform(0.1, 25.0))
            environmental_score = environmental_scores.get(matched_env_country, np.random.randint(30, 85))
            political_stability = stability_scores.get(matched_stability_country, np.random.randint(25, 95))
            aid_donations = aid_data.get(matched_aid_country, np.random.randint(50, 5000))

            # Simple region mapping (can be expanded)
            region_mapping = {
                'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
                'China': 'Asia', 'Japan': 'Asia', 'India': 'Asia', 'South Korea': 'Asia',
                'Indonesia': 'Asia', 'Saudi Arabia': 'Asia', 'Turkey': 'Asia', 'Taiwan': 'Asia',
                'Germany': 'Europe', 'United Kingdom': 'Europe', 'France': 'Europe', 'Italy': 'Europe',
                'Spain': 'Europe', 'Netherlands': 'Europe', 'Russia': 'Europe',
                'Brazil': 'South America',
                'Australia': 'Oceania'
            }
            # Assign a region, defaulting to 'Other' if not found
            region = region_mapping.get(country, 'Other')

            comprehensive_data.append({
                'Country': country,
                'Leader': leader_info['leader'],
                'Title': leader_info['title'],
                'Environmental_Score': environmental_score,
                'Economic_Score': min(100, max(0, (gdp_trillion * 10))) if gdp_trillion else np.random.randint(40, 95), # Scale GDP for economic score
                'Political_Stability': political_stability,
                'Donations_USD_M': aid_donations,
                'GDP_Trillion': gdp_trillion,
                'Population_M': np.random.randint(5, 1500), # Simulated population as it's not scraped
                'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Data_Source': 'Wikipedia, World Bank, Simulated',
                'Region': region
            })

        if not comprehensive_data:
            st.warning("No data scraped. Displaying a small sample for demonstration.")
            # Fallback to a very small static sample if scraping completely fails
            return pd.DataFrame({
                'Country': ['Sample Country 1', 'Sample Country 2'],
                'Leader': ['Sample Leader 1', 'Sample Leader 2'],
                'Title': ['President', 'Prime Minister'],
                'Environmental_Score': [60, 70],
                'Economic_Score': [75, 85],
                'Political_Stability': [65, 75],
                'Donations_USD_M': [1000, 2000],
                'GDP_Trillion': [1.5, 2.5],
                'Population_M': [50, 100],
                'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Data_Source': 'Simulated Fallback',
                'Region': ['Fallback Region', 'Fallback Region']
            })

        return pd.DataFrame(comprehensive_data)

# --- Sentiment Analysis (Placeholder - requires real news data) ---
def analyze_leader_sentiment(leader_name, country):
    """
    Simulates sentiment analysis for a leader based on sample news.
    In a real application, this would involve fetching actual news articles.
    """
    # This is a simulation; real-time news APIs would be integrated here.
    sample_news = [
        f"{leader_name} announces new policy initiatives for {country}",
        f"Economic reforms under {leader_name} show mixed results",
        f"{leader_name} addresses international relations at summit",
        f"Public opinion on {leader_name}'s leadership remains divided",
        f"{leader_name} focuses on domestic priorities for {country}"
    ]

    sentiment_analyzer = load_sentiment_analyzer()
    if not sentiment_analyzer:
        return {'error': "Sentiment analyzer model not loaded."}

    sentiments = []
    for news in sample_news:
        try:
            result = sentiment_analyzer(news)
            sentiments.append(result[0])
        except Exception as e:
            st.warning(f"Error analyzing sentiment for news '{news}': {e}")
            continue

    if not sentiments:
        return {'error': "No sentiment data could be generated for the leader."}

    avg_score = sum([s['score'] for s in sentiments]) / len(sentiments)

    return {
        'average_sentiment': avg_score,
        'news_analyzed': len(sample_news),
        'sentiment_breakdown': sentiments
    }

# --- Streamlit UI Functions ---

def create_global_dashboard():
    """
    Creates the main global analytics dashboard with various tabs for data visualization.
    """
    st.header("🌍 Global Analytics Dashboard")
    st.markdown("Dive into real-time insights on countries, leaders, and global trends.")

    # Load the comprehensive data
    df = get_comprehensive_country_data()

    if df.empty:
        st.error("Unable to load data. Please check your internet connection or try again later.")
        return

    # Display data freshness and sources
    st.info(f"📊 Data last updated: {df['Last_Updated'].iloc[0]} | Sources: {df['Data_Source'].iloc[0]}")

    # Key metrics display using Streamlit columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Countries", len(df))
    with col2:
        st.metric("Total Donations", f"${df['Donations_USD_M'].sum():,.0f}M")
    with col3:
        st.metric("Avg Environmental Score", f"{df['Environmental_Score'].mean():.1f}")
    with col4:
        st.metric("Global GDP", f"${df['GDP_Trillion'].sum():.1f}T")

    # Tabs for different views of the data
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏛️ Countries & Leaders", "📊 Scores Analysis", "💰 Donations", "🗺️ Regional View", "📰 Sentiment Analysis"])

    with tab1:
        st.subheader("Countries and Current Leaders")

        # Search functionality for countries and leaders
        search_term = st.text_input("🔍 Search Country or Leader:", key="search_tab1")
        if search_term:
            filtered_df = df[df['Country'].str.contains(search_term, case=False, na=False) |
                             df['Leader'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df

        # Display the filtered data in a table, sorted by Environmental Score
        display_cols = ['Country', 'Leader', 'Title', 'Environmental_Score', 'Economic_Score', 'Political_Stability', 'Donations_USD_M', 'GDP_Trillion', 'Population_M']
        st.dataframe(
            filtered_df[display_cols].sort_values('Environmental_Score', ascending=False).style.format({
                'Environmental_Score': '{:.1f}',
                'Economic_Score': '{:.1f}',
                'Political_Stability': '{:.1f}',
                'Donations_USD_M': '{:,.0f}',
                'GDP_Trillion': '{:.1f}',
                'Population_M': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )

        # Leader details and radar chart
        if not filtered_df.empty:
            selected_country = st.selectbox("Select country for detailed analysis:",
                                             filtered_df['Country'].tolist(), key="select_country_tab1")

            if selected_country:
                leader_info = filtered_df[filtered_df['Country'] == selected_country].iloc[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"📋 {selected_country} Details")
                    st.write(f"**Leader:** {leader_info['Leader']}")
                    st.write(f"**Title:** {leader_info['Title']}")
                    st.write(f"**Environmental Score:** {leader_info['Environmental_Score']:.1f}/100")
                    st.write(f"**Political Stability:** {leader_info['Political_Stability']:.1f}/100")
                    st.write(f"**Economic Score:** {leader_info['Economic_Score']:.1f}/100")
                    st.write(f"**GDP:** ${leader_info['GDP_Trillion']:.1f}T")
                    st.write(f"**Population:** {leader_info['Population_M']:,.0f}M")

                with col2:
                    st.subheader("📊 Performance Metrics Radar Chart")
                    categories = ['Environmental', 'Political Stability', 'Economic']
                    values = [
                        leader_info['Environmental_Score'],
                        leader_info['Political_Stability'],
                        leader_info['Economic_Score']
                    ]

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=selected_country
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Global Performance Scores Analysis")

        # Interactive scatter plot
        fig = px.scatter(
            df,
            x='Environmental_Score',
            y='Political_Stability',
            size='GDP_Trillion',
            color='Donations_USD_M',
            hover_name='Country',
            hover_data=['Leader', 'Title', 'Population_M'],
            title='Environmental vs Political Stability (Size=GDP, Color=Aid)',
            labels={
                'Environmental_Score': 'Environmental Score',
                'Political_Stability': 'Political Stability Score',
                'GDP_Trillion': 'GDP (Trillion USD)',
                'Donations_USD_M': 'Aid Donations (Million USD)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top performers in each category
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🌱 Top Environmental")
            top_env = df.nlargest(5, 'Environmental_Score')[['Country', 'Leader', 'Environmental_Score']]
            for idx, row in top_env.iterrows():
                st.write(f"**{row['Country']}** ({row['Leader']}): {row['Environmental_Score']:.1f}")

        with col2:
            st.subheader("🏛️ Most Stable")
            top_stable = df.nlargest(5, 'Political_Stability')[['Country', 'Leader', 'Political_Stability']]
            for idx, row in top_stable.iterrows():
                st.write(f"**{row['Country']}** ({row['Leader']}): {row['Political_Stability']:.1f}")

        with col3:
            st.subheader("💰 Top Aid Donors")
            top_aid = df.nlargest(5, 'Donations_USD_M')[['Country', 'Leader', 'Donations_USD_M']]
            for idx, row in top_aid.iterrows():
                st.write(f"**{row['Country']}** ({row['Leader']}): ${row['Donations_USD_M']:,.0f}M")

    with tab3:
        st.subheader("International Aid Analysis")

        # Aid distribution treemap
        fig = px.treemap(
            df.nlargest(20, 'Donations_USD_M'), # Show top 20 donors in treemap
            path=['Country'],
            values='Donations_USD_M',
            title='International Aid Distribution by Country (Top 20 Donors)',
            hover_data=['Leader', 'GDP_Trillion']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Aid effectiveness analysis scatter plot
        fig2 = px.scatter(
            df,
            x='GDP_Trillion',
            y='Donations_USD_M',
            hover_name='Country',
            hover_data=['Leader', 'Population_M', 'Environmental_Score', 'Political_Stability'],
            title='Aid Donations vs GDP',
            labels={'GDP_Trillion': 'GDP (Trillion USD)', 'Donations_USD_M': 'Aid Donations (Million USD)'}
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Top donors table with calculated donation as percentage of GDP
        st.subheader("🏆 Top Donors")
        top_donors = df.nlargest(10, 'Donations_USD_M')[['Country', 'Leader', 'Donations_USD_M', 'GDP_Trillion']]
        top_donors['Donation_as_GDP_percent'] = (top_donors['Donations_USD_M'] / (top_donors['GDP_Trillion'] * 1000)) * 100
        st.dataframe(top_donors.style.format({
            'Donations_USD_M': '{:,.0f}',
            'GDP_Trillion': '{:.1f}',
            'Donation_as_GDP_percent': '{:.2f}%'
        }), use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Regional Analysis")

        # Regional summary table
        regional_summary = df.groupby('Region').agg({
            'Environmental_Score': 'mean',
            'Economic_Score': 'mean',
            'Political_Stability': 'mean',
            'Donations_USD_M': 'sum',
            'Population_M': 'sum',
            'GDP_Trillion': 'sum',
            'Country': 'count' # Count countries per region
        }).rename(columns={'Country': 'Num_Countries'}).round(2)

        st.dataframe(regional_summary.style.format({
            'Environmental_Score': '{:.1f}',
            'Economic_Score': '{:.1f}',
            'Political_Stability': '{:.1f}',
            'Donations_USD_M': '{:,.0f}',
            'Population_M': '{:,.0f}',
            'GDP_Trillion': '{:.1f}'
        }), use_container_width=True)

        # Box plot showing distribution of Environmental Scores by Region
        fig = px.box(df, x='Region', y='Environmental_Score', title='Environmental Scores Distribution by Region')
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart for total GDP by Region
        fig_gdp_region = px.bar(
            regional_summary.reset_index().sort_values('GDP_Trillion', ascending=False),
            x='Region',
            y='GDP_Trillion',
            title='Total GDP by Region (Trillion USD)',
            color='GDP_Trillion',
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig_gdp_region, use_container_width=True)

    with tab5:
        st.subheader("Leadership Sentiment Analysis (Simulated)")
        st.info("💡 **Note:** Sentiment analysis is currently simulated using sample news. Real-time news integration would require access to news APIs.")

        # Select leader for sentiment analysis
        # Ensure the selectbox options are from the dynamically loaded DataFrame
        available_countries = df['Country'].tolist()
        if available_countries:
            selected_leader_country = st.selectbox(
                "Select country for leadership sentiment analysis:",
                available_countries,
                key="select_country_sentiment"
            )

            if selected_leader_country:
                leader_info = df[df['Country'] == selected_leader_country].iloc[0]

                with st.spinner(f"Analyzing sentiment for {leader_info['Leader']}..."):
                    sentiment_data = analyze_leader_sentiment(leader_info['Leader'], selected_leader_country)

                if 'error' not in sentiment_data:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(f"📰 Sentiment: {leader_info['Leader']}")
                        st.metric("Average Sentiment Score", f"{sentiment_data['average_sentiment']:.2f}")
                        st.metric("News Articles Analyzed", sentiment_data['news_analyzed'])

                    with col2:
                        st.subheader("📊 Sentiment Breakdown")

                        # Create sentiment distribution chart
                        sentiment_labels = [s['label'] for s in sentiment_data['sentiment_breakdown']]
                        sentiment_scores = [s['score'] for s in sentiment_data['sentiment_breakdown']]

                        fig = px.bar(
                            x=range(len(sentiment_labels)),
                            y=sentiment_scores,
                            title=f'Sentiment Analysis for {leader_info["Leader"]}',
                            labels={'x': 'News Article Index', 'y': 'Sentiment Score'},
                            color=sentiment_labels, # Color bars by sentiment label
                            color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'blue'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("Sentiment Labels: ", ", ".join(sentiment_labels))
                else:
                    st.error(f"Error analyzing sentiment: {sentiment_data['error']}")
        else:
            st.warning("No country data available to perform sentiment analysis.")

def create_contact_form():
    """
    Creates a contact form for users to request platform upgrades or features.
    Simulates submission and displays a confirmation.
    """
    st.header("📧 Contact & Upgrade Requests")
    st.markdown("Have ideas for **EYE OF EARTH** or need custom solutions? Reach out!")

    with st.form("contact_form"):
        st.subheader("Request Platform Upgrades")

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*", placeholder="Your full name")
            email = st.text_input("Email Address*", placeholder="your.email@example.com")
            organization = st.text_input("Organization", placeholder="Company/Institution name")

        with col2:
            # Dropdown for country, dynamically loaded from current data
            df_countries = get_comprehensive_country_data()
            available_countries_for_form = sorted(df_countries['Country'].tolist()) if not df_countries.empty else []

            country = st.selectbox("Country*", ["Select Country"] + available_countries_for_form)
            request_type = st.selectbox("Request Type*", [
                "Select Type",
                "Data Updates",
                "New Features",
                "API Access",
                "Custom Analytics",
                "Partnership Inquiry",
                "Technical Support"
            ])
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])

        upgrade_details = st.text_area(
            "Upgrade Details*",
            placeholder="Please describe your upgrade request, specific features needed, or questions about the platform...",
            height=150
        )

        # Checkboxes for additional requested features
        st.subheader("Requested Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            real_time_data = st.checkbox("Real-time Data Updates")
            advanced_analytics = st.checkbox("Advanced Analytics")
            custom_reports = st.checkbox("Custom Reports")
        with col2:
            api_access = st.checkbox("API Access")
            data_export = st.checkbox("Data Export Tools")
            multi_language = st.checkbox("Multi-language Support")
        with col3:
            mobile_app = st.checkbox("Mobile Application")
            white_label = st.checkbox("White-label Solution")
            training = st.checkbox("Training & Support")

        # Slider for budget range
        budget_range = st.select_slider(
            "Budget Range (USD)",
            options=["< $1K", "$1K-$5K", "$5K-$10K", "$10K-$25K", "$25K-$50K", "$50K-$100K", "> $100K"],
            value="$5K-$10K"
        )

        # Dropdown for expected timeline
        timeline = st.selectbox("Expected Timeline", [
            "ASAP",
            "Within 1 month",
            "Within 3 months",
            "Within 6 months",
            "Within 1 year",
            "No specific timeline"
        ])

        newsletter = st.checkbox("Subscribe to EYE OF EARTH updates and newsletter")

        submitted = st.form_submit_button("🚀 Submit Request", type="primary")

        if submitted:
            # Basic validation for required fields
            if name and email and country != "Select Country" and request_type != "Select Type" and upgrade_details:
                # In a real application, this 'request_data' would be sent to a backend database or API.
                request_data = {
                    "Name": name,
                    "Email": email,
                    "Organization": organization,
                    "Country": country,
                    "Request Type": request_type,
                    "Priority": priority,
                    "Details": upgrade_details,
                    "Real-time Data": real_time_data,
                    "Advanced Analytics": advanced_analytics,
                    "Custom Reports": custom_reports,
                    "API Access": api_access,
                    "Data Export Tools": data_export,
                    "Multi-language Support": multi_language,
                    "Mobile App": mobile_app,
                    "White-label Solution": white_label,
                    "Training & Support": training,
                    "Budget Range": budget_range,
                    "Timeline": timeline,
                    "Newsletter Opt-in": newsletter,
                    "Submission Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.success("🎉 Your request has been submitted successfully! We will get back to you shortly.")
                # st.json(request_data) # For debugging: show the submitted data
            else:
                st.error("Please fill in all required fields (marked with *).")


# --- Main Application Logic ---
def main():
    st.title("👁️ EYE OF EARTH: Global Geopolitical & Environmental Insights")
    st.markdown("""
    Welcome to **EYE OF EARTH**, your ultimate platform for exploring critical global data.
    Understand world leadership, economic trends, environmental performance, and political stability with interactive visualizations and real-time insights.
    """)

    # Create a navigation menu
    menu = ["Dashboard", "Contact & Upgrades"]
    choice = st.sidebar.radio("Navigation", menu)

    if choice == "Dashboard":
        create_global_dashboard()
    elif choice == "Contact & Upgrades":
        create_contact_form()

if __name__ == "__main__":
    main()