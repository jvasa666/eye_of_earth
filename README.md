# 🌍 Eye of Earth — Global Leader Transparency System

**Real-time AI-powered dashboard for global citizen oversight.**

Eye of Earth monitors the world’s top leaders using live data scraped directly from official government domains. It uses advanced language models to summarize content, detect sentiment, assign trust and truth scores, and flag potential crises — so the people of Earth always know where their leaders stand.

---

## 🛰️ Features

- 🔍 Live scraping of government domains (UN, US, UK, China, India, etc.)
- 🧠 AI summarization via HuggingFace Transformers
- 📊 Per-leader sentiment, trust score, and truth index
- 🚨 Crisis keyword detection
- 📬 Email subscription support (Brevo)
- 💬 Public feature request input form
- ☁️ Streamlit-powered live web interface

---

## 🔗 Try It Live

🟢 **Streamlit App:**  
https://eyeofearth.streamlit.app  
*(replace with your actual link if different)*

---

## ⚙️ Run Locally

Clone this repo and install dependencies:

```bash
git clone https://github.com/jvasa666/eye_of_earth.git
cd eye_of_earth
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📦 Requirements

These are listed in `requirements.txt` and auto-installed by Streamlit or pip:

```
aiohttp
requests
pandas
geopandas
streamlit
plotly
beautifulsoup4
langdetect
transformers
huggingface_hub
scikit-learn
torch
```

---

## 🧠 Powered By

- 🤖 HuggingFace Transformers (`distilbart`, `distilbert`)
- 📡 Streamlit Cloud
- 💌 Brevo SMTP
- 🧠 AI-driven scoring logic
- 📂 SQLite (optional logging)

---

## 💬 Suggest Features

Use the app’s “💬 Suggest a Feature” input box  
or open a GitHub issue:  
https://github.com/jvasa666/eye_of_earth/issues

---

## ☕ Support the Mission

No ads. No trackers. No bias.

If you support global truth and transparency:
- ⭐ Star this repo
- 💬 Share feedback
- 🧠 Spread the word

---

## 📜 License

MIT License  
Free for public use, remix, or contribution.

---

## 👨‍🚀 Credits

- Commander: [@jvasa666](https://github.com/jvasa666)  
- AI Tactical Unit: Gemini Protocol  
- Mission: Transparency for All Earth Citizens 🌍
