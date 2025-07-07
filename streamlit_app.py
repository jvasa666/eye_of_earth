import streamlit as st
import qrcode
from io import BytesIO
from PIL import Image

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Support XenoTech",
    page_icon="☕",
    layout="centered" # "wide" also works, depending on overall app design
)

# --- Custom CSS for Styling ---
# This CSS will make buttons look nicer and provide some spacing
st.markdown(
    """
    <style>
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
        background-color: #f0f2f6; /* Light gray background for code */
        padding: 8px;
        border-radius: 5px;
        font-family: monospace;
        word-break: break-all; /* Ensure long addresses wrap */
        white-space: pre-wrap; /* Preserve whitespace and wrap */
        font-size: 0.9em;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Function to generate QR code as a Streamlit image ---
def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# --- Main App Content ---
st.title("💖 Support XenoTech Development 💖")
st.markdown("""
Thank you for considering supporting **XenoTech**! Your contributions, big or small,
help keep this project alive, fund further development, and cover operational costs.
Every bit of support is deeply appreciated and fuels future innovations.
""")

st.markdown("---") # Visual separator

st.markdown('<p class="section-header">☕ Buy Me a Coffee</p>', unsafe_allow_html=True)
st.markdown("""
The easiest way to show your appreciation is through a virtual coffee!
""")
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


st.markdown("---") # Visual separator

st.markdown('<p class="section-header">💸 Crypto Donations</p>', unsafe_allow_html=True)
st.markdown("""
For those who prefer to support with cryptocurrency, here are the addresses.
""")

# --- Crypto Addresses with QR Codes and Copy Buttons ---

# ETH Address
st.subheader("Ethereum (ETH)")
eth_address = "0x5036dbcEEfae0a7429e64467222e1E259819c7C7"
col_eth1, col_eth2 = st.columns([1, 2])
with col_eth1:
    st.image(generate_qr_code(eth_address), width=150, caption="Scan for ETH")
with col_eth2:
    st.markdown("Address:")
    st.markdown(f'<div class="crypto-address">{eth_address}</div>', unsafe_allow_html=True)
    if st.button("Copy ETH Address", key="copy_eth"):
        st.code(eth_address, language="text") # Display in a code block for easy copy (Streamlit's native copy)
        st.success("ETH address copied to clipboard (see code block above)!")


st.markdown("---") # Separator for crypto sections

# BTC Address
st.subheader("Bitcoin (BTC)")
btc_address = "bc1qzncgc94kgtcpumx80m5uedsp3hqp4fec2e3rvr" # Replace with your actual BTC address
col_btc1, col_btc2 = st.columns([1, 2])
with col_btc1:
    st.image(generate_qr_code(btc_address), width=150, caption="Scan for BTC")
with col_btc2:
    st.markdown("Address:")
    st.markdown(f'<div class="crypto-address">{btc_address}</div>', unsafe_allow_html=True)
    if st.button("Copy BTC Address", key="copy_btc"):
        st.code(btc_address, language="text")
        st.success("BTC address copied to clipboard (see code block above)!")

st.markdown("---") # Separator for crypto sections

# Solana Address
st.subheader("Phantom/Solana (SOL)")
sol_address = "7ckfzhhkwkpdTRHdXoEorD5gN3Yg6ggaTHw2B6gF6hKq" # Replace with your actual SOL address
col_sol1, col_sol2 = st.columns([1, 2])
with col_sol1:
    st.image(generate_qr_code(sol_address), width=150, caption="Scan for SOL")
with col_sol2:
    st.markdown("Address:")
    st.markdown(f'<div class="crypto-address">{sol_address}</div>', unsafe_allow_html=True)
    if st.button("Copy SOL Address", key="copy_sol"):
        st.code(sol_address, language="text")
        st.success("SOL address copied to clipboard (see code block above)!")

st.markdown("---") # Separator for social links

st.markdown('<p class="section-header">🔗 Connect with XenoTech</p>', unsafe_allow_html=True)
st.markdown("""
Stay updated and connect with the community:
""")

st.markdown("""
- 🌐 [Farcaster Profile](https://warpcast.com/xenotech)
- 📢 [Telegram Channel](https://t.me/xenodrop)
- 🔗 [Follow Lens](https://lens.xyz/u/xenotech)
""")

st.markdown("---")
st.info("A huge **THANK YOU** for your generosity and support!")