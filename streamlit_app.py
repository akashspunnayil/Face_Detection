# streamlit_app.py
import streamlit as st
from PIL import Image


# --- Configuration ---
st.set_page_config(page_title="AKASH.S", layout="wide")

# --- Custom Sidebar Navigation ---
st.markdown("""
<style>
/* Ensure sidebar content is always visible in both themes */

/* Target the entire sidebar */
section[data-testid="stSidebar"] {
    background-color: transparent;
}

/* Sidebar headings and radio text */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: inherit !important;  /* ✨ ADAPTS TO THEME */
    font-size: 16px;
    font-weight: 500;
}

/* Optional: Make selected radio option bold and colored */
div[role="radiogroup"] > label[data-testid="stRadioOption"]:has(input:checked) {
    font-weight: bold;
    color: #0066cc !important;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("## 🧭 Navigation")


# --- Cover Image ---
from PIL import Image
#cover = Image.open("static/cover2.png")
##st.image(cover, use_container_width=True)
#resized_cover = cover.resize((800, 300))  # (width, height)
#st.image(resized_cover)


# --- Projects Page ---
from PIL import Image
from io import BytesIO
import base64

# --- Load Shared Preview Image ---
def get_base64_image(path):
    img = Image.open(path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

shared_img_base64 = get_base64_image("static/preview/preview.jpeg")  # Your current shared image


# --- Tile Renderer with Preview ---
#def render_tile(title, url, description, img_base64):
def render_tile(title, url, description, img_base64):
    return f"""
    <div style="
        display: flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 10px 0;
        width: 100%;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    ">
        <img src="data:image/png;base64,{img_base64}" alt="{title}" style="
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 10px;
            margin-right: 20px;
            flex-shrink: 0;
        "/>
        <div style="flex-grow: 1;">
            <div style="font-size: 16px; font-weight: 600; color: #fff; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 13px; color: #ccc; margin-bottom: 4px;">{description}</div>
            <a href="{url}" style="font-size: 13px; text-decoration: none; color: #00BFFF;">→ Launch</a>
        </div>
    </div>
    """


    
nav_options = ["Projects"]
#menu = st.sidebar.radio("Navigation", nav_options, index=0)
menu = st.sidebar.radio(" ", nav_options, index=0)


if menu == "Projects":
    st.subheader("🧠 Computer Vision Projects")

    # Load base64 image
    import base64
    with open("static/preview/preview.jpeg", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    # Display ribbon-style tile
    st.markdown(render_tile(
        title="Face Detection App",
        url="/?app=face-detection",
        description="Streamlit webcam face detection with Haar cascades.",
        img_base64=img_base64
    ), unsafe_allow_html=True)

    # Query param logic
    query_params = st.experimental_get_query_params()
    if query_params.get("app", [None])[0] == "face-detection":
        from apps.face_detection_app import run_face_detection
        run_face_detection()

        
        
