import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import base64
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===========================
# COOL TONE COLOR PALETTE
# ===========================
COLORS = {
    'primary_bg': '#2D2822',
    'secondary_bg': '#3A3530',
    'tertiary_bg': '#45403B',
    'accent_light': '#B8AFA4',
    'accent_mid': '#9D9589',
    'accent_dark': '#6B6157',
    'text_light': '#E8E3DC',
    'text_mid': '#D1CCC4',
    'border': '#7A736A',
    'button_base': 'rgba(58, 53, 48, 0.75)',
    'button_hover': 'rgba(157, 149, 137, 0.95)',
    'button_border': 'rgba(122, 115, 106, 0.8)',
}

# ===========================
# CONSTANTS - DEPLOYMENT READY
# ===========================
# Get the directory where this script is located
BASE_DIR = Path(__file__).parent

# Use relative paths that work both locally and in deployment
IMAGE_PATH = BASE_DIR / "bagg.png"
MASTER_CSV_PATH = BASE_DIR / "features_master_dataset.csv"

IMAGE_NATIVE_W = 1728
IMAGE_NATIVE_H = 1080

# Investigation sections
INVESTIGATION_SECTIONS = [
    {"key": "overview", "icon": "🎒", "label": "The Case File", "subtitle": "Opening the investigation"},
    {"key": "persona", "icon": "👤", "label": "Meet Jane", "subtitle": "The person behind the data"},
    {"key": "wallet", "icon": "💳", "label": "Financial Footprints", "subtitle": "Credit card confessions"},
    {"key": "receipts", "icon": "🧾", "label": "Store Receipts", "subtitle": "Paper trail evidence"},
    {"key": "phone", "icon": "📱", "label": "Digital Emotions", "subtitle": "Twitter tells all"},
    {"key": "earphones", "icon": "🎵", "label": "The Soundtrack", "subtitle": "Music as a mood map"},
    {"key": "meds", "icon": "⌚", "label": "The Body's Truth", "subtitle": "Wearable witness"},
    {"key": "cigs_lighter", "icon": "🎯", "label": "The Prediction", "subtitle": "Tomorrow's truth, today"},
    {"key": "methodology", "icon": "📚", "label": "The Evidence Log", "subtitle": "How we know what we know"},
]

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="What's In My Bag - An Intimate Investigation",
    page_icon="🎒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================
# FORCE DARK THEME GLOBALLY
# ===========================
st.markdown(f"""
<style>
    /* Force dark theme colors */
    .stApp {{
        background-color: {COLORS['primary_bg']} !important;
        color: {COLORS['text_light']} !important;
    }}
    
    /* All text elements */
    .stMarkdown, p, span, div, label, .stTextInput, .stSelectbox {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['secondary_bg']} !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Buttons - ensure text is visible */
    .stButton > button {{
        color: {COLORS['primary_bg']} !important;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {COLORS['accent_light']} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {COLORS['accent_mid']} !important;
    }}
    
    /* Code blocks */
    .stCodeBlock, code {{
        background-color: {COLORS['secondary_bg']} !important;
        color: {COLORS['text_light']} !important;
    }}
    
    /* Tables */
    .dataframe {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Input fields */
    input {{
        color: {COLORS['text_light']} !important;
        background-color: {COLORS['secondary_bg']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ===========================
# SESSION STATE INITIALIZATION
# ===========================
if 'visited_items' not in st.session_state:
    st.session_state.visited_items = set()

if 'welcome_shown' not in st.session_state:
    st.session_state.welcome_shown = False

if 'investigation_started' not in st.session_state:
    st.session_state.investigation_started = False

# ===========================
# UTILITY FUNCTIONS
# ===========================
def setup_earthy_plot_style():
    """Configure matplotlib for cool brown theme"""
    plt.rcParams.update({
        'figure.facecolor': COLORS['primary_bg'],
        'axes.facecolor': COLORS['secondary_bg'],
        'axes.edgecolor': COLORS['accent_light'],
        'axes.labelcolor': COLORS['text_light'],
        'text.color': COLORS['text_light'],
        'xtick.color': COLORS['accent_light'],
        'ytick.color': COLORS['accent_light'],
        'grid.color': COLORS['border'],
        'grid.alpha': 0.3,
        'legend.facecolor': COLORS['secondary_bg'],
        'legend.edgecolor': COLORS['border'],
    })
    cool_colors = [COLORS['accent_light'], COLORS['accent_mid'], COLORS['accent_dark'], 
                   '#9D9589', '#B8AFA4', '#6B6157', '#C4BCB2', '#8A8278']
    return cool_colors

def setup_plotly_theme():
    """Configure Plotly for cool brown theme"""
    return {
        'plot_bgcolor': COLORS['secondary_bg'],
        'paper_bgcolor': COLORS['primary_bg'],
        'font': {'color': COLORS['text_light'], 'family': 'Arial'},
        'xaxis': {'gridcolor': COLORS['border'], 'linecolor': COLORS['accent_light']},
        'yaxis': {'gridcolor': COLORS['border'], 'linecolor': COLORS['accent_light']},
    }

@st.cache_data
def load_master():
    """Load the master dataset with error handling"""
    try:
        if MASTER_CSV_PATH.exists():
            df = pd.read_csv(MASTER_CSV_PATH)
            
            total_rows = len(df)
            missing_pct = (df.isnull().sum() / total_rows * 100).round(2)
            
            quality_report = {
                'total_rows': total_rows,
                'total_cols': len(df.columns),
                'missing_data': missing_pct[missing_pct > 0].to_dict() if (missing_pct > 0).any() else None,
                'date_range': None
            }
            
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    quality_report['date_range'] = {
                        'start': df['date'].min().strftime('%Y-%m-%d'),
                        'end': df['date'].max().strftime('%Y-%m-%d')
                    }
                except:
                    pass
            
            return df, None, quality_report
        else:
            return None, f"Master dataset not found at: {MASTER_CSV_PATH}", None
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}", None

def get_image_base64():
    """Convert image to base64 for embedding"""
    try:
        if IMAGE_PATH.exists():
            with open(IMAGE_PATH, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        else:
            st.error(f"Image not found at: {IMAGE_PATH}")
            return None
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return None

def get_next_prev_items(current_key):
    """Get next and previous items for navigation"""
    current_idx = next((i for i, h in enumerate(INVESTIGATION_SECTIONS) if h['key'] == current_key), None)
    if current_idx is None:
        return None, None
    
    prev_item = INVESTIGATION_SECTIONS[current_idx - 1] if current_idx > 0 else None
    next_item = INVESTIGATION_SECTIONS[current_idx + 1] if current_idx < len(INVESTIGATION_SECTIONS) - 1 else None
    
    return prev_item, next_item

def show_welcome_dialog():
    """Display intimate welcome dialog"""
    
    # Apply styling
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {COLORS['primary_bg']} 0%, {COLORS['secondary_bg']} 100%);
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Center content
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center; font-size: 42px; font-weight: 300; letter-spacing: 2px; margin-bottom: 30px;'>What's In My Bag?</h1>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; font-size: 18px; line-height: 1.9; margin-bottom: 25px;'>Hi, I'm <strong>Yasmine</strong>. This isn't just another data science portfolio.</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; font-size: 18px; line-height: 1.9; margin-bottom: 25px;'>Every day, we carry pieces of our lives. A wallet holds choices. A phone contains emotions. Earphones soundtrack our internal weather.</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; font-size: 18px; line-height: 1.9; margin-bottom: 25px;'>Meet <strong>Jane Doe</strong>—35, trying to quit smoking. Her credit card knows when she's stressed. Her Spotify reveals when sadness creeps in. Her fitness tracker records sleepless nights.</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; font-size: 18px; line-height: 1.9; margin-bottom: 25px;'>Together, we'll analyze Jane's data as intimate moments, not cold numbers. We'll build models that predict tomorrow's behavior from today's signals, and ask uncomfortable questions about privacy and algorithmic surveillance.</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; font-style: italic; font-size: 17px; margin-top: 30px; margin-bottom: 40px;'>\"How well can an algorithm know you? Should it?\"</p>", unsafe_allow_html=True)
        
        if st.button("🔍 Open the Bag", use_container_width=True, type="primary", key="begin_btn"):
            st.session_state.welcome_shown = True
            st.rerun()

def render_progress_sidebar():
    """Render progress tracker in sidebar"""
    with st.sidebar:
        st.markdown("### 🔍 Investigation Progress")
        
        total_items = len(INVESTIGATION_SECTIONS)
        visited_count = len(st.session_state.visited_items)
        progress = visited_count / total_items
        
        st.progress(progress)
        st.markdown(f"**{visited_count}/{total_items} pieces examined**")
        
        st.markdown("---")
        st.markdown("### 📋 Evidence Board")
        
        for i, section in enumerate(INVESTIGATION_SECTIONS, 1):
            status = "✅" if section['key'] in st.session_state.visited_items else "⭕"
            st.markdown(f"{status} **{i}.** {section['label']}")
            if section['key'] in st.session_state.visited_items:
                st.caption(f"   ↳ {section['subtitle']}")

# ===========================
# LANDING PAGE WITH BAG
# ===========================
def render_landing():
    """Render the bag landing page with corner note and centered button"""
    
    if not st.session_state.welcome_shown:
        show_welcome_dialog()
        return
    
    img_base64 = get_image_base64()
    if not img_base64:
        st.error("Image not found. Please check the path.")
        return
    
    st.markdown(f"""
    <style>
        .stApp {{
            background: url(data:image/png;base64,{img_base64}) center center / cover no-repeat fixed;
            color: {COLORS['text_light']} !important;
        }}
        
        .main > div {{
            padding: 0 !important;
            min-height: 100vh;
        }}
        
        .corner-note {{
            position: fixed;
            top: 25px;
            left: 25px;
            background: rgba(45, 40, 34, 0.9);
            padding: 20px 25px;
            border-radius: 8px;
            max-width: 300px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(122, 115, 106, 0.7);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
            z-index: 100;
        }}
        
        .corner-note-title {{
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['accent_light']};
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }}
        
        .corner-note-text {{
            font-size: 13px;
            line-height: 1.6;
            color: {COLORS['text_mid']};
        }}
        
        .button-spacer {{
            height: 45vh;
        }}
        
        .center-button-container {{
            text-align: center;
            max-width: 400px;
            margin: 0 auto;
        }}
        
        .stButton > button {{
            background: rgba(157, 149, 137, 0.95) !important;
            color: {COLORS['primary_bg']} !important;
            border: 2px solid rgba(232, 227, 220, 0.9) !important;
            border-radius: 50px !important;
            font-weight: 700 !important;
            font-size: 20px !important;
            padding: 18px 50px !important;
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.6);
            transition: all 0.3s ease;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.8), 0 0 40px rgba(184, 175, 164, 0.5);
            background: rgba(184, 175, 164, 1) !important;
            border-color: rgba(232, 227, 220, 1) !important;
        }}
        
        .portfolio-badge {{
            position: fixed;
            bottom: 25px;
            right: 25px;
            background: rgba(45, 40, 34, 0.85);
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 11px;
            color: {COLORS['text_mid']};
            border: 1px solid rgba(122, 115, 106, 0.5);
            backdrop-filter: blur(5px);
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Corner note
    st.markdown("""
    <div class="corner-note">
        <div class="corner-note-title">🔍 The Investigation</div>
        <div class="corner-note-text">
            Inside this bag lies the evidence—Jane's daily life captured in data. 
            Credit cards, playlists, tweets, and wearables all tell part of her story.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add vertical spacer to push button down
    st.markdown('<div class="button-spacer"></div>', unsafe_allow_html=True)
    
    # Centered button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown('<div class="center-button-container">', unsafe_allow_html=True)
        if st.button("Begin Investigation", key="start_investigation", use_container_width=True):
            st.session_state.investigation_started = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Portfolio badge
    st.markdown("""
    <div class="portfolio-badge">
        💡 Interactive Portfolio<br>
        Data Science × Storytelling
    </div>
    """, unsafe_allow_html=True)

# ===========================
# INVESTIGATION DASHBOARD
# ===========================
def render_dashboard():
    """Render the investigation dashboard with all sections"""
    
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {COLORS['primary_bg']} 0%, {COLORS['secondary_bg']} 50%, {COLORS['tertiary_bg']} 100%) !important;
        }}
        
        .dashboard-header {{
            text-align: center;
            padding: 40px 20px;
            margin-bottom: 40px;
        }}
        
        .dashboard-title {{
            font-size: 42px;
            color: {COLORS['text_light']};
            margin-bottom: 15px;
            font-weight: 300;
            letter-spacing: 2px;
        }}
        
        .dashboard-subtitle {{
            font-size: 18px;
            color: {COLORS['text_mid']};
            font-weight: 300;
            line-height: 1.6;
        }}
        
        .section-card {{
            background: rgba(58, 53, 48, 0.6);
            border: 2px solid {COLORS['border']};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
            backdrop-filter: blur(10px);
        }}
        
        .section-card:hover {{
            background: rgba(58, 53, 48, 0.85);
            border-color: {COLORS['accent_light']};
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}
        
        .section-icon {{
            font-size: 40px;
            margin-bottom: 15px;
        }}
        
        .section-title {{
            font-size: 22px;
            color: {COLORS['text_light']};
            margin-bottom: 8px;
            font-weight: 500;
        }}
        
        .section-subtitle {{
            font-size: 14px;
            color: {COLORS['accent_mid']};
            font-style: italic;
        }}
        
        .visited-badge {{
            display: inline-block;
            background: {COLORS['accent_dark']};
            color: {COLORS['text_light']};
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            margin-top: 10px;
        }}
        
        .completion-message {{
            background: rgba(69, 64, 59, 0.9);
            border: 3px solid {COLORS['accent_light']};
            border-radius: 15px;
            padding: 40px;
            margin: 30px auto;
            max-width: 800px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        }}
        
        .completion-title {{
            font-size: 36px;
            color: {COLORS['accent_light']};
            margin-bottom: 20px;
            font-weight: 300;
            letter-spacing: 1px;
        }}
        
        .completion-text {{
            font-size: 17px;
            line-height: 1.9;
            color: {COLORS['text_mid']};
            margin-bottom: 15px;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    render_progress_sidebar()
    
    # Check if all sections have been visited
    all_visited = len(st.session_state.visited_items) == len(INVESTIGATION_SECTIONS)
    
    if all_visited:
        st.markdown("""
        <div class="completion-message">
            <div class="completion-title">🎉 Investigation Complete</div>
            <p class="completion-text">
                You've examined every piece of evidence in Jane's bag. You've seen her financial footprints, 
                her emotional patterns, her musical moods, and her body's testimony. You've built predictive models 
                and confronted the ethical implications of behavioral surveillance.
            </p>
            <p class="completion-text">
                <strong>Thank you</strong> for taking this journey with me. This project represents not just technical skills, 
                but a commitment to responsible data science—building systems that work <em>and</em> questioning whether they should exist.
            </p>
            <p class="completion-text">
                Jane's story may be synthetic, but the questions are real: <em>How much should algorithms know about us? 
                Where's the line between insight and intrusion? Can prediction ever truly serve the person being predicted?</em>
            </p>
            <p class="completion-text" style="margin-top: 30px; font-size: 15px; font-style: italic; color: {COLORS['accent_light']};">
                "In the end, data doesn't tell stories. People do. And every dataset is a person, whether we see them or not."
            </p>
            <p class="completion-text" style="margin-top: 30px; font-size: 14px;">
                — Yasmine
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Jane's Evidence Board</h1>
        <p class="dashboard-subtitle">
            Nine pieces of evidence. Each one a window into Jane's daily life.<br>
            Click any section to begin your analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create 3x3 grid
    for row in range(3):
        cols = st.columns(3)
        for col_idx in range(3):
            section_idx = row * 3 + col_idx
            if section_idx < len(INVESTIGATION_SECTIONS):
                section = INVESTIGATION_SECTIONS[section_idx]
                
                with cols[col_idx]:
                    visited = section['key'] in st.session_state.visited_items
                    visited_badge = '<span class="visited-badge">✓ Examined</span>' if visited else ''
                    
                    st.markdown(f"""
                    <div class="section-card">
                        <div class="section-icon">{section['icon']}</div>
                        <div class="section-title">{section['label']}</div>
                        <div class="section-subtitle">{section['subtitle']}</div>
                        {visited_badge}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Investigate →", key=f"btn_{section['key']}", use_container_width=True):
                        st.query_params["item"] = section["key"]
                        st.session_state.visited_items.add(section["key"])
                        st.rerun()

# ===========================
# PANEL RENDERER
# ===========================
def render_panel(item_key):
    """Render analysis panel for specific item"""
    
    st.session_state.visited_items.add(item_key)
    
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {COLORS['primary_bg']} 0%, {COLORS['secondary_bg']} 50%, {COLORS['tertiary_bg']} 100%) !important;
            color: {COLORS['text_light']} !important;
        }}
        .main > div {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px !important;
            background: rgba(45, 40, 34, 0.3);
            border-radius: 10px;
        }}
        
        h1, h2, h3 {{ color: {COLORS['text_light']} !important; }}
        .markdown-text-container, .stMarkdown, p, li {{ color: {COLORS['text_mid']} !important; }}
        [data-testid="stMetricValue"] {{ color: {COLORS['accent_light']} !important; }}
        [data-testid="stMetricLabel"] {{ color: {COLORS['accent_mid']} !important; }}
        .stAlert {{ background-color: rgba(58, 53, 48, 0.8) !important; color: {COLORS['text_light']} !important; border-color: {COLORS['border']} !important; }}
        .breadcrumb {{ color: {COLORS['accent_light']}; margin-bottom: 20px; font-size: 14px; }}
        
        .narrative-intro {{
            background: rgba(58, 53, 48, 0.7);
            border-left: 4px solid {COLORS['accent_light']};
            padding: 20px 25px;
            margin: 25px 0;
            border-radius: 8px;
            font-style: italic;
            color: {COLORS['text_mid']};
            line-height: 1.8;
        }}
        
        .insight-box {{
            background: rgba(69, 64, 59, 0.8);
            border: 2px solid {COLORS['accent_mid']};
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .insight-title {{
            color: {COLORS['accent_light']};
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 16px;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    section = next((s for s in INVESTIGATION_SECTIONS if s["key"] == item_key), None)
    if not section:
        st.error("Unknown section")
        return
    
    section_num = next((i for i, s in enumerate(INVESTIGATION_SECTIONS) if s['key'] == item_key), 0) + 1
    st.markdown(f'<div class="breadcrumb">🎒 Evidence Board → Piece {section_num} of {len(INVESTIGATION_SECTIONS)} → {section["label"]}</div>', unsafe_allow_html=True)
    
    prev_item, next_item = get_next_prev_items(item_key)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if prev_item and st.button(f"← {prev_item['label']}", key="prev_btn", use_container_width=True):
            st.query_params["item"] = prev_item["key"]
            st.rerun()
    
    with col2:
        if st.button("🎒 Back to Evidence Board", key="back_to_bag", use_container_width=True):
            st.query_params.clear()
            st.rerun()
    
    with col3:
        if next_item and st.button(f"{next_item['label']} →", key="next_btn", use_container_width=True):
            st.query_params["item"] = next_item["key"]
            st.rerun()
    
    st.title(f"{section['icon']} {section['label']}")
    st.caption(section['subtitle'])
    
    render_progress_sidebar()
    
    with st.spinner("Analyzing evidence..."):
        df, error, quality_report = load_master()
    
    if error and item_key not in ["overview", "persona", "methodology"]:
        st.error(error)
        return
    
    if item_key == "overview":
        render_overview_panel(quality_report)
    elif item_key == "persona":
        render_persona_panel()
    elif item_key == "wallet":
        render_wallet_panel(df)
    elif item_key == "receipts":
        render_receipts_panel(df)
    elif item_key == "phone":
        render_phone_panel(df)
    elif item_key == "earphones":
        render_earphones_panel(df)
    elif item_key == "meds":
        render_meds_panel(df)
    elif item_key == "cigs_lighter":
        render_model_panel(df)
    elif item_key == "methodology":
        render_methodology_panel(df, quality_report)

# ===========================
# PANEL FUNCTIONS WITH INTIMATE NARRATIVE
# ===========================
def render_overview_panel(quality_report):
    """Project overview panel with intimate framing"""
    
    st.markdown("""
    <div class="narrative-intro">
    Every forensic investigation begins with a case file. This is ours.
    <br><br>
    Jane doesn't know we're watching. Well, not the real Jane—she's a synthetic persona, a ghost made of algorithms. 
    But her data patterns are real. They mirror thousands of people struggling with addiction, tracked by their phones, 
    their banks, their wearables. Jane is everyone and no one.
    <br><br>
    Our mission: predict tomorrow's cigarette purchase using only today's behavioral signals. Can we catch the craving 
    before it catches her?
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎯 The Question")
    st.markdown("""
    **Will Jane smoke tomorrow?**
    
    Not "might she" or "could she"—but *will she*. A binary answer from infinite human complexity. 
    This is the hubris and promise of predictive analytics.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Data Sources")
        st.markdown("""
        - **Financial**: Credit cards, store receipts
        - **Emotional**: Twitter sentiment analysis
        - **Cultural**: Spotify listening history
        - **Physical**: Sleep, steps, heart rate
        """)
    with col2:
        st.markdown("### 🔬 Approach")
        st.markdown("""
        - **Temporal lag**: Today predicts tomorrow
        - **No cheating**: Spending data excluded
        - **Class balance**: Smoking days are rare
        - **Interpretability**: Feature importance revealed
        """)
    
    if quality_report:
        st.markdown("## 📈 The Dataset")
        st.markdown("""
        <div class="narrative-intro">
        Numbers never lie, but they rarely tell the whole truth. Here's what we're working with:
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Days Observed", f"{quality_report['total_rows']:,}")
            st.caption("Each row is one day in Jane's life")
        with col2:
            st.metric("Behavioral Signals", quality_report['total_cols'])
            st.caption("Features capturing digital existence")
        with col3:
            if quality_report['date_range']:
                days = (pd.to_datetime(quality_report['date_range']['end']) - pd.to_datetime(quality_report['date_range']['start'])).days
                st.metric("Time Span", f"{days} days")
                st.caption("Enough to spot patterns")
    
    st.markdown("## 🎓 Why This Matters")
    st.markdown("""
    This project demonstrates:
    
    **Technical Skills**
    - Multi-source data integration with temporal alignment
    - Sophisticated feature engineering (lag transformations, leakage prevention)
    - Class-balanced binary classification with proper validation
    - Model comparison and interpretability analysis
    
    **Conceptual Depth**
    - Behavioral prediction as both tool and ethical minefield
    - The intimacy of data: what our digital traces reveal about us
    - The gap between prediction and understanding
    
    **Communication**
    - Translating technical work into human stories
    - Making ethics tangible, not abstract
    - Building engaging, interactive experiences
    """)
    
    st.info("💡 **Ethical Note**: All data is synthetically generated. Jane is a fictional persona created to explore real methodological and ethical questions in behavioral prediction.")

def render_persona_panel():
    """Intimate persona introduction"""
    
    st.markdown("""
    <div class="narrative-intro">
    Let me introduce you to Jane Doe. Not the data, not the numbers—the person.
    <br><br>
    She's 35. She works in an office where the coffee is bad and the stress is chronic. She's been smoking since college—
    a decade and a half of lighting up when life gets heavy. She wants to quit. She's tried before. She'll try again.
    <br><br>
    This year, she's tracking everything. Not for us—for herself. The Fitbit on her wrist, the banking app that judges 
    her Friday night purchases, the Spotify playlist titled "getting through it." She's trying to understand her own patterns, 
    hoping that visibility will lead to control.
    <br><br>
    We're using her data to predict her behavior. She would've consented to this study. But would she have understood what 
    that consent meant? That's the question that haunts this entire project.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 👤 Profile: Jane Doe")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### The Person")
        st.markdown("""
        **Age**: 35  
        **Occupation**: Marketing coordinator  
        **Smoking History**: 15 years, pack-a-day habit  
        **Quit Attempts**: Multiple, longest streak 6 months  
        **Current Goal**: Reduce to zero by year-end
        
        **Personality Snapshot**:
        - Self-aware about triggers but struggles with follow-through
        - Uses technology to gamify self-improvement
        - Social smoker turned stress smoker
        - High achiever, perfectionist tendencies
        """)
    
    with col2:
        st.markdown("### 🔴 Triggers (What We'll Track)")
        st.markdown("""
        **Emotional**
        - Negative Twitter sentiment
        - Sad music patterns
        - Social isolation indicators
        
        **Physical**
        - Poor sleep (<6 hours)
        - Low activity days
        - Elevated resting heart rate
        
        **Behavioral**
        - Alcohol purchases
        - Late-night activity
        - Weekend patterns
        """)
    
    st.markdown("### 🟢 Protective Factors")
    st.markdown("""
    Jane isn't just fighting cravings—she's building defenses:
    
    - **Nicotine patches**: Used on high-stress days
    - **Activity tracking**: Gamification of health
    - **Music therapy**: Curated "mood boost" playlists
    - **Health monitoring**: Feedback loop of progress
    """)
    
    st.markdown("## 📱 The Data Sources")
    st.markdown("""
    <div class="insight-box">
    <div class="insight-title">🔍 What We're Actually Measuring</div>
    
    **Credit Card Transactions**: Not just spending—but *when* and *what*. Convenience store at 11pm? Red flag. 
    Bar tab on Tuesday? Another piece of the puzzle.
    
    **Twitter Activity**: Sentiment analysis of her tweets. Cynical humor spikes on hard days. Radio silence on the worst ones.
    
    **Spotify History**: Valence scores (musical happiness). Her playlist shifts from upbeat morning jams to melancholic 
    evening tracks predict trouble.
    
    **Health Wearable**: Sleep, steps, heart rate. The body keeps score when the mind won't admit struggle.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔒 A Note on Privacy")
    st.markdown("""
    <div class="narrative-intro">
    This depth of surveillance—because let's call it what it is—raises profound questions:
    
    **Is prediction help or harm?** If we can predict Jane's relapse, can we intervene? Should we? 
    At what point does support become control?
    
    **Who owns this knowledge?** Jane's insurance company would love this model. So would her employer. 
    The line between care and exploitation is razor-thin.
    
    **What does consent mean?** Jane checked a box. But did she understand she was consenting to be *known* 
    at this level? To have her tomorrow predicted by her today?
    
    We'll return to these questions. They don't have clean answers.
    </div>
    """, unsafe_allow_html=True)
    
    st.success("✨ **Remember**: Jane Doe is synthetic, but these ethical dilemmas are not. Every dataset has a person behind it, whether we meet them or not.")

def render_wallet_panel(df):
    """Financial analysis with intimate narrative"""
    if df is None:
        return
    
    earthy_colors = setup_earthy_plot_style()
    theme = setup_plotly_theme()
    
    st.markdown("""
    <div class="narrative-intro">
    A wallet is a diary written in receipts. Every swipe tells a story—where you were, what you needed, what you were feeling.
    <br><br>
    Jane's credit card data shows us something stark: her spending patterns change on smoking days. Not dramatically—she's not 
    buying cartons—but subtly. A convenience store stop that wouldn't happen otherwise. A bar tab that suggests 
    she's drinking, and drinking means smoking.
    <br><br>
    But here's the thing: we can't use this data in our prediction model. Why? Because the cigarette purchase itself appears 
    in the spending records. Using spending to predict spending would be circular logic—a model cheating on its own test.
    <br><br>
    Still, let's look. Sometimes understanding *why* we can't use data is as important as using it.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💳 The Alcohol-Cigarette Connection")
    
    with st.spinner("Cross-referencing transactions..."):
        try:
            if 'alcohol_purchase_day' in df.columns and 'cigarette_purchase_day' in df.columns:
                st.markdown("""
                **Detective's Question**: *Does Jane light up after drinking? Let's check the receipts.*
                """)
                
                cross_tab = pd.crosstab(df['alcohol_purchase_day'], df['cigarette_purchase_day'])
                fig = px.imshow(cross_tab, labels=dict(x="Cigarette Purchase", y="Alcohol Purchase", color="Days"),
                               title='Co-occurrence Matrix: Alcohol & Cigarettes', 
                               color_continuous_scale='Greys', text_auto=True)
                fig.update_layout(**theme)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate co-occurrence rate
                total_alcohol_days = df['alcohol_purchase_day'].sum()
                alcohol_and_cigs = ((df['alcohol_purchase_day'] == 1) & (df['cigarette_purchase_day'] == 1)).sum()
                if total_alcohol_days > 0:
                    co_occur_rate = (alcohol_and_cigs / total_alcohol_days) * 100
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <div class="insight-title">🔍 The Pattern Emerges</div>
                    On days when Jane bought alcohol, she also bought cigarettes <strong>{co_occur_rate:.1f}%</strong> of the time.
                    <br><br>
                    This isn't causation—alcohol doesn't "make" Jane smoke. But it's a powerful correlation. Alcohol lowers inhibitions, 
                    weakens resolve, creates social contexts where smoking feels normal. For someone trying to quit, that beer at happy hour 
                    isn't just a drink. It's a risk factor.
                    <br><br>
                    <em>Jane probably knows this pattern herself. But knowing and avoiding aren't the same thing.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Nicotine patch analysis
            st.markdown("## 🩹 The Patch Strategy")
            if 'nicotine_patch_day' in df.columns and 'cigarette_purchase_day' in df.columns:
                st.markdown("""
                **Detective's Theory**: *Jane uses nicotine patches on high-risk days. Is this prevention working?*
                """)
                
                # Cross-tabulation of patches vs cigarettes
                patch_cig_cross = pd.crosstab(df['nicotine_patch_day'], df['cigarette_purchase_day'])
                
                fig = px.imshow(patch_cig_cross, 
                               labels=dict(x="Cigarette Purchase", y="Nicotine Patch Used", color="Days"),
                               title='Nicotine Patch Use vs Cigarette Purchase', 
                               color_continuous_scale='Greys', text_auto=True)
                fig.update_layout(**theme)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate effectiveness
                patch_days = df['nicotine_patch_day'].sum()
                patch_and_smoke = ((df['nicotine_patch_day'] == 1) & (df['cigarette_purchase_day'] == 1)).sum()
                no_patch_days = len(df) - patch_days
                no_patch_and_smoke = ((df['nicotine_patch_day'] == 0) & (df['cigarette_purchase_day'] == 1)).sum()
                
                if patch_days > 0 and no_patch_days > 0:
                    smoke_rate_with_patch = (patch_and_smoke / patch_days) * 100
                    smoke_rate_without_patch = (no_patch_and_smoke / no_patch_days) * 100
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <div class="insight-title">🔍 Patch Effectiveness Analysis</div>
                    <strong>On days Jane used a patch</strong>: She smoked <strong>{smoke_rate_with_patch:.1f}%</strong> of the time
                    <br>
                    <strong>On days without a patch</strong>: She smoked <strong>{smoke_rate_without_patch:.1f}%</strong> of the time
                    <br><br>
                    {"<em>The patch appears to help—smoking rates are lower when she uses it. But it's not perfect. Even with pharmacological support, willpower still plays a role.</em>" if smoke_rate_with_patch < smoke_rate_without_patch else "<em>Interestingly, Jane smokes at similar rates with or without the patch. This might indicate she only uses patches AFTER she's already smoked, or that patches alone aren't enough without behavioral support.</em>"}
                    <br><br>
                    <strong>Usage frequency</strong>: Jane used patches on <strong>{patch_days}</strong> out of <strong>{len(df)}</strong> days 
                    (<strong>{(patch_days/len(df)*100):.1f}%</strong> of the time). This suggests sporadic rather than consistent use—
                    possibly deployed reactively when cravings are already strong.
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("## ⚠️ The Leakage Problem")
            st.markdown(f"""
            <div style="background: rgba(139, 69, 19, 0.2); border-left: 4px solid {COLORS['accent_dark']}; padding: 20px; border-radius: 5px; margin-top: 30px;">
            <strong>Methodological Note</strong>
            <br><br>
            All credit card spending features are <strong>excluded from our predictive model</strong>. Here's why:
            <br><br>
            <strong>The Leakage Problem</strong>: When Jane buys cigarettes, that purchase appears in her credit card data. If we use 
            spending patterns to predict smoking, we're essentially using the outcome variable to predict itself. The model would learn 
            "cigarette purchases predict cigarette purchases"—accurate but useless.
            <br><br>
            <strong>What about alcohol and patches?</strong> These are <em>behavioral</em> variables we CAN use—they're not the outcome itself, 
            but related behaviors that might precede smoking. However, we still need to be careful: if alcohol is purchased at the same 
            convenience store visit as cigarettes, that's still leakage.
            <br><br>
            <strong>The Lesson</strong>: Good data science isn't just about accuracy. It's about *meaningful* predictions. We need to 
            predict tomorrow's purchase from today's *behavioral signals*—not from data that already contains the answer.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analysis error: {e}")

def render_receipts_panel(df):
    """Store receipts with narrative depth"""
    if df is None:
        return
    
    theme = setup_plotly_theme()
    
    st.markdown("""
    <div class="narrative-intro">
    Convenience stores are strange liminal spaces—not quite grocery stores, not quite gas stations. They're where you go 
    when you need something *now*. A pack of cigarettes. A energy drink at 2am. Milk because you forgot.
    <br><br>
    For Jane, convenience store visits are markers of unplanned moments. A deviation from routine. These aren't the 
    carefully considered purchases of a weekly grocery trip. These are impulse, need, craving.
    <br><br>
    The data shows us something interesting: Jane's convenience store spending spikes irregularly. These aren't weekly 
    patterns—they're chaos. And in that chaos, we find the cigarettes.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🧾 The Convenience Store Pattern")
    
    with st.spinner("Analyzing purchase patterns..."):
        try:
            if 'cc_spend_convenience' in df.columns and 'date' in df.columns:
                st.markdown("""
                **Detective's Observation**: *Tracking the unplanned stops—where habits hide in plain sight.*
                """)
                
                df_plot = df.copy()
                df_plot['date'] = pd.to_datetime(df_plot['date'], errors='coerce')
                df_plot = df_plot.dropna(subset=['date']).sort_values('date')
                df_plot['rolling'] = df_plot['cc_spend_convenience'].rolling(7, min_periods=1).mean()
                
                fig = px.line(df_plot, x='date', y='rolling', 
                             title='Convenience Store Spending: 7-Day Moving Average',
                             labels={'rolling': 'Average Daily Spending ($)', 'date': 'Date'})
                fig.update_traces(line_color=COLORS['accent_dark'], line_width=3)
                fig.update_layout(**theme)
                st.plotly_chart(fig, use_container_width=True)
                
                avg_spending = df_plot['cc_spend_convenience'].mean()
                max_spending = df_plot['cc_spend_convenience'].max()
                spike_days = (df_plot['cc_spend_convenience'] > avg_spending * 2).sum()
                
                st.markdown(f"""
                <div class="insight-box">
                <div class="insight-title">🔍 What the Numbers Tell Us</div>
                <strong>Baseline behavior</strong>: Jane averages <strong>${avg_spending:.2f}</strong> per day at convenience stores. 
                This includes the occasional coffee, snacks, necessities.
                <br><br>
                <strong>Spike days</strong>: On <strong>{spike_days} occasions</strong>, her spending jumped to over <strong>${avg_spending * 2:.2f}</strong>—
                double her normal pattern. These outliers aren't random. They correlate with smoking days.
                <br><br>
                <strong>The invisible purchase</strong>: A pack of cigarettes costs around $8-12. It blends into convenience store totals. 
                Without itemized receipts, we're inferring the purchase from the pattern. The cigarettes hide in the noise.
                <br><br>
                <em>This is how surveillance works: not through confession, but through inference.</em>
                </div>
                """, unsafe_allow_html=True)
                
            st.warning("""
            ⚠️ **Data Leakage Alert**: Like credit card data, convenience store spending is NOT used in our predictive model. 
            Why? Because it contains the signal we're trying to predict—the cigarette purchase itself. 
            
            We're showing you this analysis to understand Jane's behavior, not to build the model. The model must predict 
            *before* the purchase, using only prior behavioral signals.
            """)
            
        except Exception as e:
            st.error(f"Analysis error: {e}")

def render_phone_panel(df):
    """Twitter sentiment with emotional depth"""
    if df is None:
        return
    
    st.markdown("""
    <div class="narrative-intro">
    Social media is performative, but it's not fake. Jane's tweets aren't lies—they're curated truths. She tweets when 
    she needs to be heard, when silence would be worse. And the *way* she tweets—the sentiment, the frequency, the time of day—
    reveals patterns she might not consciously recognize.
    <br><br>
    Sentiment analysis is crude. It reduces human expression to positive/negative/neutral scores. It misses sarcasm, context, 
    poetry. But crudeness has its uses. When you analyze hundreds of tweets, the noise cancels out. What remains is signal: 
    the emotional baseline, and how it shifts.
    <br><br>
    Jane tweets differently on smoking days. Not dramatically—she doesn't announce her relapses. But the tone shifts. 
    The darkness creeps in at the edges.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📱 The Emotional Fingerprint")
    theme = setup_plotly_theme()
    
    with st.spinner("Analyzing digital emotions..."):
        try:
            if 'avg_compound' in df.columns and 'cigarette_purchase_day' in df.columns:
                st.markdown("""
                **Detective's Theory**: *Negative emotion precedes relapse. Let's test it against the data.*
                """)
                
                df_clean = df.dropna(subset=['avg_compound', 'cigarette_purchase_day']).copy()
                df_clean['Status'] = df_clean['cigarette_purchase_day'].map({0: 'Non-Smoking Days', 1: 'Smoking Days'})
                
                fig = px.box(df_clean, x='Status', y='avg_compound', 
                            title='Twitter Sentiment Distribution by Smoking Status',
                            color='Status', 
                            color_discrete_map={'Non-Smoking Days': COLORS['accent_light'], 'Smoking Days': COLORS['accent_dark']})
                fig.update_layout(**theme, showlegend=False)
                fig.update_yaxes(title_text='Compound Sentiment (-1=Very Negative, +1=Very Positive)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate statistics
                no_cig_sentiment = df_clean[df_clean['cigarette_purchase_day'] == 0]['avg_compound'].mean()
                cig_sentiment = df_clean[df_clean['cigarette_purchase_day'] == 1]['avg_compound'].mean()
                
                if cig_sentiment < no_cig_sentiment:
                    diff_pct = abs((no_cig_sentiment - cig_sentiment) / no_cig_sentiment) * 100
                    st.markdown(f"""
                    <div class="insight-box">
                    <div class="insight-title">🔍 The Emotional Gap</div>
                    On smoking days, Jane's average sentiment score drops to <strong>{cig_sentiment:.3f}</strong>, compared to 
                    <strong>{no_cig_sentiment:.3f}</strong> on non-smoking days. That's a <strong>{diff_pct:.1f}%</strong> decrease 
                    in expressed positivity.
                    <br><br>
                    <strong>What this means</strong>: Jane doesn't tweet "I'm having a cigarette." She tweets things like "exhausted", 
                    "why is everything", "never mind." The algorithm catches what she doesn't say directly—that she's struggling.
                    <br><br>
                    <em>This is the intimacy of data: it knows you're suffering before you find the words.</em>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="insight-box">
                    <div class="insight-title">🔍 The Unexpected Pattern</div>
                    Interestingly, Jane's sentiment on smoking days (<strong>{cig_sentiment:.3f}</strong>) isn't dramatically different 
                    from non-smoking days (<strong>{no_cig_sentiment:.3f}</strong>).
                    <br><br>
                    <strong>What this suggests</strong>: Maybe Jane doesn't tweet when she's at her lowest. Maybe smoking isn't always 
                    about sadness—sometimes it's habit, boredom, social context. Or maybe her coping mechanism is to stay silent online 
                    when things are bad.
                    <br><br>
                    <em>Absence of signal is also signal. What we don't see matters too.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Tweet volume analysis
            if 'tweets_count' in df.columns:
                st.markdown("### 📊 Volume vs. Silence")
                st.markdown("*Does Jane tweet more or less on smoking days?*")
                
                df_tweets = df.dropna(subset=['tweets_count', 'cigarette_purchase_day']).copy()
                df_tweets['Status'] = df_tweets['cigarette_purchase_day'].map({0: 'Non-Smoking Days', 1: 'Smoking Days'})
                
                avg_tweets = df_tweets.groupby('Status')['tweets_count'].mean()
                
                fig = px.bar(avg_tweets.reset_index(), x='Status', y='tweets_count',
                            title='Average Daily Tweet Count',
                            color='Status',
                            color_discrete_map={'Non-Smoking Days': COLORS['accent_mid'], 'Smoking Days': COLORS['accent_dark']})
                fig.update_layout(**theme, showlegend=False)
                fig.update_yaxes(title_text='Average Tweets Per Day')
                st.plotly_chart(fig, use_container_width=True)
                
                no_cig_tweets = avg_tweets.get('Non-Smoking Days', 0)
                cig_tweets = avg_tweets.get('Smoking Days', 0)
                
                st.markdown(f"""
                <div class="insight-box">
                <div class="insight-title">💡 Activity Patterns</div>
                Jane tweets an average of <strong>{cig_tweets:.1f}</strong> times on smoking days versus <strong>{no_cig_tweets:.1f}</strong> 
                on non-smoking days.
                <br><br>
                {"<em>She's more active online when struggling—using social media as an outlet, a distraction, or a cry for connection.</em>" if cig_tweets > no_cig_tweets else "<em>She withdraws online when struggling—silence as a symptom of overwhelm.</em>"}
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analysis error: {e}")

def render_earphones_panel(df):
    """Spotify analysis with cultural intimacy"""
    if df is None:
        return
    
    st.markdown("""
    <div class="narrative-intro">
    Music is emotion made tangible. We don't just *hear* songs—we *use* them. To amplify joy, to process sadness, to escape, 
    to remember. Your Spotify history is a diary of your internal weather.
    <br><br>
    Jane's playlists shift with her mood. On good days: upbeat pop, energetic indie, songs that make you want to move. 
    On hard days: melancholic indie, sad girl autumn, songs that hold space for grief.
    <br><br>
    Spotify measures something called "valence"—musical positivity. A score from 0 (sad) to 1 (happy) based on tempo, 
    mode, key. It's reductive, but it works. And Jane's valence scores tell a story she might not realize she's telling.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎵 The Soundtrack of Struggle")
    theme = setup_plotly_theme()
    
    with st.spinner("Analyzing listening patterns..."):
        try:
            valence_col = 'avg_valence_adjusted' if 'avg_valence_adjusted' in df.columns else 'avg_valence'
            
            if valence_col in df.columns and 'cigarette_purchase_day' in df.columns:
                st.markdown("""
                **Detective's Hypothesis**: *Sad music predicts sad behavior. Let's look at the valence scores.*
                """)
                
                df_clean = df.dropna(subset=[valence_col, 'cigarette_purchase_day']).copy()
                df_clean['Status'] = df_clean['cigarette_purchase_day'].map({0: 'Non-Smoking Days', 1: 'Smoking Days'})
                
                fig = px.box(df_clean, x='Status', y=valence_col, 
                            title='Musical Valence by Smoking Status',
                            color='Status', 
                            color_discrete_map={'Non-Smoking Days': COLORS['accent_mid'], 'Smoking Days': COLORS['accent_dark']})
                fig.update_layout(**theme, showlegend=False)
                fig.update_yaxes(title_text='Valence Score (0=Sad, 1=Happy)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate stats
                no_cig_valence = df_clean[df_clean['cigarette_purchase_day'] == 0][valence_col].mean()
                cig_valence = df_clean[df_clean['cigarette_purchase_day'] == 1][valence_col].mean()
                
                if cig_valence < no_cig_valence:
                    diff = ((no_cig_valence - cig_valence) / no_cig_valence) * 100
                    st.markdown(f"""
                    <div class="insight-box">
                    <div class="insight-title">🔍 The Emotional Soundtrack</div>
                    On smoking days, Jane's music is <strong>{diff:.1f}% sadder</strong>. Her average valence score drops from 
                    <strong>{no_cig_valence:.3f}</strong> (non-smoking) to <strong>{cig_valence:.3f}</strong> (smoking).
                    <br><br>
                    <strong>What we're seeing</strong>: Jane doesn't consciously think "I feel bad, let me queue up sad songs, then buy cigarettes." 
                    It's more subtle. She wakes up heavy. Reaches for music that matches her mood. The day unfolds from there. The cigarette 
                    comes later, almost inevitable.
                    <br><br>
                    <strong>The pattern</strong>: Sad music → rumination → weakened resolve → relapse.
                    <br><br>
                    <em>Music doesn't cause the smoking. But it's an early warning system—a canary in the coal mine of her emotional state.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Track mood breakdown
            if 'avg_happy_tracks' in df.columns and 'avg_sad_tracks' in df.columns:
                st.markdown("### 🎭 Playlist Composition Analysis")
                st.markdown("*Breaking down the ratio of happy, neutral, and sad tracks...*")
                
                mood_cols = ['avg_happy_tracks', 'avg_neutral_tracks', 'avg_sad_tracks']
                available_mood = [col for col in mood_cols if col in df.columns]
                
                if available_mood and 'cigarette_purchase_day' in df.columns:
                    df_mood = df.dropna(subset=available_mood + ['cigarette_purchase_day']).copy()
                    df_mood['Status'] = df_mood['cigarette_purchase_day'].map({0: 'Non-Smoking Days', 1: 'Smoking Days'})
                    
                    mood_summary = df_mood.groupby('Status')[available_mood].mean().reset_index()
                    mood_summary_melted = mood_summary.melt(id_vars='Status', var_name='Mood', value_name='Average Tracks')
                    mood_summary_melted['Mood'] = mood_summary_melted['Mood'].str.replace('avg_', '').str.replace('_tracks', '').str.title()
                    
                    fig = px.bar(mood_summary_melted, x='Status', y='Average Tracks', color='Mood',
                                title='Playlist Mood Breakdown by Smoking Status',
                                color_discrete_sequence=[COLORS['accent_light'], COLORS['accent_mid'], COLORS['accent_dark']],
                                barmode='stack')
                    fig.update_layout(**theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                    <div class="insight-title">💡 The Ratio Reveals</div>
                    Notice how the proportions shift? On smoking days, the stack changes—more sad tracks, fewer happy ones. 
                    It's not that Jane stops listening to music altogether. She curates differently.
                    <br><br>
                    <strong>This matters for prediction</strong>: We don't need to know *which* songs Jane plays. Just the overall emotional 
                    tone of her soundtrack. That tone is a feature we can measure, track, and use to forecast tomorrow.
                    <br><br>
                    <em>Your playlist is a confession you didn't know you were making.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analysis error: {e}")

def render_meds_panel(df):
    """Health metrics with body-as-witness narrative"""
    if df is None:
        return
    
    st.markdown("""
    <div class="narrative-intro">
    The body doesn't lie. It can't. While Jane's mind justifies, rationalizes, bargains, her body keeps score. 
    The Fitbit on her wrist records what she won't admit: the restless nights, the sedentary days, the elevated heart rate 
    that signals stress before consciousness recognizes it.
    <br><br>
    Wearable health devices are intimate witnesses. They know when you're asleep, when you're moving, when your heart races. 
    They track the physical manifestations of emotional states. Jane might not realize she slept poorly until she checks the app. 
    But the data knew all along.
    <br><br>
    Here's what Jane's body tells us about her smoking patterns.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## ⌚ The Body's Testimony")
    theme = setup_plotly_theme()
    
    with st.spinner("Analyzing physiological patterns..."):
        try:
            health_cols = ['sleep_hours', 'steps', 'avg_heart_rate']
            available = [col for col in health_cols if col in df.columns]
            
            if available and 'cigarette_purchase_day' in df.columns:
                df_clean = df.dropna(subset=available + ['cigarette_purchase_day']).copy()
                df_clean['Status'] = df_clean['cigarette_purchase_day'].map({0: 'Non-Smoking Days', 1: 'Smoking Days'})
                
                # Sleep analysis
                if 'sleep_hours' in available:
                    st.markdown("### 💤 Sleep Deprivation as Predictor")
                    st.markdown("""
                    **The Body's Report**: *Poor sleep weakens willpower. Let's see if Jane's data confirms it.*
                    """)
                    
                    fig = px.box(df_clean, x='Status', y='sleep_hours', 
                                title='Sleep Duration by Smoking Status',
                                color='Status', 
                                color_discrete_map={'Non-Smoking Days': COLORS['accent_mid'], 'Smoking Days': COLORS['accent_dark']})
                    fig.update_layout(**theme, showlegend=False)
                    fig.update_yaxes(title_text='Hours of Sleep')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    no_cig_sleep = df_clean[df_clean['cigarette_purchase_day'] == 0]['sleep_hours'].mean()
                    cig_sleep = df_clean[df_clean['cigarette_purchase_day'] == 1]['sleep_hours'].mean()
                    
                    if cig_sleep < no_cig_sleep:
                        sleep_deficit = no_cig_sleep - cig_sleep
                        st.markdown(f"""
                        <div class="insight-box">
                        <div class="insight-title">🔍 The Sleep Deficit</div>
                        On non-smoking days, Jane averages <strong>{no_cig_sleep:.1f} hours</strong> of sleep. On smoking days? 
                        Only <strong>{cig_sleep:.1f} hours</strong>—a deficit of <strong>{sleep_deficit:.1f} hours</strong>.
                        <br><br>
                        <strong>Why this matters</strong>: Sleep deprivation impairs executive function—the brain's ability to resist temptation. 
                        Jane wakes up tired, her prefrontal cortex foggy, her resolve already weakened before she's had her morning coffee.
                        <br><br>
                        <strong>The cascade</strong>: Poor sleep → reduced self-control → increased stress → craving → relapse.
                        <br><br>
                        <em>Her body predicted the cigarette before she consciously chose it.</em>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Activity analysis
                if 'steps' in available:
                    st.markdown("### 🚶 Movement as Protection")
                    st.markdown("""
                    **The Body's Defense**: *Physical activity reduces stress, improves mood. Does Jane move less before relapse?*
                    """)
                    
                    fig = px.box(df_clean, x='Status', y='steps', 
                                title='Daily Step Count by Smoking Status',
                                color='Status', 
                                color_discrete_map={'Non-Smoking Days': COLORS['accent_mid'], 'Smoking Days': COLORS['accent_dark']})
                    fig.update_layout(**theme, showlegend=False)
                    fig.update_yaxes(title_text='Steps Per Day')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    no_cig_steps = df_clean[df_clean['cigarette_purchase_day'] == 0]['steps'].mean()
                    cig_steps = df_clean[df_clean['cigarette_purchase_day'] == 1]['steps'].mean()
                    
                    if cig_steps < no_cig_steps:
                        step_deficit = no_cig_steps - cig_steps
                        pct_decrease = (step_deficit / no_cig_steps) * 100
                        st.markdown(f"""
                        <div class="insight-box">
                        <div class="insight-title">🔍 The Sedentary Signal</div>
                        Jane walks <strong>{step_deficit:.0f} fewer steps</strong> on smoking days—a <strong>{pct_decrease:.1f}%</strong> decrease 
                        from her typical <strong>{no_cig_steps:.0f} steps</strong>.
                        <br><br>
                        <strong>The interpretation</strong>: Lower activity might indicate fatigue, low mood, or social withdrawal—all risk factors 
                        for relapse. Or it could be practical: she's indoors, at her desk, stressed, and the cigarette is a break she takes 
                        instead of a walk.
                        <br><br>
                        <em>Movement is medicine. Stillness is vulnerability.</em>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Heart rate analysis
                if 'avg_heart_rate' in available:
                    st.markdown("### ❤️ The Heart's Warning")
                    st.markdown("""
                    **The Body's Alarm**: *Elevated resting heart rate signals physiological stress. Let's check Jane's baseline.*
                    """)
                    
                    fig = px.box(df_clean, x='Status', y='avg_heart_rate', 
                                title='Average Heart Rate by Smoking Status',
                                color='Status', 
                                color_discrete_map={'Non-Smoking Days': COLORS['accent_mid'], 'Smoking Days': COLORS['accent_dark']})
                    fig.update_layout(**theme, showlegend=False)
                    fig.update_yaxes(title_text='Beats Per Minute (BPM)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    no_cig_hr = df_clean[df_clean['cigarette_purchase_day'] == 0]['avg_heart_rate'].mean()
                    cig_hr = df_clean[df_clean['cigarette_purchase_day'] == 1]['avg_heart_rate'].mean()
                    
                    hr_diff = abs(cig_hr - no_cig_hr)
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <div class="insight-title">🔍 Cardiovascular Testimony</div>
                    Jane's average heart rate on smoking days: <strong>{cig_hr:.1f} BPM</strong>. On non-smoking days: <strong>{no_cig_hr:.1f} BPM</strong>.
                    {"A difference of <strong>" + f"{hr_diff:.1f} BPM</strong>." if hr_diff > 1 else "Remarkably similar."}
                    <br><br>
                    <strong>What this means</strong>: {
                    "Elevated heart rate suggests physiological stress—anxiety, poor sleep, caffeine overconsumption, or the body's response to craving. The cardiovascular system is responding to something Jane may not consciously recognize yet." 
                    if cig_hr > no_cig_hr + 1
                    else "Interestingly, Jane's heart rate doesn't dramatically shift on smoking days. Stress might be mental/emotional rather than physiological, or her body has adapted to chronic stress levels."
                    }
                    <br><br>
                    <em>The heart rate monitor doesn't judge. It just reports. And sometimes, that report is an early warning.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="narrative-intro" style="margin-top: 40px;">
            <strong>The Body as Archive</strong>
            <br><br>
            Wearable data transforms the body into a readable text. Every physiological metric is a sentence in an ongoing story. 
            Jane's body tells us what her conscious mind might deny: she's struggling. The sleep deficit, the reduced activity, 
            the subtle cardiovascular changes—these aren't coincidences. They're symptoms.
            <br><br>
            And here's the profound part: <em>the body knows before the mind decides</em>. The poor sleep comes first. Then the 
            reduced activity. Then the elevated stress markers. And finally, almost inevitably, the cigarette.
            <br><br>
            If we can read these signals early enough, can we intervene before the relapse? That's the promise of predictive health. 
            It's also the ethical dilemma we'll confront in the next section.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analysis error: {e}")

def render_model_panel(df):
    """Predictive modeling with full ethical weight"""
    if df is None:
        return
    
    st.markdown("""
    <div class="narrative-intro">
    This is it. The moment we've been building toward. Can we predict Jane's tomorrow from her today?
    <br><br>
    We've examined her financial patterns, her emotional tweets, her musical choices, her body's testimony. Each piece of evidence 
    tells part of her story. Now we combine them into a predictive model—an algorithm that claims to know Jane's future behavior 
    before she does.
    <br><br>
    This feels like power. It is power. The question is whether it's power used responsibly.
    <br><br>
    Let's build the model. Then we'll ask the uncomfortable questions.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎯 The Prediction Engine")
    earthy_colors = setup_earthy_plot_style()
    
    with st.spinner("Building predictive models..."):
        try:
            df_model = df.copy()
            
            # Replace avg_valence if adjusted version exists
            if 'avg_valence_adjusted' in df_model.columns:
                df_model['avg_valence'] = df_model['avg_valence_adjusted']
            
            # Remove leakage columns
            leakage_cols = ['visits_cc', 'spend_cc', 'cc_spend_convenience', 'cc_spend_food_out', 
                           'cc_spend_leisure', 'cc_spend_medical', 'cc_spend_nightlife', 
                           'cc_spend_other', 'cc_spend_shopping', 'visits', 'total_spend', 
                           'avg_spend_per_visit', 'cc_spend_7d_sum', 'cc_spend_7d_mean',
                           'cigarette_purchase_day_original', 'smoke_score', 'avg_valence_adjusted']
            
            for col in leakage_cols:
                if col in df_model.columns:
                    df_model = df_model.drop(col, axis=1)
            
            if 'cigarette_purchase_day' not in df_model.columns:
                st.error("Target variable not found")
                return
            
            y = df_model['cigarette_purchase_day'].astype(int)
            X_cols = [col for col in df_model.columns if col not in ['cigarette_purchase_day', 'date']]
            X = df_model[X_cols]
            
            st.markdown("### 📊 Model Foundations")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Behavioral Signals", len(X_cols))
                st.caption("Features extracted from digital life")
            with col2:
                st.metric("Days Observed", len(X))
                st.caption("Training examples")
            with col3:
                smoking_rate = (y.sum() / len(y) * 100)
                st.metric("Smoking Rate", f"{smoking_rate:.1f}%")
                st.caption("Base rate (class imbalance)")
            
            st.markdown("""
            <div class="insight-box">
            <div class="insight-title">🔬 Methodological Choice: Lag-1 Transformation</div>
            <strong>The Problem</strong>: We can't use today's data to predict today's outcome. That's not prediction—it's description.
            <br><br>
            <strong>The Solution</strong>: <em>Lag-1 transformation</em>. We shift all features back one day. Now row <em>i</em> contains 
            day <em>i</em>'s features predicting day <em>i+1</em>'s outcome.
            <br><br>
            <strong>Why this matters</strong>: This is how prediction works in the real world. Jane's Monday Twitter sentiment, Monday sleep, 
            Monday Spotify—all used to predict Tuesday's cigarette purchase. By the time we make the prediction, the behavioral signals 
            are already in the past. No cheating. No leakage.
            <br><br>
            <strong>The cost</strong>: We lose the first day of data (no features to look back to). But we gain temporal integrity.
            </div>
            """, unsafe_allow_html=True)
            
            # Lag-1 transformation
            X_lagged = X.shift(1)
            y_lagged = y[1:]
            X_lagged = X_lagged.iloc[1:]
            
            st.info(f"✅ After lag-1 transformation: **{len(X_lagged)} samples** ready for temporal prediction")
            
            # Split
            split_idx = int(0.8 * len(X_lagged))
            X_train, X_test = X_lagged.iloc[:split_idx], X_lagged.iloc[split_idx:]
            y_train, y_test = y_lagged.iloc[:split_idx], y_lagged.iloc[split_idx:]
            
            st.markdown(f"""
            **📐 Temporal Split**: First 80% ({len(X_train)} days) for training, last 20% ({len(X_test)} days) for testing.
            
            *Why temporal?* Because we can't train on the future to predict the past. The split respects time's arrow.
            """)
            
            # Preprocessing
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            
            preprocessor = ColumnTransformer([
                ('num', SimpleImputer(strategy='median'), numeric_cols),
                ('cat', SimpleImputer(strategy='most_frequent'), categorical_cols)
            ])
            
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Scale
            scaler = StandardScaler()
            if len(numeric_cols) > 0:
                numeric_indices = list(range(len(numeric_cols)))
                X_train_processed[:, numeric_indices] = scaler.fit_transform(X_train_processed[:, numeric_indices])
                X_test_processed[:, numeric_indices] = scaler.transform(X_test_processed[:, numeric_indices])
            
            # Logistic Regression
            st.markdown("---")
            st.markdown("## 📊 Model 1: Logistic Regression")
            st.markdown("""
            <div class="narrative-intro">
            <strong>The Interpretable Baseline</strong>
            <br><br>
            Logistic regression is elegant in its simplicity. Each feature gets a coefficient—a weight indicating how much it 
            increases or decreases smoking probability. Positive coefficients = risk factors. Negative = protective factors.
            <br><br>
            This is the model we can <em>explain</em>. When it predicts Jane will smoke tomorrow, we can point to the specific 
            signals that triggered the alert. That's crucial for any intervention—we need to know <em>why</em>, not just <em>that</em>.
            </div>
            """, unsafe_allow_html=True)
            
            lr_model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
            lr_model.fit(X_train_processed, y_train)
            lr_pred = lr_model.predict(X_test_processed)
            lr_pred_proba = lr_model.predict_proba(X_test_processed)[:, 1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Performance Metrics**")
                lr_report = classification_report(y_test, lr_pred, output_dict=True, zero_division=0)
                st.text(classification_report(y_test, lr_pred, zero_division=0))
            with col2:
                lr_auc = roc_auc_score(y_test, lr_pred_proba)
                st.metric("ROC-AUC Score", f"{lr_auc:.3f}")
                st.caption("0.5 = random guessing, 1.0 = perfect prediction")
                
                cm_lr = confusion_matrix(y_test, lr_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor(COLORS['primary_bg'])
                sns.heatmap(cm_lr, annot=True, fmt='d', ax=ax, cmap='Greys', 
                           annot_kws={'color': COLORS['text_light']},
                           cbar_kws={'label': 'Count'})
                ax.set_title('Confusion Matrix', color=COLORS['text_light'], fontweight='bold', pad=15)
                ax.set_ylabel('Actual', color=COLORS['text_light'])
                ax.set_xlabel('Predicted', color=COLORS['text_light'])
                ax.set_facecolor(COLORS['secondary_bg'])
                ax.tick_params(colors=COLORS['accent_light'])
                st.pyplot(fig)
                plt.close()
            
            if lr_auc > 0.65:
                st.markdown(f"""
                <div class="insight-box">
                <div class="insight-title">🎯 Model Performance Interpretation</div>
                With an ROC-AUC of <strong>{lr_auc:.3f}</strong>, the model demonstrates meaningful predictive power. It's better than random 
                chance (0.50) and approaching clinical utility thresholds (typically 0.70+).
                <br><br>
                <strong>What this means practically</strong>: If we deployed this model, it would correctly flag many of Jane's high-risk days. 
                Not all of them—human behavior is too complex for perfect prediction—but enough to be useful for intervention timing.
                <br><br>
                <strong>The catch</strong>: A model that's 70% accurate makes wrong predictions 30% of the time. Those false positives and false 
                negatives aren't just statistics—they're days when Jane gets flagged incorrectly or missed when she needs support.
                </div>
                """, unsafe_allow_html=True)
            elif lr_auc > 0.55:
                st.markdown(f"""
                <div class="insight-box">
                <div class="insight-title">⚠️ Modest Predictive Power</div>
                With an ROC-AUC of <strong>{lr_auc:.3f}</strong>, the model shows some ability to distinguish smoking from non-smoking days, 
                but the signal is weak. We're only slightly better than random guessing.
                <br><br>
                <strong>Why might this be?</strong>
                <br>• Human behavior is stochastic—there's inherent randomness we can't capture
                <br>• Important variables are missing (life events, social context, external stressors)
                <br>• The lag-1 window might be too short—some patterns develop over weeks, not days
                <br>• Addiction is more complex than any dataset can fully represent
                <br><br>
                <strong>The takeaway</strong>: Behavioral prediction has limits. Even with intimate surveillance of Jane's digital life, 
                we can't fully anticipate her choices. That's actually... reassuring? Free will exists in the noise.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"""
                **⚠️ Limited Predictive Utility**: ROC-AUC of {lr_auc:.3f} suggests the model struggles to reliably predict smoking days. 
                This could indicate that the features we're tracking don't capture the most important drivers of Jane's behavior, or that 
                smoking decisions are more random/contextual than we can model with daily aggregate data.
                """)
            
            # Feature importance
            st.markdown("### 🔍 What Drives the Predictions?")
            st.markdown("""
            <div class="narrative-intro">
            <strong>Opening the Black Box</strong>
            <br><br>
            Here are the features that matter most—the signals the model learned to watch for. Positive coefficients increase smoking 
            probability. Negative coefficients decrease it. The magnitude shows importance.
            <br><br>
            This is what the algorithm "knows" about Jane.
            </div>
            """, unsafe_allow_html=True)
            
            feature_names = numeric_cols + categorical_cols
            if len(lr_model.coef_[0]) == len(feature_names):
                coef_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': lr_model.coef_[0],
                    'abs_coefficient': np.abs(lr_model.coef_[0])
                }).sort_values('abs_coefficient', ascending=False).head(12)
                
                coef_df['impact'] = coef_df['coefficient'].apply(lambda x: 'Increases Risk ⬆️' if x > 0 else 'Decreases Risk ⬇️')
                
                theme = setup_plotly_theme()
                fig = px.bar(coef_df, x='coefficient', y='feature', orientation='h',
                            title='Top 12 Predictive Features (Lag-1 Coefficients)',
                            color='impact',
                            color_discrete_map={'Increases Risk ⬆️': COLORS['accent_mid'], 'Decreases Risk ⬇️': COLORS['accent_dark']})
                fig.update_layout(**theme, height=500)
                fig.update_xaxes(title_text='Coefficient (Impact on Smoking Probability)')
                st.plotly_chart(fig, use_container_width=True)
                
                top_risk = coef_df.nlargest(1, 'coefficient').iloc[0]
                top_protect = coef_df.nsmallest(1, 'coefficient').iloc[0]
                
                st.markdown(f"""
                <div class="insight-box">
                <div class="insight-title">🔍 The Algorithm's Verdict</div>
                <strong>Biggest Risk Factor</strong>: <code>{top_risk['feature']}</code> (coefficient: {top_risk['coefficient']:.3f})
                <br>
                <em>When this signal appears, smoking probability increases.</em>
                <br><br>
                <strong>Strongest Protective Factor</strong>: <code>{top_protect['feature']}</code> (coefficient: {top_protect['coefficient']:.3f})
                <br>
                <em>This signal is associated with successful resistance.</em>
                <br><br>
                These aren't causes—they're correlations. But correlations the model learned to trust.
                </div>
                """, unsafe_allow_html=True)
            
            # Decision Tree
            st.markdown("---")
            st.markdown("## 🌳 Model 2: Decision Tree")
            st.markdown("""
            <div class="narrative-intro">
            <strong>Capturing Complexity</strong>
            <br><br>
            Logistic regression assumes linear relationships—more stress = more smoking. But human behavior isn't linear. Maybe Jane only 
            smokes when <em>both</em> she's sad <em>and</em> she drank alcohol. Or when sleep is poor <em>and</em> music is sad 
            <em>and</em> it's Friday.
            <br><br>
            Decision trees capture these interaction effects. They split data based on yes/no questions, creating rules like 
            "IF avg_valence < 0.4 AND alcohol_purchase = 1 THEN predict smoking."
            <br><br>
            The cost: harder to interpret. The benefit: more accurate for complex patterns.
            </div>
            """, unsafe_allow_html=True)
            
            dt_model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
            dt_model.fit(X_train_processed, y_train)
            dt_pred = dt_model.predict(X_test_processed)
            dt_pred_proba = dt_model.predict_proba(X_test_processed)[:, 1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Performance Metrics**")
                dt_report = classification_report(y_test, dt_pred, output_dict=True, zero_division=0)
                st.text(classification_report(y_test, dt_pred, zero_division=0))
            with col2:
                dt_auc = roc_auc_score(y_test, dt_pred_proba)
                st.metric("ROC-AUC Score", f"{dt_auc:.3f}")
                st.caption("0.5 = random guessing, 1.0 = perfect prediction")
                
                cm_dt = confusion_matrix(y_test, dt_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor(COLORS['primary_bg'])
                sns.heatmap(cm_dt, annot=True, fmt='d', ax=ax, cmap='Greys', 
                           annot_kws={'color': COLORS['text_light']},
                           cbar_kws={'label': 'Count'})
                ax.set_title('Confusion Matrix', color=COLORS['text_light'], fontweight='bold', pad=15)
                ax.set_ylabel('Actual', color=COLORS['text_light'])
                ax.set_xlabel('Predicted', color=COLORS['text_light'])
                ax.set_facecolor(COLORS['secondary_bg'])
                ax.tick_params(colors=COLORS['accent_light'])
                st.pyplot(fig)
                plt.close()
            
            # Comparison
            st.markdown("---")
            st.markdown("## ⚖️ Model Comparison")
            
            comparison_df = pd.DataFrame({
                'Model': ['Logistic Regression', 'Decision Tree'],
                'ROC-AUC': [lr_auc, dt_auc],
                'Precision': [lr_report['1']['precision'], dt_report['1']['precision']],
                'Recall': [lr_report['1']['recall'], dt_report['1']['recall']],
                'F1 Score': [lr_report['1']['f1-score'], dt_report['1']['f1-score']]
            })
            
            st.dataframe(comparison_df.style.format({
                'ROC-AUC': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1 Score': '{:.3f}'
            }).highlight_max(axis=0, subset=['ROC-AUC', 'Precision', 'Recall', 'F1 Score'], 
                            color='rgba(184, 175, 164, 0.3)'), 
            use_container_width=True)
            
            winner = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
            winner_auc = comparison_df['ROC-AUC'].max()
            
            st.markdown(f"""
            <div class="insight-box">
            <div class="insight-title">🏆 The Winner</div>
            <strong>{winner}</strong> achieves the highest ROC-AUC of <strong>{winner_auc:.3f}</strong>.
            <br><br>
            {"<em>The decision tree's ability to capture interaction effects gives it an edge—real behavior involves combinations of factors, not just linear effects.</em>" if "Tree" in winner else "<em>Logistic regression's simplicity is its strength—the linear model is often enough, and interpretability is invaluable for deployment.</em>"}
            </div>
            """, unsafe_allow_html=True)
            
            # Ethics - THE BIG SECTION
            st.markdown("---")
            st.markdown("## ⚠️ Now the Hard Questions")
            st.markdown("""
            <div class="narrative-intro" style="background: rgba(139, 69, 19, 0.3); border-color: rgba(139, 69, 19, 0.8);">
            <strong>What Have We Built?</strong>
            <br><br>
            We've just created a system that predicts Jane's intimate behavior—her relapses, her moments of weakness—from the digital 
            breadcrumbs of her daily life. The model works. That's the easy part.
            <br><br>
            Now: <em>Should it exist?</em>
            <br><br>
            This is where data science becomes philosophy. Where code becomes ethics. Where prediction becomes power, and power demands justification.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔒 The Privacy Problem")
            st.warning("""
            **Surveillance Capitalism Meets Behavioral Health**
            
            To build this model, we tracked:
            - Jane's every purchase (credit card data)
            - Her emotional state (Twitter sentiment)
            - Her cultural consumption (Spotify history)
            - Her physical condition (wearable health data)
            
            This is **total information awareness**. Nothing is private. Nothing is off-limits. Jane's entire existence—financial, emotional, 
            cultural, physical—is legible to the algorithm.
            
            **The Questions:**
            - Where do we draw the line? What's too intimate to quantify?
            - Who has access to this prediction? Jane? Her doctor? Her employer? Her insurance company?
            - Can surveillance ever truly be consensual when opting out means losing access to healthcare/insurance/employment?
            
            **The Uncomfortable Truth:** We wouldn't have built this model without comprehensive surveillance. The accuracy depends on the 
            intimacy of the data. Privacy and prediction are in direct tension.
            """)
            
            st.markdown("### ✋ The Consent Problem")
            st.info("""
            **What Does "Yes" Mean?**
            
            Jane would've clicked "I Agree" to participate in this study. But did she understand what she was agreeing to?
            
            **Standard Consent**: "We'll use your data to study smoking behavior."
            
            **What That Actually Means**:
            - Algorithms will infer your emotional state from your tweets
            - Your music choices will be interpreted as mood indicators
            - Your sleep patterns will be framed as "risk factors"
            - Your tomorrow will be predicted from your today
            - You will be *knowable* in ways you may not know yourself
            
            **The Gap:** There's a difference between consenting to data collection and understanding the implications of algorithmic inference. 
            Jane agreed to be studied. Did she agree to be *known* at this level? To have her behavior forecasted?
            
            **The Core Dilemma:** You can't truly consent to something you don't fully understand. And understanding modern ML systems is beyond 
            most people—including many data scientists.
            """)
            
            st.markdown("### ⚠️ The Harm Problem")
            st.error("""
            **Potential Harms of Behavioral Prediction**
            
            **Insurance Discrimination**
            - Health insurers use behavioral predictions to adjust premiums
            - Life insurance denies coverage based on "high-risk" profiles
            - Jane's smoking prediction becomes grounds for financial penalties
            
            **Employment Impact**
            - Employers screen applicants for "addiction risk"
            - Promotions denied based on behavioral health predictions
            - Jane loses job opportunities because an algorithm flagged her as unreliable
            
            **Targeted Manipulation**
            - Tobacco companies identify moments of vulnerability
            - Ads for cigarettes/alcohol appear exactly when resistance is lowest
            - Jane is exploited by the same data meant to help her
            
            **Social Stigma**
            - Predictions leak (data breaches, subpoenas, "anonymized" datasets)
            - Jane is labeled an "addict" before relapse even occurs
            - The label becomes self-fulfilling—she believes the prediction, gives up trying
            
            **Intervention Backlash**
            - Constant monitoring creates anxiety, resentment
            - Jane feels controlled, not supported
            - The intervention becomes another source of stress, triggering more smoking
            
            **Accuracy Errors**
            - False positives: Jane flagged on days she was fine (boy who cried wolf effect)
            - False negatives: Jane missed on days she needed support (failures that erode trust)
            - Model drift: Behavior changes, model doesn't adapt, predictions become wrong
            
            **The Fundamental Risk:** Once you're known by algorithms, that knowledge can be weaponized against you. Even if collected with 
            good intentions, data has no inherent loyalty. It serves whoever controls it.
            """)
            
            st.markdown("### 💡 So Why Build This?")
            st.success("""
            **The Case for Behavioral Prediction (With Guardrails)**
            
            **Potential Benefits:**
            - **Timely Intervention**: Catch high-risk moments before relapse, offer support when it's needed most
            - **Resource Allocation**: Limited counseling resources directed to those most likely to benefit
            - **Self-Awareness**: Jane sees her own patterns, gains insight into her triggers
            - **Personalization**: Interventions tailored to individual risk profiles (not one-size-fits-all)
            
            **But ONLY if:**
            ✅ **Jane retains full control** over her data and predictions  
            ✅ **Predictions are explanatory**, not just verdicts (show *why*, not just *what*)  
            ✅ **No third-party access** without explicit, informed, revocable consent  
            ✅ **Built-in oversight**: Regular audits for bias, drift, harm  
            ✅ **Right to be forgotten**: Jane can delete her data and model at any time  
            ✅ **Clinical validation**: Models tested in real interventions, not just academic papers  
            ✅ **Transparent limitations**: Users told when models are uncertain or likely wrong  
            ✅ **Human override**: Predictions are suggestions, not mandates. Jane's agency remains intact.
            
            **The Ideal:** Behavioral prediction as a tool of *empowerment*, not control. A mirror that helps Jane see herself more clearly, 
            not a cage that traps her in algorithmic expectations.
            """)
            
            st.markdown("### 🎓 What This Project Actually Is")
            st.markdown("""
            <div class="narrative-intro">
            <strong>A Demonstration, Not a Deployment</strong>
            <br><br>
            Jane is synthetic. Her data is fake. No real person's privacy was violated to build this model. This is an *exercise*—a way 
            to demonstrate technical skills (feature engineering, temporal modeling, class balancing) while grappling with real ethical dilemmas.
            <br><br>
            <strong>If this were real</strong>, deploying it would require:
            <br>• IRB (Institutional Review Board) approval
            <br>• Clinical validation in controlled trials
            <br>• Regulatory compliance (HIPAA, GDPR, etc.)
            <br>• Continuous monitoring for bias and harm
            <br>• Transparent documentation of limitations
            <br>• Meaningful consent processes that actually educate users
            <br><br>
            <strong>The Goal of This Project</strong>: Not to advocate for behavioral surveillance, but to show that I *could* build it, 
            and more importantly, that I *understand the stakes*. Technical competence without ethical awareness is dangerous. This portfolio 
            demonstrates both.
            <br><br>
            <em>Prediction is easy. Responsible prediction is hard. That's why it matters.</em>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Model error: {e}")
            import traceback
            st.text(traceback.format_exc())

def render_methodology_panel(df, quality_report):
    """Methodology documentation"""
    st.markdown("""
    <div class="narrative-intro">
    <strong>The Evidence Log</strong>
    <br><br>
    Every investigation needs documentation. Not just what we found, but *how* we found it. The methodology matters as much as 
    the conclusions—because conclusions without rigorous methods are just guesses.
    <br><br>
    This is the technical appendix. The procedural manual. The reproducible workflow. Less narrative, more protocol. 
    But even here, choices reveal values.
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["🔬 Pipeline", "🤖 Models", "⚠️ Limitations", "📥 Downloads"])
    
    with tabs[0]:
        st.markdown("### Data Integration Pipeline")
        st.markdown("""
        **1. Data Collection Simulation**
        - Multiple CSV sources representing distinct data streams (credit, receipts, social, music, health)
        - Temporal alignment across heterogeneous sources (all keyed by date)
        - Realistic missingness and noise patterns (not artificially clean data)
        """)
        
        st.markdown("**2. Feature Engineering**")
        st.code("""# Lag-1 Transformation for Temporal Integrity
# Today's features → Tomorrow's outcome
X_lagged = X.shift(1)  # Shift features back one day
y_lagged = y[1:]       # Drop first outcome (no prior features)
X_lagged = X_lagged.iloc[1:]  # Drop first feature row (all NaN after shift)

# Now each row i uses day i features to predict day i+1 outcome
# This prevents data leakage - we can only use information available BEFORE the prediction""", language='python')
        
        st.markdown("""
        **3. Feature Selection & Leakage Prevention**
        
        **Excluded** (to prevent leakage):
        - All credit card spending features (`cc_spend_*`, `total_spend`, `avg_spend_per_visit`)
        - Convenience store spending (contains cigarette purchase signal)
        - Any variable that directly or indirectly encodes the outcome
        
        **Included** (behavioral signals):
        - Twitter sentiment scores (emotional state)
        - Spotify valence and track mood distribution (cultural/emotional indicators)
        - Sleep hours, steps, heart rate (physiological stress markers)
        - Alcohol purchase flag (known trigger, but distinct from cigarette purchase)
        """)
        
        st.markdown("**4. Preprocessing Pipeline**")
        st.code("""from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Separate numeric and categorical features
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Build preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), numeric_cols),  # Handle missing numerics
    ('cat', SimpleImputer(strategy='most_frequent'), categorical_cols)  # Handle missing categoricals
])

# Apply
X_processed = preprocessor.fit_transform(X)

# Scale numeric features for model stability
scaler = StandardScaler()
X_processed[:, :len(numeric_cols)] = scaler.fit_transform(X_processed[:, :len(numeric_cols)])""", language='python')
        
        if quality_report and quality_report['missing_data']:
            st.markdown("### Data Quality Report")
            missing_df = pd.DataFrame(list(quality_report['missing_data'].items()), columns=['Feature', 'Missing %'])
            missing_df = missing_df.sort_values('Missing %', ascending=False).head(10)
            
            theme = setup_plotly_theme()
            fig = px.bar(missing_df, x='Feature', y='Missing %', 
                        title='Top 10 Features with Missing Data',
                        color='Missing %', color_continuous_scale='Greys')
            fig.update_layout(**theme)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Total features with missing data: {len(quality_report['missing_data'])}")
    
    with tabs[1]:
        st.markdown("### Model Architecture")
        st.markdown("""
        **Model 1: Logistic Regression**
        - **Purpose**: Interpretable baseline with feature importance
        - **Configuration**: `max_iter=500`, `class_weight='balanced'`, `random_state=42`
        - **Strengths**: 
          - Coefficients are directly interpretable (positive = risk factor, negative = protective factor)
          - Fast to train, robust to overfitting
          - Industry standard for binary classification
        - **Limitations**: 
          - Assumes linear relationships (may miss interaction effects)
          - Limited capacity for complex patterns
        
        **Model 2: Decision Tree Classifier**
        - **Purpose**: Capture non-linear patterns and feature interactions
        - **Configuration**: `max_depth=5`, `class_weight='balanced'`, `random_state=42`
        - **Strengths**: 
          - Captures interaction effects (e.g., "sad music + alcohol = high risk")
          - No scaling required
          - Can model thresholds naturally
        - **Limitations**: 
          - Prone to overfitting without depth limits
          - Less interpretable than linear models
        
        **Why Class Balancing?**
        
        Smoking days are minority class (~20-30% of days). Without `class_weight='balanced'`, models would learn 
        to predict "no smoking" every day and achieve ~70-80% accuracy while being useless for prediction.
        
        Class balancing forces the model to treat both classes as equally important during training, improving 
        recall for smoking days (the class we care about).
        """)
        
        st.markdown("### Evaluation Framework")
        st.markdown("""
        **Temporal Split (80/20)**
        - First 80% of chronological days → training set
        - Last 20% of days → test set
        - **Why temporal?** Can't train on future to predict past. Preserves time's arrow.
        
        **Metrics Explained**
        - **Precision**: Of predicted smoking days, how many were actually smoking days?
          - *High precision* = few false alarms
        - **Recall**: Of actual smoking days, how many did we catch?
          - *High recall* = few missed cases
        - **F1 Score**: Harmonic mean of precision and recall
          - Balances false positives and false negatives
        - **ROC-AUC**: Area under receiver operating characteristic curve
          - Measures overall discrimination ability
          - 0.5 = random, 1.0 = perfect, 0.7+ = clinically useful
        """)
        
        st.code("""# Training and evaluation example
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Train model
model = LogisticRegression(
    max_iter=500,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {auc:.3f}")""", language='python')
    
    with tabs[2]:
        st.markdown("### Limitations & Caveats")
        st.markdown("""
        **Data Limitations**
        
        1. **Synthetic Nature**: Jane is fake. Her data patterns are plausible but not derived from real human behavior. 
        Real data would be messier, more complex, with confounds we can't imagine.
        
        2. **Limited Sample Size**: ~100-200 days is small for behavioral modeling. Real studies need months-to-years 
        across multiple individuals for robust patterns.
        
        3. **Missing Context**: Life doesn't fit in a dataset. We can't capture:
           - Major life events (breakups, job changes, family crises)
           - Environmental factors (weather, holidays, news events)
           - Social dynamics (peer pressure, social support networks)
           - Internal experiences (cravings, motivation fluctuations)
        
        4. **Single Subject**: Jane's patterns won't generalize to others. Every person's triggers are unique.
        
        **Model Limitations**
        
        1. **Temporal Window**: 1-day lag may miss longer patterns. Maybe Jane's smoking is predicted by last *week's* stress, not yesterday's.
        
        2. **Stationarity Assumption**: We assume relationships stay constant. In reality, Jane's triggers may evolve over time.
        
        3. **No Causal Claims**: Correlation ≠ causation. Sad music and smoking co-occur, but sad music doesn't *cause* smoking. 
        Both may be symptoms of underlying emotional state.
        
        4. **Human Unpredictability**: People aren't deterministic. Free will exists. No model can predict with 100% certainty.
        
        5. **Context Collapse**: Aggregate daily features lose temporal structure. Maybe Jane smokes at night after drinking, but our 
        "avg_sentiment" feature averages over the whole day, masking the evening downturn.
        """)
        
        st.markdown("### What This Model CANNOT Do ❌")
        st.markdown("""
        - **Replace clinical assessment** by trained addiction specialists
        - **Account for external shocks** (trauma, major life changes, global events)
        - **Adapt in real-time** without retraining on new data
        - **Guarantee accuracy** for any individual prediction
        - **Make value judgments** about whether intervention is appropriate
        - **Understand subjective experience** (what smoking *means* to Jane)
        """)
        
        st.markdown("### What This Model CAN Do ✅")
        st.markdown("""
        - **Identify statistical patterns** in behavioral data
        - **Flag high-risk periods** for targeted support
        - **Track trends** over time (is Jane improving or struggling?)
        - **Generate hypotheses** for clinical follow-up
        - **Demonstrate technical skills** in ML pipeline development
        - **Provoke ethical discussion** about predictive analytics in behavioral health
        """)
        
        st.warning("""
        **Portfolio Context**
        
        This is a demonstration project, not a deployable product. Real-world behavioral health applications require:
        - Clinical validation studies with real patients
        - Regulatory approval (FDA for medical devices, if applicable)
        - Institutional Review Board (IRB) oversight
        - Ongoing bias auditing and fairness assessment
        - Transparent communication of uncertainty and limitations
        - Meaningful consent processes
        - Clear governance around data access and use
        
        The goal here is to show I can build the technical system *and* understand the ethical implications. 
        Both are necessary. Neither alone is sufficient.
        """)
    
    with tabs[3]:
        st.markdown("### Resources & Downloads")
        
        if df is not None:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Master Dataset (CSV)",
                data=csv_data,
                file_name="jane_behavioral_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Days", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                if 'date' in df.columns:
                    days = (df['date'].max() - df['date'].min()).days
                    st.metric("Time Span", f"{days} days")
        
        st.markdown("### 🎓 Skills Demonstrated")
        st.markdown("""
        **Data Engineering**
        - Multi-source data integration with temporal alignment
        - Feature engineering (lag transformations, aggregations, derived variables)
        - Missing data imputation strategies
        - Leakage detection and prevention
        
        **Machine Learning**
        - Supervised binary classification
        - Class imbalance handling
        - Model comparison (linear vs non-linear)
        - Hyperparameter configuration
        - Temporal cross-validation
        - Evaluation metrics interpretation
        - Feature importance analysis
        
        **Software Engineering**
        - Python (Pandas, NumPy, Scikit-learn, Plotly, Matplotlib, Seaborn)
        - Interactive web application (Streamlit)
        - Modular code organization
        - Error handling and robustness
        - Documentation and reproducibility
        
        **Communication & Storytelling**
        - Translating technical work into narrative
        - Data visualization for insight communication
        - Ethical reasoning and argumentation
        - Audience-aware presentation (technical + non-technical)
        
        **Ethical AI**
        - Privacy and surveillance considerations
        - Consent and agency preservation
        - Harm identification and mitigation
        - Limitation transparency
        - Responsible deployment guidelines
        """)
        
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Framework**: Streamlit (interactive web apps)
        - **Data Processing**: Pandas, NumPy
        - **Machine Learning**: Scikit-learn
        - **Visualization**: Plotly (interactive), Matplotlib & Seaborn (static)
        - **Styling**: Custom CSS for thematic consistency
        """)
        
        st.markdown("""
        <div class="narrative-intro" style="margin-top: 40px;">
        <strong>Thank You for This Investigation</strong>
        <br><br>
        If you've made it this far, you've walked through Jane's entire digital life with me. You've seen her struggles rendered 
        in data, her patterns extracted by algorithms, her tomorrow predicted from her today.
        <br><br>
        More importantly, you've engaged with the questions that haunt me: <em>Should we build systems that know people this intimately? 
        Can surveillance ever be care? Where's the line between insight and intrusion?</em>
        <br><br>
        I don't have perfect answers. But I know the questions matter. And I know that data scientists who ignore these questions 
        are the ones who build the systems we later regret.
        <br><br>
        This portfolio demonstrates technical competence. But technical competence in service of what? That's the question I hope 
        stays with you.
        <br><br>
        — Yasmine
        </div>
        """, unsafe_allow_html=True)

# ===========================
# MAIN
# ===========================
def main():
    """Main application logic"""
    query_params = st.query_params
    item = query_params.get("item", None)
    
    if item is None:
        if not st.session_state.investigation_started:
            render_landing()
        else:
            render_dashboard()
    else:
        render_panel(item)

if __name__ == "__main__":
    main()
