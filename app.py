'''Authos: Samantha Lee
App: Agribot Command Center'''
import os
import sys
from pathlib import Path
import json
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- 0. SYSTEM CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

load_dotenv()

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="Agribot Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL STYLING ---
st.markdown("""
<style>
    /* GLOBAL THEME */
    .stApp { 
        background-color: #0e1117; 
        color: #e0e0e0; 
    }
    
    /* HEADERS */
    h1 { 
        color: #4caf50 !important; 
        font-family: 'Helvetica', sans-serif; 
        font-weight: 700; 
    }
    h2, h3, h4 { 
        color: #f0f2f6 !important; 
        font-family: 'Helvetica', sans-serif; 
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 20px; 
    }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        background-color: #1f2937; 
        border-radius: 5px; 
        color: #bdbdbd; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #2e7d32 !important; 
        color: white !important; 
        font-weight: bold;
    }
    
    /* CHAT INPUT - Remove fixed positioning */
    .stChatInput { 
        border-top: 2px solid #2e7d32;
        background-color: #0e1117;
        padding-top: 10px;
    }
    
    /* TABLE TEXT */
    table { 
        color: #e0e0e0 !important; 
    }
    thead tr th { 
        color: #ffffff !important; 
        background-color: #2d3748 !important; 
    }
    tbody tr td { 
        color: #e0e0e0 !important; 
    }
    
    /* STATUS BADGES */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    .badge-healthy { background-color: #2e7d32; color: white; }
    .badge-moderate { background-color: #ffa726; color: black; }
    .badge-stressed { background-color: #d32f2f; color: white; }
    .badge-eval { background-color: #1976d2; color: white; }
    .badge-live { background-color: #7b1fa2; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 3. IMPORT AI AGENT ---
try:
    from ai_agent.src.agents.crop_health_chatbot_agentV6 import CropHealthChatbotAgentV6
except ImportError as e:
    st.error(f"Critical Error: AI Agent Import Failed. {e}")
    st.stop()

# --- 4. DATA LOADING ---
@st.cache_data
def load_map_data():
    """Loads main Parquet data AND Evaluation JSONs."""
    parquet_path = current_dir / "data" / "final_lstm_dataset_cleaned.parquet"
    json_paths = [
        current_dir / "machine_learning/src/feat/Crop_Health_Model_V6/training_results/SET1_growing_season_predictions.json",
        current_dir / "machine_learning/src/feat/Crop_Health_Model_V6/training_results/SET2_non_growing_season_predictions.json"
    ]

    dfs = []
    
    # Load Main
    if parquet_path.exists():
        try:
            df_main = pd.read_parquet(parquet_path, engine='fastparquet')
            df_main['source'] = 'Main Data'
            dfs.append(df_main)
        except Exception as e:
            st.warning(f"Could not load main data: {e}")

    # Load JSONs
    for jp in json_paths:
        if jp.exists():
            try:
                df_json = pd.read_json(jp)
                if not df_json.empty:
                    # Standardize column name
                    if 'actual_ndvi' in df_json.columns:
                        df_json['NDVI_mean'] = df_json['actual_ndvi']
                    elif 'predicted_ndvi' in df_json.columns:
                        df_json['NDVI_mean'] = df_json['predicted_ndvi']
                    df_json['source'] = 'Evaluation Data'
                    dfs.append(df_json)
            except Exception as e:
                st.warning(f"Could not load {jp.name}: {e}")

    if not dfs: 
        return None

    try:
        df_final = pd.concat(dfs, ignore_index=True)
        df_final['date'] = pd.to_datetime(df_final['date'])
        
        # Fill missing columns for map compatibility
        for col in ['era5_gdd_cumsum', 'weather_precip_7d', 'weather_temp_7d_mean', 
                    'smap_soil_moisture', 'heat_stress', 'drought_stress', 'cold_stress']:
            if col not in df_final.columns: 
                df_final[col] = 0

        # Drop rows without GPS for map
        if 'latitude' in df_final.columns and 'longitude' in df_final.columns:
            df_final = df_final.dropna(subset=['latitude', 'longitude'])
        
        return df_final.sort_values(by=['field_id', 'date']).reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Data Merge Failed: {e}")
        return None

df = load_map_data()

# --- 5. AI AGENT INITIALIZATION ---
@st.cache_resource
def initialize_ai_agent(df_data):
    if df_data is not None and not df_data.empty:
        default_date = df_data['date'].max().strftime('%Y-%m-%d')
    else:
        default_date = "2025-10-22"
    
    agent = CropHealthChatbotAgentV6(date=default_date, field_id=None)
    
    welcome_msg = {
        "role": "assistant", 
        "content": f"**Agribot System Online**\n\nInitialized to **{default_date}**. Ready to analyze crop health and generate forecasts.",
        "timestamp": datetime.now().strftime("%H:%M")
    }
    return agent, [welcome_msg]

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    with st.spinner("Initializing Agribot Neural Core..."):
        try:
            agent, init_msgs = initialize_ai_agent(df)
            st.session_state.chatbot = agent
            if not st.session_state.messages:
                st.session_state.messages.extend(init_msgs)
        except Exception as e:
            st.error(f"Agent Initialization Failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# --- 6. MAP FUNCTIONS ---
@st.cache_data
def get_filtered_data(df, selected_region, selected_date):
    """Filter data by region and date."""
    if df is None or df.empty:
        return pd.DataFrame()
        
    if selected_region != "All":
        pattern = rf"\|{selected_region.replace(' ', '-').lower()}\|"
        region_df = df[df['field_id'].str.contains(pattern, na=False, case=False, regex=True)] 
    else:
        region_df = df
        
    dates = region_df['date'].dt.date.unique()
    if len(dates) == 0: 
        return pd.DataFrame()
        
    nearest = min(dates, key=lambda x: abs(x - selected_date))
    return region_df[region_df['date'].dt.date == nearest]

def get_field_history(df, field_id, days=90):
    """Get historical data for a specific field."""
    if df is None or not field_id: 
        return pd.DataFrame()
    return df[df['field_id'] == field_id].sort_values('date').tail(days)

# --- 7. VISUALIZATION ---
def plot_ndvi_trend(history_df, current_date):
    """Plot NDVI trend for a field."""
    if history_df.empty: 
        return
        
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot line
    sns.lineplot(data=history_df, x='date', y='NDVI_mean', ax=ax, 
                 color='#4caf50', linewidth=2.5, label='NDVI')
    
    # Add threshold and current date
    ax.axhline(0.6, color='#ffa726', linestyle='--', alpha=0.7, label='Healthy Threshold')
    ax.axvline(pd.to_datetime(current_date), color='#ef5350', linestyle='-', 
               alpha=0.7, label='Analysis Date')
    
    ax.set_title("90-Day Crop Health Trend", fontsize=12, color='white')
    ax.set_ylabel("NDVI", fontsize=10)
    ax.set_xlabel("")
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# --- 8. TAB 1: SATELLITE MONITOR ---
def render_satellite_monitor(df_data):
    """Renders the satellite monitoring interface."""
    if df_data is None or df_data.empty:
        st.warning("System Offline: Data unavailable.")
        return

    col_controls, col_map = st.columns([1, 3])
    
    # --- CONTROLS ---
    with col_controls:
        st.subheader("Configuration")
        
        # Extract regions
        try:
            regions = sorted(list(set([
                f.split('|')[1].replace('-', ' ').title() 
                for f in df_data['field_id'].unique() 
                if len(f.split('|')) > 1
            ])))
            all_regs = ["All"] + regions
        except:
            all_regs = ["All"]
        
        sel_region = st.selectbox("Region", all_regs)
        
        # Date slider
        min_d = df_data['date'].min().date()
        max_d = df_data['date'].max().date()
        sel_date = st.slider("Analysis Date", min_d, max_d, max_d)
        
        # Filter data
        daily = get_filtered_data(df_data, sel_region, sel_date)
        
        if not daily.empty:
            # Metrics
            st.metric("Active Fields", len(daily))
            avg_ndvi = daily['NDVI_mean'].mean()
            st.metric("Regional NDVI", f"{avg_ndvi:.2f}")
            
            # Health distribution
            st.markdown("#### Health Distribution")
            healthy = len(daily[daily['NDVI_mean'] > 0.6])
            moderate = len(daily[(daily['NDVI_mean'] >= 0.3) & (daily['NDVI_mean'] <= 0.6)])
            stressed = len(daily[daily['NDVI_mean'] < 0.3])
            
            st.markdown(f"Healthy: **{healthy}**")
            st.markdown(f"Moderate: **{moderate}**")
            st.markdown(f"Stressed: **{stressed}**")
            
            st.divider()
            
            # Field selector
            sel_field = st.selectbox(
                "Select Field Detail", 
                ["Overview"] + list(daily['field_id'].unique())
            )
        else:
            st.info("No data found.")
            sel_field = "Overview"

    # --- MAP ---
    with col_map:
        st.subheader("Satellite Field Monitor")
        
        if not daily.empty:
            center = [daily['latitude'].mean(), daily['longitude'].mean()]
            
            # Create map with satellite imagery
            m = folium.Map(
                location=center, 
                zoom_start=11,
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri World Imagery'
            )

            # Add markers
            for _, r in daily.iterrows():
                ndvi = r['NDVI_mean']
                
                # Color based on health
                if ndvi < 0.3:
                    color = "#d32f2f"
                    status = "Stressed"
                elif ndvi < 0.6:
                    color = "#ffa726"
                    status = "Moderate"
                else:
                    color = "#4caf50"
                    status = "Healthy"
                
                # Highlight selected field
                opacity = 1.0 if r['field_id'] == sel_field else 0.7
                radius = 8 if r['field_id'] == sel_field else 6
                
                # Popup
                crop = r['field_id'].split('|')[2].capitalize() if '|' in r['field_id'] else "Unknown"
                popup_html = f"""
                <div style='font-family: Arial; min-width: 180px;'>
                    <h4 style='margin: 0 0 8px 0;'>{crop} Field</h4>
                    <p style='margin: 4px 0;'><b>NDVI:</b> {ndvi:.3f}</p>
                    <p style='margin: 4px 0;'><b>Status:</b> {status}</p>
                    <p style='margin: 4px 0; font-size: 0.85em;'>{r['field_id']}</p>
                </div>
                """
                
                folium.CircleMarker(
                    [r['latitude'], r['longitude']], 
                    radius=radius,
                    color="white",
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=opacity, 
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(m)
            
            st_folium(m, height=500, width="100%")
        else:
            st.info("No map data available for this selection.")

    # --- FIELD DIAGNOSTICS ---
    if sel_field != "Overview":
        st.divider()
        st.subheader(f"Diagnostics: {sel_field}")
        
        hist = get_field_history(df_data, sel_field, days=90)
        
        if not hist.empty:
            # Get observation date data
            obs_date = pd.to_datetime(sel_date)
            plot_data = hist[hist['date'] <= obs_date]
            
            if not plot_data.empty:
                dc1, dc2 = st.columns([2, 1])
                
                with dc1:
                    plot_ndvi_trend(plot_data, sel_date)
                
                with dc2:
                    latest = plot_data.iloc[-1]
                    
                    st.markdown("#### Key Metrics")
                    
                    # Create metrics table
                    metrics_data = {
                        "Metric": [
                            "GDD Cumsum",
                            "7-Day Precip",
                            "7-Day Temp",
                            "Soil Moisture"
                        ],
                        "Value": [
                            f"{latest.get('era5_gdd_cumsum', 0):.1f}",
                            f"{latest.get('weather_precip_7d', 0):.1f} mm",
                            f"{latest.get('weather_temp_7d_mean', 0):.1f} °C",
                            f"{latest.get('smap_soil_moisture', 0):.3f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, width='stretch')
                    
                    st.markdown("#### Stress Status")
                    if latest.get('heat_stress') == 1:
                        st.error("Heat Stress Active")
                    elif latest.get('drought_stress') == 1:
                        st.error("Drought Stress Active")
                    elif latest.get('cold_stress') == 1:
                        st.error("Cold Stress Active")
                    else:
                        st.success("Nominal Conditions")
            else:
                st.warning("No historical data available for selected date.")
        else:
            st.warning("No historical data available for this field.")

# --- 9. TAB 2: AI CONSULTANT ---
def render_ai_consultant():
    """Renders the AI chat interface."""
    st.header("Agribot Consultant")
    
    if "chatbot" not in st.session_state:
        st.error("Chatbot not initialized.")
        return

    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("timestamp", "")
        
        with st.chat_message(role):
            st.markdown(content)
            if ts:
                st.caption(ts)

    # Chat input
    if prompt := st.chat_input("Ask Agribot..."):
        ts_now = datetime.now().strftime("%H:%M")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt, 
            "timestamp": ts_now
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(ts_now)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = st.session_state.chatbot.handle_input(prompt)
                except Exception as e:
                    response = f"**Error Processing Request**\n\n{str(e)}"
                    import traceback
                    st.code(traceback.format_exc())
            
            ts_resp = datetime.now().strftime("%H:%M")
            
            # Display response
            st.markdown(response)
            st.caption(ts_resp)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "timestamp": ts_resp
            })

# --- 10. MAIN APP ---
st.title("Agribot Command Center")
st.markdown("*AI-Powered Crop Health Monitoring & Forecasting*")

if 'chatbot' in st.session_state:
    tab1, tab2 = st.tabs(["Satellite Monitor", "AI Consultant"])
    
    with tab1:
        render_satellite_monitor(df)
    
    with tab2:
        render_ai_consultant()
else:
    st.error("System initialization failed. Please refresh the page.")
    if st.button("Retry"):
        st.rerun()