import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from ultralytics import YOLO
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import tempfile
import random
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Infrastructure AI Mapper", layout="wide", page_icon="🛣️")


# --- LOAD AI MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- HELPER FUNCTIONS ---
def extract_gps_from_image(image):
    try:
        exif = image._getexif()
        if exif is not None:
            gps_info = {}
            for key, val in exif.items():
                decode = ExifTags.TAGS.get(key, key)
                if decode == "GPSInfo":
                    for t in val:
                        sub_decoded = ExifTags.GPSTAGS.get(t, t)
                        gps_info[sub_decoded] = val[t]
            lat = gps_info.get('GPSLatitude')
            lat_ref = gps_info.get('GPSLatitudeRef')
            lon = gps_info.get('GPSLongitude')
            lon_ref = gps_info.get('GPSLongitudeRef')
            if lat and lon and lat_ref and lon_ref:
                lat_d = lat[0] + (lat[1]/60.0) + (lat[2]/3600.0)
                lon_d = lon[0] + (lon[1]/60.0) + (lon[2]/3600.0)
                if lat_ref == 'S': lat_d = -lat_d
                if lon_ref == 'W': lon_d = -lon_d
                return lat_d, lon_d
    except Exception:
        pass
    return None, None

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ System Controls")
    st.markdown("Upload infrastructure survey data below.")
    uploaded_file = st.file_uploader("Upload Dashcam Video (.mp4) or Image (.jpg)", type=["jpg", "jpeg", "png", "mp4"])
    
    st.markdown("---")
    conf_threshold = st.slider("AI Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
    
    st.markdown("---")
    st.caption("ITSOLERA Internship Project | Computer Vision Division")

# --- MAIN DASHBOARD AREA ---
st.title("🛣️ Automated Road Damage Mapper")
st.markdown("Geospatial infrastructure monitoring powered by YOLOv8 Computer Vision.")

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    detections = []
    frames_with_damage = []
    
    with st.spinner("AI is analyzing the infrastructure..."):
        
        # --- IMAGE LOGIC ---
        if file_type in ['jpg', 'jpeg', 'png']:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            real_lat, real_lon = extract_gps_from_image(image)
            if real_lat and real_lon:
                base_lat, base_lon = real_lat, real_lon
            else:
                base_lat, base_lon = 26.2483, 68.4096 # Fallback Coordinates

            results = model(img_array, conf=conf_threshold) 
            annotated_img = results[0].plot()
            frames_with_damage.append(annotated_img)
            
            for box in results[0].boxes:
                lat = base_lat + random.uniform(-0.005, 0.005)
                lon = base_lon + random.uniform(-0.005, 0.005)
                detections.append({
                    "Damage Type": model.names[int(box.cls[0])].upper(),
                    "Confidence": float(box.conf[0]),
                    "Latitude": lat, 
                    "Longitude": lon
                })

        # --- VIDEO LOGIC ---
        elif file_type == 'mp4':
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0
            base_lat, base_lon = 26.2483, 68.4096 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_count % fps == 0:
                    results = model(frame, conf=conf_threshold)
                    if len(results[0].boxes) > 0:
                        rgb_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                        frames_with_damage.append(rgb_frame)
                        for box in results[0].boxes:
                            base_lat += 0.0005 
                            base_lon += 0.0005
                            detections.append({
                                "Damage Type": model.names[int(box.cls[0])].upper(),
                                "Confidence": float(box.conf[0]),
                                "Latitude": base_lat, "Longitude": base_lon
                            })
                frame_count += 1
            cap.release()

    # --- RENDER KPI METRICS ---
    st.markdown("### 📊 Survey Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Issues Detected", value=len(detections))
    with col2:
        avg_conf = sum(d['Confidence'] for d in detections) / len(detections) if detections else 0
        st.metric(label="Average AI Confidence", value=f"{avg_conf:.0%}")
    with col3:
        status = "Needs Maintenance" if len(detections) > 0 else "Clear"
        st.metric(label="Overall Status", value=status)

    st.markdown("---")

    # --- RENDER TABS ---
    tab1, tab2, tab3 = st.tabs(["📍 Geospatial Map", "📸 Visual Analysis", "📊 Analytics & Export"])
    
    with tab1:
        st.subheader("Interactive Damage Heatmap & Markers")
        if len(detections) > 0:
            map_center = [detections[0]['Latitude'], detections[0]['Longitude']]
            # Removed the custom CartoDB tileset to ensure standard loading
            m = folium.Map(location=map_center, zoom_start=15)
            
            # Simplified Heatmap data so it doesn't crash the browser renderer
            heat_data = [[d['Latitude'], d['Longitude']] for d in detections]
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
            # Add Individual Markers
            for d in detections:
                folium.Marker(
                    [d['Latitude'], d['Longitude']],
                    popup=f"{d['Damage Type']} ({d['Confidence']:.2f})",
                    icon=folium.Icon(color="red", icon="info-sign") 
                ).add_to(m)
                
            # Changed width to 700 and added returned_objects=[] to prevent blanking
            st_folium(m, width=700, height=500, returned_objects=[])
        else:
            st.success("The area is clear. No markers to display.")
            
    with tab2:
        st.subheader("AI Vision Verification")
        if len(frames_with_damage) > 0:
            st.image(frames_with_damage[0], use_container_width=True)
            if len(frames_with_damage) > 1:
                st.caption(f"Note: {len(frames_with_damage)-1} additional damaged frames were logged.")
        else:
            st.info("No damaged frames to display.")
            
    with tab3:
        st.subheader("Data Distribution & Export")
        if len(detections) > 0:
            df = pd.DataFrame(detections)
            
            col_chart, col_data = st.columns([1.5, 1])
            
            with col_chart:
                damage_counts = df['Damage Type'].value_counts().reset_index()
                damage_counts.columns = ['Damage Type', 'Count']
                fig = px.bar(
                    damage_counts, 
                    x='Damage Type', 
                    y='Count', 
                    color='Damage Type',
                    title="Detected Issues Breakdown",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_data:
                st.dataframe(df, use_container_width=True, height=250)
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_data,
                    file_name="infrastructure_damage_report.csv",
                    mime="text/csv",
                    type="primary"
                )
        else:
            st.info("No data logged. Run an analysis to generate reports.")
else:
    st.info("👈 Please upload a road image or dashcam video from the sidebar to begin the AI analysis.")