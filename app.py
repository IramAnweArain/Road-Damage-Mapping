import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
import tempfile
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="Road Damage Mapper Pro", layout="wide", page_icon="🛣️")
st.title("🛣️ Automated Road Damage Mapper (Pro Version)")
st.markdown("Upload a road image or dashcam video to detect damage and extract geospatial data.")

# --- LOAD AI MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- HELPER: GPS EXTRACTION ---
def get_decimal_coordinates(info):
    # Converts EXIF Degree/Minute/Second format into Decimal format
    for key in ['Latitude', 'Longitude']:
        if 'GPS'+key in info and 'GPS'+key+'Ref' in info:
            e = info['GPS'+key]
            ref = info['GPS'+key+'Ref']
            d = e[0] + (e[1] / 60.0) + (e[2] / 3600.0)
            if ref in ['S', 'W']:
                d = -d
            return d
    return None

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

# --- UPLOADER (Image or Video) ---
uploaded_file = st.file_uploader("Upload Dashcam Video (.mp4) or Image (.jpg)", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    detections = []
    frames_with_damage = []
    
    with st.spinner("AI is analyzing the infrastructure..."):
        
        # ==========================================
        # 🟢 IMAGE PROCESSING LOGIC
        # ==========================================
        if file_type in ['jpg', 'jpeg', 'png']:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Extract real GPS, or fallback to simulation
            real_lat, real_lon = extract_gps_from_image(image)
            if real_lat and real_lon:
                base_lat, base_lon = real_lat, real_lon
                st.success("✅ Real GPS metadata extracted from image.")
            else:
                base_lat, base_lon = 26.2483, 68.4096 # Nawabshah Fallback
                st.warning("⚠️ No GPS metadata found. Simulating location.")

            results = model(img_array)
            annotated_img = results[0].plot()
            frames_with_damage.append(annotated_img)
            
            for box in results[0].boxes:
                lat = base_lat + random.uniform(-0.005, 0.005)
                lon = base_lon + random.uniform(-0.005, 0.005)
                detections.append({
                    "type": model.names[int(box.cls[0])],
                    "conf": float(box.conf[0]),
                    "lat": lat, "lon": lon
                })

        # ==========================================
        # 🔵 VIDEO PROCESSING LOGIC
        # ==========================================
        elif file_type == 'mp4':
            st.info("Processing video frame by frame. This may take a moment...")
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0
            
            # Simulated base for video route
            base_lat, base_lon = 26.2483, 68.4096 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Analyze 1 frame every second to save time
                if frame_count % fps == 0:
                    results = model(frame)
                    
                    if len(results[0].boxes) > 0:
                        # Convert BGR to RGB for Streamlit
                        rgb_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                        frames_with_damage.append(rgb_frame)
                        
                        for box in results[0].boxes:
                            # Move the GPS coordinate slightly to simulate a moving car
                            base_lat += 0.0005 
                            base_lon += 0.0005
                            detections.append({
                                "type": model.names[int(box.cls[0])],
                                "conf": float(box.conf[0]),
                                "lat": base_lat, "lon": base_lon
                            })
                frame_count += 1
            cap.release()

    # --- DASHBOARD DISPLAY ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📷 Damage Detected ({len(detections)} issues)")
        if len(frames_with_damage) > 0:
            # Show the first frame where damage was found
            st.image(frames_with_damage[0], use_container_width=True)
            if len(frames_with_damage) > 1:
                st.caption(f"+ {len(frames_with_damage)-1} more frames with damage (Data logged to map)")
        else:
            st.success("No road damage detected!")
            
    with col2:
        st.subheader("📍 Geospatial Damage Map")
        if len(detections) > 0:
            map_center = [detections[0]['lat'], detections[0]['lon']]
            m = folium.Map(location=map_center, zoom_start=15)
            
            for d in detections:
                folium.Marker(
                    [d['lat'], d['lon']],
                    popup=f"{d['type'].upper()} (Conf: {d['conf']:.2f})",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(m)
            st_folium(m, width=700, height=500)