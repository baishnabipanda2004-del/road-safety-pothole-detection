import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import os
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Pothole Detector AI", page_icon="🛣️", layout="wide")
st.title("🛣️ Pothole Detection AI & Dashboard")
st.markdown("Upload a picture or video of a road to identify potholes, assess severity, and track the total count.")

# ==========================================
# 2. SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("⚙️ Model Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# ==========================================
# 3. LOAD THE MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO('best.onnx', task='detect') 

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Could not load the model. Make sure 'best.onnx' is in your directory.")
    st.stop()

# ==========================================
# 4. ANALYSIS FUNCTIONS
# ==========================================
def analyze_potholes(boxes):
    """Calculates severity based on normalized bounding box area."""
    counts = {"Small": 0, "Injurious": 0, "Hazardous": 0}
    
    for box in boxes.xyxyn:
        x1, y1, x2, y2 = box[:4].tolist()
        area = (x2 - x1) * (y2 - y1) 
        
        if area < 0.01:
            counts["Small"] += 1
        elif area < 0.04:
            counts["Injurious"] += 1
        else:
            counts["Hazardous"] += 1
            
    return counts

def get_road_condition(counts):
    if counts["Hazardous"] > 0:
        return "🚨 CRITICAL"
    elif counts["Injurious"] > 1 or counts["Small"] > 3:
        return "⚠️ POOR"
    elif counts["Injurious"] > 0 or counts["Small"] > 0:
        return "🟡 FAIR"
    else:
        return "✅ GOOD"

# ==========================================
# 5. MAIN UI & UPLOAD LOGIC
# ==========================================
uploaded_file = st.file_uploader(
    "Upload a Road Image or Video (JPG/PNG/MP4/AVI)", 
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # ----------------------------------------
    # IMAGE PROCESSING
    # ----------------------------------------
    if file_extension in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        with st.spinner("Scanning road surface..."):
            results = model.predict(image, conf=confidence)
            result_img_bgr = results[0].plot() 
            result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
            
            severity_counts = analyze_potholes(results[0].boxes)
            total_potholes = sum(severity_counts.values())
            road_condition = get_road_condition(severity_counts)
            
        with col2:
            st.subheader("Detection Results")
            st.image(result_img_rgb, use_container_width=True)
            
        st.markdown("---")
        st.subheader(f"📊 Dashboard | Road Condition: {road_condition}")
        
        # 4-column layout to match the video dashboard structure
        m0, m1, m2, m3, m4 = st.columns(5)
        m0.metric("Grand Total", total_potholes) # For an image, Grand Total is just the image total
        m1.metric("Current Frame", total_potholes)
        m2.metric("🟢 Small", severity_counts["Small"])
        m3.metric("🟡 Injurious", severity_counts["Injurious"])
        m4.metric("🔴 Hazardous", severity_counts["Hazardous"])

    # ----------------------------------------
    # VIDEO PROCESSING
    # ----------------------------------------
    elif file_extension in ["mp4", "avi", "mov"]:
        st.markdown("---")
        st.subheader("📹 Live Video Detection")
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        condition_banner = st.empty()
        
        # 5 Columns for Dashboard Placeholders
        dash_cols = st.columns(5)
        m0 = dash_cols[0].empty() # Grand Total
        m1 = dash_cols[1].empty() # Current Frame Total
        m2 = dash_cols[2].empty() # Small
        m3 = dash_cols[3].empty() # Injurious
        m4 = dash_cols[4].empty() # Hazardous
        
        chart_placeholder = st.empty()
        stop_button = st.button("Stop Video Processing")

        history = []
        unique_potholes = set() # This will store the unique IDs

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break
            
            # Use .track() instead of .predict() to get unique IDs across frames
            results = model.track(frame, conf=confidence, persist=True, verbose=False)
            result_frame_bgr = results[0].plot()
            result_frame_rgb = cv2.cvtColor(result_frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Check if any objects were tracked and have IDs assigned
            if results[0].boxes.id is not None:
                # Extract the IDs and add them to our unique set
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                unique_potholes.update(ids)
                
            # Calculate Grand Total
            grand_total = len(unique_potholes)
            
            # Frame-specific severity & condition
            current_severity = analyze_potholes(results[0].boxes)
            current_total = sum(current_severity.values())
            current_condition = get_road_condition(current_severity)
            
            history.append({
                "Grand Total": grand_total,
                "Current Frame": current_total
            })
            
            # Render video frame
            stframe.image(result_frame_rgb, use_container_width=True)
            
            # Update Dashboards
            condition_banner.markdown(f"### 📊 Live Dashboard | Current Road Condition: **{current_condition}**")
            
            m0.metric("🏆 Grand Total", grand_total)
            m1.metric("Current Frame", current_total)
            m2.metric("🟢 Small", current_severity["Small"])
            m3.metric("🟡 Injurious", current_severity["Injurious"])
            m4.metric("🔴 Hazardous", current_severity["Hazardous"])
            
            # Update live chart 
            df = pd.DataFrame(history[-50:])
            chart_placeholder.line_chart(df)
                
        cap.release()
        os.unlink(tfile.name) 
        
        if not stop_button:
            st.success(f"Video processing complete. Final Grand Total: {len(unique_potholes)} potholes detected.")