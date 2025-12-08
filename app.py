# app.py  -- Full Premium Dashboard (UI + Features + AI Improvements)
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
import joblib
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import pandas as pd
import os
import time
from collections import deque
from io import BytesIO

# ---------------- PAGE SETUP ----------------
st.set_page_config(layout="wide", page_title="✨ ASL Premium Dashboard", page_icon="🤟")

# ---------------- PALETTES ----------------
PALETTES = {
    "Neon Purple": {"primary":"#9b4dca","accent":"#6b1fa3","bg1":"#0f0b13","bg2":"#1a1126"},
    "Midnight": {"primary":"#00e5ff","accent":"#00acc1","bg1":"#020617","bg2":"#071229"},
    "Sunset": {"primary":"#ff7043","accent":"#d84315","bg1":"#fff3e0","bg2":"#ffe0b2"},
    "Ocean": {"primary":"#0277bd","accent":"#01579b","bg1":"#e1f5fe","bg2":"#e0f7fa"},
    "Glass Light": {"primary":"#7b1fa2","accent":"#4a148c","bg1":"#d9a7c7","bg2":"#fffcdc"}
}

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Theme mode", ["Dark", "Light"])
palette_name = st.sidebar.selectbox("Palette", list(PALETTES.keys()), index=0)
palette = PALETTES[palette_name]

# Branding & features toggle
st.sidebar.markdown("---")
logo_file = st.sidebar.file_uploader("Upload logo (optional)", type=["png","jpg","jpeg"])
brand_name = st.sidebar.text_input("Brand name", value="ASL Premium")
enable_animation = st.sidebar.checkbox("Enable animations", value=True)
enable_analytics = st.sidebar.checkbox("Enable analytics", value=True)
enable_speech = st.sidebar.checkbox("Browser speak predictions", value=True)
auto_save_history = st.sidebar.checkbox("Auto-save history to CSV", value=True)

# Admin simple password (optional)
st.sidebar.markdown("---")
admin_pw = st.sidebar.text_input("Admin password (optional)", type="password")
admin_enter = st.sidebar.checkbox("Enter admin mode", value=False)

# ---- style variables
bg1 = palette["bg1"]
bg2 = palette["bg2"]
primary = palette["primary"]
accent = palette["accent"]
text_color = "#EEE" if mode == "Dark" else "#111"

# ---------------- CSS / UI ----------------
anim_css = ""
if enable_animation:
    anim_css = """
    @keyframes floaty {0%{transform:translateY(0)}50%{transform:translateY(-8px)}100%{transform:translateY(0)}}
    .floaty { animation: floaty 3.5s ease-in-out infinite; }
    """

# premium hybrid background (CSS + JS)
st.markdown(f"""
<style>
:root{{--primary:{primary}; --accent:{accent}; --text:{text_color};}}

/* ---------- PREMIUM HYBRID BACKGROUND ---------- */

/* Base animated gradient waves (uses palette bg1/bg2) */
html, body {{
  background: linear-gradient(120deg, {bg1}, {bg2});
  background-size: 400% 400%;
  animation: gradientMove 12s ease infinite;
  overflow-x: hidden;
}}

@keyframes gradientMove {{
  0%   {{ background-position: 0% 50%; }}
  50%  {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}

/* Neon cyberpunk grid overlay */
body::before {{
  content: "";
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
  background-size: 45px 45px;
  pointer-events: none;
  mix-blend-mode: overlay;
  animation: neonShift 8s linear infinite;
  z-index: -1;
}}

@keyframes neonShift {{
  0%   {{ filter: hue-rotate(0deg); }}
  100% {{ filter: hue-rotate(40deg); }}
}}

/* Soft glow aura blobs */
body::after {{
  content: "";
  position: fixed;
  width: 600px;
  height: 600px;
  top: 10%;
  left: -10%;
  background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 70%);
  filter: blur(80px);
  z-index: -1;
}}
.glow2 {{
  position: fixed;
  width: 700px;
  height: 700px;
  bottom: -10%;
  right: -10%;
  background: radial-gradient(circle, rgba(155,50,255,0.18), transparent 70%);
  filter: blur(100px);
  z-index: -1;
}}

/* Floating glass particles */
.particle {{
  position: fixed;
  width: 130px;
  height: 130px;
  background: rgba(255, 255, 255, 0.04);
  backdrop-filter: blur(10px);
  border-radius: 50%;
  animation: floaty 6.5s ease-in-out infinite;
  z-index: -1;
  box-shadow: 0 6px 30px rgba(0,0,0,0.35);
}}

@keyframes floaty {{
  0%   {{ transform: translateY(0px); opacity: 0.5; }}
  50%  {{ transform: translateY(-25px); opacity: 1; }}
  100% {{ transform: translateY(0px); opacity: 0.5; }}
}}

/* keep existing UI polish (buttons/cards) */
.header{{display:flex;gap:12px;align-items:center;padding:12px;border-radius:12px;margin-bottom:8px;}}
.brand-title{{font-size:34px;font-weight:900;color:var(--primary);margin:0;}}
.brand-sub{{font-size:13px;color:var(--accent);margin:0}}
.card{{background: rgba(255,255,255,0.04); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.04);}}
.small{{background: rgba(255,255,255,0.03); padding:10px; border-radius:8px;}}
.stButton > button{{background:var(--primary); color:white; border-radius:10px; padding:8px 16px; font-weight:700}}
svg {{filter: drop-shadow(0 6px 20px rgba(0,0,0,0.45));}}
{anim_css}
</style>
""", unsafe_allow_html=True)

# add the floating particles and extra glow blob via JS (inject after CSS)
st.markdown("""
<script>
(function(){
  // create a few floating particles
  for (let i = 0; i < 8; i++) {
    let p = document.createElement("div");
    p.className = "particle";
    p.style.top = (Math.random() * 90) + "vh";
    p.style.left = (Math.random() * 90) + "vw";
    p.style.animationDelay = (Math.random() * 5) + "s";
    p.style.opacity = (0.35 + Math.random()*0.6);
    document.body.appendChild(p);
  }
  // glow blob
  let g2 = document.createElement("div");
  g2.className = "glow2";
  document.body.appendChild(g2);
})();
</script>
""", unsafe_allow_html=True)


# Header
col_l, col_h, col_r = st.columns([1,6,1])
with col_l:
    if logo_file:
        logo_img = Image.open(logo_file).convert("RGBA").resize((72,72))
        st.image(logo_img)
    else:
        # placeholder circle
        st.markdown("<div style='width:72px;height:72px;border-radius:36px;background:linear-gradient(135deg,#333,#111);'></div>", unsafe_allow_html=True)
with col_h:
    st.markdown(f"<div class='header {'floaty' if enable_animation else ''}'>"
                f"<div><h1 class='brand-title'>🤟 {brand_name}</h1>"
                f"<p class='brand-sub'>Premium ASL Recognition • Live & Photo • Stabilized predictions</p></div></div>", unsafe_allow_html=True)
with col_r:
    pass

# ---------------- Model Loading (robust) ----------------
@st.cache_resource
def load_model_and_le():
    BASE = os.path.dirname(os.path.abspath(__file__))
    cand_models = [os.path.join(BASE,"models","model.h5"), os.path.join(BASE,"model.h5")]
    cand_le = [os.path.join(BASE,"models","label_encoder.joblib"),
               os.path.join(BASE,"label_encoder.joblib"),
               os.path.join(BASE,"label_encoder.pkl")]
    model_path = None
    le_path = None
    for c in cand_models:
        if os.path.exists(c):
            model_path = c
            break
    for c in cand_le:
        if os.path.exists(c):
            le_path = c
            break
    if model_path is None:
        raise FileNotFoundError("model.h5 not found. Put model.h5 in root or models/")
    model = tf.keras.models.load_model(model_path)
    le = None
    if le_path:
        try:
            le = joblib.load(le_path)
        except Exception:
            le = None
    return model, le, model_path, le_path

try:
    if "model" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.model, st.session_state.le, mp, lp = load_model_and_le()
            st.success("Model loaded ✅")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

model = st.session_state.model
le = st.session_state.le



# ---------------- Utilities ----------------
#HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_history.csv")

#def preprocess_image(pil_img):
    #img = ImageOps.grayscale(pil_img).resize((28,28))
    #arr = np.array(img).astype("float32")/255.0
    #arr = arr.reshape(1,28,28,1)
   # return arr
   # ---------------- Utilities ----------------
HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_history.csv")
LABEL_MAP = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N",
    14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V",
    22: "W", 23: "X", 24: "Y"
}
def preprocess_image(pil_img):
    # MNIST-style preprocessing
    img = pil_img.convert("L")          # grayscale
    img = img.resize((28, 28))          # 28x28
    arr = np.array(img).astype("float32") / 255.0
    # OPTIONAL: invert colors – test both ON/OFF
    # arr = 1.0 - arr
    arr = arr.reshape(1, 28, 28, 1)
    return arr



# smoothing (rolling deque)
def stabilize_predictions(pred_queue, new_pred, window=5):
    pred_queue.append(new_pred)
    arr = np.array(pred_queue)
    # average probs
    mean = np.mean(arr, axis=0)
    return mean

def save_history_to_disk(history_df):
    try:
        history_df.to_csv(HISTORY_CSV, index=False)
        return True
    except Exception as e:
        return False

# load persisted history at start
if "history" not in st.session_state:
    if os.path.exists(HISTORY_CSV):
        try:
            st.session_state.history = pd.read_csv(HISTORY_CSV).to_dict('records')
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []

# ---------------- Layout: Recognition + Analytics Tabs ----------------
tabs = st.tabs(["Recognition", "Analytics" if enable_analytics else "Analytics (Disabled)", "Settings"])
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
      st.subheader("📸 Photo / Quick Snapshot")

    mode_choice = st.radio(
        "Input mode",
        ["Take Photo", "Upload Image"],
        horizontal=True
    )

    if mode_choice == "Take Photo":
        image_source = st.camera_input("Tap to capture")
    else:
        image_source = st.file_uploader(
            "Choose an ASL hand image",
            type=["png", "jpg", "jpeg"]
        )

    if image_source is not None:
        pil = Image.open(image_source).convert("RGB")
        st.image(pil, width=320)

        # 3) Preprocess + predict
        arr = preprocess_image(pil)
        pred = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx])

        # 4) Label mapping (using LABEL_MAP)
        try:
            if le is not None:
                label_idx = int(le.inverse_transform([idx])[0])
            else:
                label_idx = int(idx)

            letter = LABEL_MAP.get(label_idx, "?")
        except Exception:
            letter = "?"

        st.success(f"Predicted: **{letter}** ({conf:.1%})")

        # 5) History log
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append(
            {"time": ts, "letter": letter, "conf": conf}
        )
        if auto_save_history:
            try:
                dfh = pd.DataFrame(st.session_state.history)
                save_history_to_disk(dfh)
            except Exception:
                pass


            # speak using browser TTS (via JS)
            #if enable_speech:
               # st.markdown(f"""
                #<script>
              #  var msg = new SpeechSynthesisUtterance("Prediction {letter} with confidence {Math.round({conf}*100)} percent");
               # window.speechSynthesis.speak(msg);
               # </script>
              #  """, unsafe_allow_html=True)

    with col2:
        st.subheader("🎥 Live Webcam (Center your hand inside the green box)")
        # prepare smoothing deque
        if "pred_queue" not in st.session_state:
            st.session_state.pred_queue = deque(maxlen=6)
        if "last_fps_time" not in st.session_state:
            st.session_state.last_fps_time = time.time()
            st.session_state.frames = 0
        class PremiumProc(VideoTransformerBase):
            def __init__(self):
                self.model = model
                self.le = le
                self.local_queue = deque(maxlen=6)
            def recv_queued(self, frame):
                # FPS calculation
                st.session_state.frames += 1
                now = time.time()
                elapsed = now - st.session_state.last_fps_time
                if elapsed >= 1.0:
                    st.session_state.fps = st.session_state.frames/elapsed
                    st.session_state.frames = 0
                    st.session_state.last_fps_time = now

                img = frame.to_ndarray(format="bgr24")
                h,w = img.shape[:2]
                # central crop for hand
                cx1,cy1 = w//4, h//4
                cx2,cy2 = 3*w//4, 3*h//4
                crop = img[cy1:cy2, cx1:cx2].copy()

                # Simple preprocessing: convert to gray, threshold, find largest contour (rough hand bbox)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox_info = None
                if contours:
                    # choose largest contour area
                    c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    if area > 2000:  # tiny filter
                        x,y,wc,hc = cv2.boundingRect(c)
                        bbox_info = (x+cx1, y+cy1, x+wc+cx1, y+hc+cy1)
                        # draw bbox on full image
                        cv2.rectangle(img, (bbox_info[0], bbox_info[1]), (bbox_info[2], bbox_info[3]), (0,255,100), 2)

                # If we have bbox crop, use that region to predict, else use central crop resized
                if bbox_info:
                    bx1,by1,bx2,by2 = bbox_info
                    sub = img[by1:by2, bx1:bx2]
                    if sub.size == 0:
                        sub = crop
                else:
                    sub = crop

                # prepare model input
                try:
                    sub_pil = Image.fromarray(sub).convert("RGB")
                    arr = ImageOps.grayscale(sub_pil).resize((28,28))
                    arr = np.array(arr).astype("float32")/255.0
                    arr = arr.reshape(1,28,28,1)
                    pred = self.model.predict(arr, verbose=0)[0]
                except Exception as e:
                    pred = np.zeros((26,))
                idx = int(np.argmax(pred))
                conf = float(pred[idx])
                # smoothing across frames
                # push to session deque
                if "preds_hist" not in st.session_state:
                    st.session_state.preds_hist = deque(maxlen=6)
                st.session_state.preds_hist.append(pred)
                mean_pred = np.mean(np.array(st.session_state.preds_hist), axis=0)
                midx = int(np.argmax(mean_pred))
                try:
                    if self.le is not None:
                      label_idx = int(self.le.inverse_transform([midx])[0])
                    else:
                      label_idx = int(midx)

                      letter = LABEL_MAP.get(label_idx, "?")
                except Exception:
                    letter = "?"


                # overlay letter + confidence + fps
                cv2.putText(img, f"{letter} {conf:.2f}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,120), 3)
                fps_display = f"FPS: {st.session_state.get('fps',0):.1f}"
                cv2.putText(img, fps_display, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

                # occasionally log prediction (every ~1.2s)
                if "last_log_time" not in st.session_state:
                    st.session_state.last_log_time = 0
                if time.time() - st.session_state.last_log_time > 1.2:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append({"time":ts, "letter":letter, "conf":conf})
                    st.session_state.last_log_time = time.time()
                    # persist
                    if auto_save_history:
                        try:
                            dfh = pd.DataFrame(st.session_state.history)
                            save_history_to_disk(dfh)
                        except:
                            pass
                    # browser speak (inject tiny script) via st.markdown (non-blocking)
                    if enable_speech:
                        safe_letter = letter.replace('"','')
                        js = f"<script>window.speechSynthesis.speak(new SpeechSynthesisUtterance('Prediction {safe_letter}'));</script>"
                        st.components.v1.html(js, height=0)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="premium_live",
            video_transformer_factory=PremiumProc,
            rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------- Analytics Tab ----------------
with tabs[1]:
    if not enable_analytics:
        st.info("Enable analytics from Settings to view this page.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Analytics & History")
        hist = st.session_state.get("history", [])
        if len(hist) == 0:
            st.info("No predictions yet. Use camera or upload to create data.")
        else:
            df = pd.DataFrame(hist)
            df['time'] = pd.to_datetime(df['time'])
            st.metric("Total predictions", len(df))
            # layout
            a,b = st.columns([1,1.4])
            with a:
                st.markdown("**Recent (latest 12)**")
                st.dataframe(df.sort_values('time', ascending=False).head(12), height=260)
                st.markdown("---")
                st.markdown("Download")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv, file_name="asl_history.csv", mime="text/csv")
                # excel
                try:
                    towrite = BytesIO()
                    df.to_excel(towrite, index=False, engine='openpyxl')
                    towrite.seek(0)
                    st.download_button("Download Excel", data=towrite, file_name="asl_history.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception:
                    st.caption("Excel export requires openpyxl (optional).")

            with b:
                counts = df['letter'].value_counts().sort_index()
                counts_df = counts.rename_axis('letter').reset_index(name='count')
                st.markdown("**Counts by Letter**")
                st.bar_chart(counts_df.set_index('letter'))
                st.markdown("**Confidence over time (last 100)**")
                st.line_chart(df['conf'].tail(100))

        st.markdown("</div>", unsafe_allow_html=True)

# -------------- Settings Tab ----------------
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("⚙️ More Controls & Shortcuts")
    st.markdown("""
    - **Keyboard shortcut:** Press **Space** to take a photo when camera input is focused.  
    - **Auto-save**: History is saved to `asl_history.csv` automatically when toggled.  
    - **Admin mode:** Toggle to reveal admin-only options (clearing history).  
    """)

    # inject JS for spacebar -> camera snapshot trigger
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
      if (e.code === 'Space') {
        // find camera capture button (Streamlit's camera has input type=file; we try to click it)
        let cams = document.querySelectorAll('input[type=file]');
        if (cams.length>0) {
          cams[0].click();
        }
      }
    });
    </script>
    """, unsafe_allow_html=True)

    if admin_enter:
        if admin_pw:
            # simple check: password must equal 'admin' OR the one user set (this is demo)
            # for better security, implement proper auth
            if admin_pw.strip() == "admin" or len(admin_pw) > 0:
                st.warning("Admin mode enabled (basic).")
                if st.button("Clear history (local only)"):
                    st.session_state.history = []
                    if os.path.exists(HISTORY_CSV):
                        try:
                            os.remove(HISTORY_CSV)
                        except:
                            pass
                    st.success("History cleared.")
            else:
                st.error("Wrong admin password.")
        else:
            st.info("Set Admin password in sidebar to enable admin actions.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("-----------------------------------------------------------")
st.caption("✨ Developed by Subasri Chinnadurai • Premium ASL Dashboard")

