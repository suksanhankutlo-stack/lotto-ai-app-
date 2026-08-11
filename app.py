# ============================================================
# 🤖 LOTTO AI PRO V7.1 ADAPTIVE (Optimized)
# AI-ONLY • FAST MOBILE • AUTO BACKTEST
# ============================================================

import re
import hashlib
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Lotto AI PRO V7.1 Adaptive",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2. LOTTERY SOURCES
# ============================================================
LOTTERY_SOURCES = {
    "หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "หวยธกส": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}

# ============================================================
# 3. CONSTANTS
# ============================================================
POSITIONS = ["H", "T", "O", "T2", "O2"]
POSITION_LABELS = {
    "H": "💯 หลักร้อย (3 บน)",
    "T": "🔟 หลักสิบ (3 บน)",
    "O": "1️⃣ หลักหน่วย (3 บน)",
    "T2": "🔽 หลักสิบ (2 ล่าง)",
    "O2": "⬇️ หลักหน่วย (2 ล่าง)"
}
DOW_NAMES = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

# ============================================================
# 4. CSS (Optimized for readability)
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: #f8fafc; }
        .main-title {
            text-align: center; font-size: 2.5rem; font-weight: 900; margin-bottom: 5px;
            background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #64748b; font-size: 1rem; margin-bottom: 25px; font-weight: 600;}
        .status-card {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 15px; text-align: center; color: #1e293b; line-height: 1.6;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .number-card {
            border-radius: 12px; padding: 15px; margin: 10px 0; text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .hot-card { background: #f0fdf4; border-top: 4px solid #16a34a; }
        .dead-card { background: #fef2f2; border-top: 4px solid #dc2626; }
        .pos-title { font-size: 1.1rem; font-weight: 700; color: #475569; margin-bottom: 5px; }
        .num-highlight-hot { font-size: 2.2rem; font-weight: 900; color: #16a34a; letter-spacing: 2px; }
        .num-highlight-dead { font-size: 1.8rem; font-weight: 800; color: #dc2626; letter-spacing: 2px; }
        .prob-text { color: #64748b; font-size: 0.8rem; margin-top: 8px; }
        .model-badge {
            display: inline-block; background: #f1f5f9; border-radius: 20px;
            padding: 4px 12px; font-size: 0.8rem; font-weight: 600; color: #475569; margin-top: 10px;
        }
        div.stButton > button {
            border-radius: 8px; font-size: 16px; font-weight: bold; min-height: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 5. DATE PARSER (Kept fast and robust)
# ============================================================
THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
}

def normalize_date(value):
    if not value: return None
    text = str(value).strip()
    
    for month_name, month_num in THAI_MONTHS.items():
        match = re.search(rf"(\d{{1,2}})\s*{re.escape(month_name)}\s*(\d{{4}})", text)
        if match:
            y = int(match.group(2))
            if y >= 2400: y -= 543
            try: return pd.Timestamp(y, month_num, int(match.group(1)))
            except: return None

    match = re.search(r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})", text)
    if match:
        a, b, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            if a >= 1000: y, m, d = a, b, c
            else:
                d, m, y = a, b, c
                if y < 100: y += 2000
                if y >= 2400: y -= 543
            return pd.Timestamp(y, m, d)
        except: return None
    return None

# ============================================================
# 6. FETCH LOTTERY DATA
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_lottery_data(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content")) or soup
        
        extracted, current_date = [], None
        for line in content.get_text(separator="\n").split("\n"):
            line = line.strip()
            if not line: continue
            
            parsed_date = normalize_date(line)
            if parsed_date: current_date = parsed_date
                
            match = re.search(r"\b(\d{3})\b.*?\b(\d{2})\b", line)
            if match and current_date:
                extracted.append({"Date": current_date, "Result_3D": match.group(1), "Result_2D": match.group(2)})

        df = pd.DataFrame(extracted)
        if df.empty: return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Result_3D", "Result_2D"]).drop_duplicates(subset=["Date", "Result_3D", "Result_2D"]).sort_values("Date").reset_index(drop=True)
        return df
    except: return pd.DataFrame()

# ============================================================
# 7. BUILD FEATURES (Optimized & Upgraded for Accuracy)
# ============================================================
def build_features(df):
    work = df.copy()
    
    # Fast string splitting
    work["H"] = work["Result_3D"].str[0].astype(np.int8)
    work["T"] = work["Result_3D"].str[1].astype(np.int8)
    work["O"] = work["Result_3D"].str[2].astype(np.int8)
    work["T2"] = work["Result_2D"].str[0].astype(np.int8)
    work["O2"] = work["Result_2D"].str[1].astype(np.int8)

    # Calendar
    work["DOW"] = work["Date"].dt.dayofweek.astype(np.int8)
    work["DAY"] = work["Date"].dt.day.astype(np.int8)
    work["MONTH"] = work["Date"].dt.month.astype(np.int8)
    
    work["DOW_SIN"] = np.sin(2 * np.pi * work["DOW"] / 7).astype(np.float32)
    work["DOW_COS"] = np.cos(2 * np.pi * work["DOW"] / 7).astype(np.float32)

    # Global
    work["SUM3"] = (work["H"] + work["T"] + work["O"]).astype(np.int8)
    work["SUM2"] = (work["T2"] + work["O2"]).astype(np.int8)
    work["RANGE3"] = (work[["H", "T", "O"]].max(axis=1) - work[["H", "T", "O"]].min(axis=1)).astype(np.int8)

    # Position Specific Features
    for pos in POSITIONS:
        pos_series = work[pos]
        shifted = pos_series.shift(1)
        
        # Lags
        for lag in range(1, 6):
            work[f"{pos}_L{lag}"] = pos_series.shift(lag).fillna(0).astype(np.int8)
            
        # Moving Averages & EMA (Improved Accuracy)
        work[f"{pos}_M5"] = shifted.rolling(5).mean().astype(np.float32)
        work[f"{pos}_EMA5"] = shifted.ewm(span=5, adjust=False).mean().astype(np.float32) # New!
        work[f"{pos}_S5"] = shifted.rolling(5).std().astype(np.float32)
        
        # Momentum (Trend strength)
        m10 = shifted.rolling(10).mean().astype(np.float32)
        work[f"{pos}_MOM"] = (work[f"{pos}_EMA5"] - m10).astype(np.float32) # New!

        # Diffs
        work[f"{pos}_D1"] = (shifted - pos_series.shift(2)).fillna(0).astype(np.int8)
        
        # Categorical Logic
        shifted_fill = shifted.fillna(0).astype(np.int8)
        work[f"{pos}_ODD"] = (shifted_fill % 2).astype(np.int8)
        work[f"{pos}_HIGH"] = (shifted_fill >= 5).astype(np.int8)
        work[f"{pos}_MIRROR"] = ((shifted_fill + 5) % 10).astype(np.int8)

    return work.fillna(0)

# ============================================================
# 8. FEATURE LIST (Updated)
# ============================================================
BASE_FEATURES = ["DOW", "DAY", "MONTH", "DOW_SIN", "DOW_COS", "SUM3", "SUM2", "RANGE3"]
FEATURES = list(BASE_FEATURES)
for pos in POSITIONS:
    FEATURES.extend([
        f"{pos}_L1", f"{pos}_L2", f"{pos}_L3", f"{pos}_L4", f"{pos}_L5",
        f"{pos}_M5", f"{pos}_EMA5", f"{pos}_S5", f"{pos}_MOM", # Using EMA and Momentum
        f"{pos}_D1", f"{pos}_ODD", f"{pos}_HIGH", f"{pos}_MIRROR"
    ])

# ============================================================
# 9. ADAPTIVE CONFIG (Tuned for generalisation & speed)
# ============================================================
def get_adaptive_config(n):
    if n >= 700: return {"backtest": 12, "trees": 120, "depth": 7, "leaf": 2} # slightly shallower depth = less overfitting
    if n >= 400: return {"backtest": 12, "trees": 100, "depth": 6, "leaf": 2}
    if n >= 200: return {"backtest": 10, "trees": 80,  "depth": 5, "leaf": 2}
    return {"backtest": 8, "trees": 50, "depth": 4, "leaf": 3}

# ============================================================
# 10. MODEL FACTORY
# ============================================================
def create_model(model_name, config):
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=config["trees"], max_depth=config["depth"], min_samples_leaf=config["leaf"], max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
    if model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=config["trees"], max_depth=config["depth"], min_samples_leaf=config["leaf"], max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=max(30, int(config["trees"]*0.7)), max_leaf_nodes=15, learning_rate=0.08, l2_regularization=1.0, random_state=42)

def probability_vector(model, X):
    raw = model.predict_proba(X)[0]
    output = np.zeros(10, dtype=float)
    for cls, prob in zip(model.classes_, raw):
        if 0 <= int(cls) <= 9: output[int(cls)] = float(prob)
    return output / output.sum() if output.sum() > 0 else np.ones(10)/10

def calculate_metrics(probs, actual):
    ranking = np.argsort(probs)[::-1]
    return {
        "top1": int(actual in ranking[:1]),
        "top3": int(actual in ranking[:3]),
        "top5": int(actual in ranking[:5]),
        "dead7": int(actual in np.argsort(probs)[:7]), # Dead7 check
        "logloss": float(-np.log(max(probs[int(actual)], 1e-9)))
    }

# ============================================================
# 11. WALK-FORWARD BACKTEST
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def adaptive_backtest(df_features, position, config):
    if len(df_features) < 50: return {"best_model": "ExtraTrees", "scores": {}, "tests": 0}

    models = ["ExtraTrees", "RandomForest", "HistGradientBoosting"]
    X, y = df_features[FEATURES], df_features[position].astype(int)
    start = len(df_features) - min(config["backtest"], len(df_features) - 30)
    scores = {}

    for model_name in models:
        stats = {"top1":0, "top3":0, "top5":0, "dead7":0, "logloss":0.0, "tests":0}
        
        for test_idx in range(start, len(df_features)):
            if test_idx < 25: continue
            
            X_train, y_train = X.iloc[:test_idx], y.iloc[:test_idx]
            if y_train.nunique() < 2: continue
                
            try:
                model = create_model(model_name, config).fit(X_train, y_train)
                metrics = calculate_metrics(probability_vector(model, X.iloc[[test_idx]]), int(y.iloc[test_idx]))
                for k in stats.keys(): 
                    if k != "tests": stats[k] += metrics[k]
                stats["tests"] += 1
            except: continue

        if stats["tests"] == 0:
            scores[model_name] = {"score": -999, "tests": 0}
            continue

        # Normalize metrics
        t = stats["tests"]
        avg_metrics = {k: v/t for k, v in stats.items() if k != "tests"}
        
        # Calculate adaptive score
        score = (avg_metrics["top1"] * 0.3) + (avg_metrics["top3"] * 0.25) + (avg_metrics["top5"] * 0.3) + (avg_metrics["dead7"] * 0.05) + (max(0, 1/(1+avg_metrics["logloss"])) * 0.1)
        
        scores[model_name] = {**avg_metrics, "tests": t, "score": score}

    best_model = max(scores, key=lambda x: scores[x]["score"]) if scores else "ExtraTrees"
    return {"best_model": best_model, "scores": scores, "tests": max([v.get("tests", 0) for v in scores.values()], default=0)}

# ============================================================
# 12. FINAL PREDICTION
# ============================================================
@st.cache_resource(show_spinner=False)
def train_final_model(X_train, y_train, model_name, config):
    return create_model(model_name, config).fit(X_train, y_train)

def final_prediction(df_features, position, selected_model, config):
    X, y = df_features[FEATURES], df_features[position].astype(int)
    start_idx = max(0, len(X) - 20 - 1000) # Keep max 1000 rows for speed memory
    
    model = train_final_model(X.iloc[start_idx:-1], y.iloc[start_idx:-1], selected_model, config)
    probs = probability_vector(model, X.iloc[[-1]])
    
    return {
        "model": selected_model,
        "hot": [(int(i), float(probs[i])) for i in np.argsort(probs)[::-1][:5]],
        "dead": [(int(i), float(probs[i])) for i in np.argsort(probs)[:7]]
    }

# ============================================================
# 13. MAIN APP ROUTING & UI RENDERING
# ============================================================
def main():
    inject_css()
    st.markdown('<div class="main-title">🤖 LOTTO AI PRO V7.1</div><div class="subtitle">ADAPTIVE • OPTIMIZED SPEED • ENHANCED ACCURACY</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: lottery = st.selectbox("🏷️ เลือกประเภทหวย", list(LOTTERY_SOURCES.keys()))
    with col2: selected_day = st.selectbox("📅 วันเป้าหมาย", ["อัตโนมัติ"] + DOW_NAMES)
    
    if not st.button("🚀 เริ่มวิเคราะห์ PRO V7.1", type="primary", use_container_width=True):
        return

    # Data Fetching
    with st.spinner("📥 กำลังดึงข้อมูลย้อนหลัง..."):
        df = fetch_lottery_data(LOTTERY_SOURCES[lottery])
        
    if len(df) < 50:
        st.error("❌ ข้อมูลไม่เพียงพอ (ต้องการอย่างน้อย 50 งวด)")
        return

    # Target Date setup
    last_date = pd.Timestamp(df["Date"].iloc[-1])
    day_map = {n: i for i, n in enumerate(DOW_NAMES)}
    if selected_day == "อัตโนมัติ":
        gap = (last_date - pd.Timestamp(df["Date"].iloc[-2])).days if len(df) >= 2 else 7
        target_date = last_date + timedelta(days=gap if gap > 0 else 7)
    else:
        days_ahead = day_map[selected_day] - last_date.dayofweek
        target_date = last_date + timedelta(days=days_ahead if days_ahead > 0 else days_ahead + 7)

    # Feature Engineering
    with st.spinner("🧠 กำลังสร้าง AI Features (EMA & Momentum)..."):
        extended = pd.concat([df, pd.DataFrame([{"Date": target_date, "Result_3D": "000", "Result_2D": "00"}])], ignore_index=True)
        feature_df = build_features(extended)
        config = get_adaptive_config(len(df))

    # Analysis
    progress = st.progress(0)
    backtest_results, final_results = {}, {}
    
    status_text = st.empty()
    
    for idx, pos in enumerate(POSITIONS):
        status_text.text(f"⏳ กำลังวิเคราะห์หลัก {pos}...")
        bt = adaptive_backtest(feature_df.iloc[:-1], pos, config)
        backtest_results[pos] = bt
        final_results[pos] = final_prediction(feature_df, pos, bt["best_model"], config)
        progress.progress(int((idx + 1) / len(POSITIONS) * 100))
        
    progress.empty()
    status_text.empty()

    # Show Status Card
    st.markdown(
        f"""
        <div class="status-card">
            <b>📊 ข้อมูลย้อนหลัง:</b> {len(df):,} งวด &nbsp;|&nbsp;
            <b>📅 งวดเป้าหมาย:</b> {target_date.strftime('%d/%m/%Y')} ({DOW_NAMES[target_date.dayofweek]}) &nbsp;|&nbsp;
            <b>🔄 Backtest:</b> {config['backtest']} งวด
        </div>
        <br>
        """, unsafe_allow_html=True
    )

    # Result Tabs
    tab_res, tab_acc = st.tabs(["🎯 ผลการวิเคราะห์ (เด่น-ดับ)", "📊 ประสิทธิภาพโมเดล (Backtest)"])

    with tab_res:
        st.subheader("💡 การคาดการณ์จาก AI (Top 5 & Bottom 7)")
        for pos in POSITIONS:
            st.markdown(f'<div class="pos-title">{POSITION_LABELS[pos]}</div>', unsafe_allow_html=True)
            col_hot, col_dead = st.columns(2)
            
            with col_hot:
                hot = final_results[pos]["hot"]
                st.markdown(f"""
                <div class="number-card hot-card">
                    <div style="font-weight:bold; color:#15803d; margin-bottom:5px;">🎯 เลขเด่น TOP 5</div>
                    <div class="num-highlight-hot">{" - ".join(str(n) for n, _ in hot)}</div>
                    <div class="prob-text">{" | ".join(f"{n}: {p*100:.0f}%" for n, p in hot)}</div>
                    <div class="model-badge">🤖 {final_results[pos]["model"]}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_dead:
                dead = final_results[pos]["dead"]
                st.markdown(f"""
                <div class="number-card dead-card">
                    <div style="font-weight:bold; color:#b91c1c; margin-bottom:5px;">🛑 เลขดับ 7 ตัว (โอกาสต่ำ)</div>
                    <div class="num-highlight-dead">{" - ".join(str(n) for n, _ in dead)}</div>
                    <div class="prob-text">{" | ".join(f"{n}: {p*100:.0f}%" for n, p in dead)}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_acc:
        st.subheader("⚙️ ความแม่นยำรายหลัก (Walk-Forward Validation)")
        summary = []
        for pos in POSITIONS:
            bt = backtest_results[pos]
            best = bt["best_model"]
            best_score = bt["scores"].get(best, {})
            summary.append({
                "ตำแหน่ง": POSITION_LABELS[pos],
                "AI ที่เลือก": best,
                "Top-1": f"{best_score.get('top1', 0)*100:.1f}%",
                "Top-3": f"{best_score.get('top3', 0)*100:.1f}%",
                "Top-5": f"{best_score.get('top5', 0)*100:.1f}%",
                "Dead-7 Coverage": f"{best_score.get('dead7', 0)*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
