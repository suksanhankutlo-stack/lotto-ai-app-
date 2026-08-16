# ============================================================
# 🤖 LOTTO AI PRO V8.4.0 OPTIMIZED (Clean UI & Tuned Model)
# ============================================================
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMLIT CONFIG & CSS
# ============================================================
st.set_page_config(
    page_title="Lotto AI V8.4.0 Optimized",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_css():
    st.markdown("""
        <style>
        .stApp { background-color: #f4f7f6; }
        .main-title { text-align: center; font-size: 2.5rem; font-weight: 900; color: #1e293b; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #64748b; font-size: 1rem; margin-bottom: 25px; font-weight: 500; }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            text-align: center;
        }
        .hot-card { border-top: 6px solid #10b981; }
        .dead-card { border-top: 6px solid #ef4444; }
        
        .pos-title { font-size: 1.2rem; font-weight: 700; color: #475569; margin-bottom: 10px; }
        .num-display { font-size: 3rem; font-weight: 900; letter-spacing: 5px; margin: 10px 0; }
        .hot-num { color: #10b981; }
        .dead-num { color: #ef4444; }
        
        .prob-details { font-size: 0.95rem; color: #64748b; font-weight: 600; background: #f8fafc; padding: 10px; border-radius: 8px; margin: 10px 0;}
        .meta-info { font-size: 0.85rem; color: #94a3b8; }
        
        .status-box {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
            padding: 20px;
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. CONSTANTS & MAPPINGS
# ============================================================
LOTTERY_SOURCES = {
    "หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "หวยธกส": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
}

DOW_NAMES = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
MODEL_NAMES = ["ExtraTrees", "HistGradientBoosting"]

THAI_POSITIONS = ["H1", "H2", "H3", "H4", "H5", "H6", "T2", "O2"]
NORMAL_POSITIONS = ["H", "T", "O", "T2", "O2"]

POSITION_LABELS = {
    "H1": "💯 หลักแสน", "H2": "🔢 หลักหมื่น", "H3": "🔢 หลักพัน",
    "H4": "💯 หลักร้อย (บน)", "H5": "🔟 หลักสิบ (บน)", "H6": "1️⃣ หลักหน่วย (บน)",
    "H": "💯 หลักร้อย (บน)", "T": "🔟 หลักสิบ (บน)", "O": "1️⃣ หลักหน่วย (บน)",
    "T2": "🔽 หลักสิบ (ล่าง)", "O2": "⬇️ หลักหน่วย (ล่าง)",
}

THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12
}

# ============================================================
# 3. SCRAPING & DATA PROCESSING
# ============================================================
def normalize_date(value):
    if not value: return None
    text = str(value).strip()
    
    # Try Thai text months
    for name, month in THAI_MONTHS.items():
        match = re.search(rf"(\d{{1,2}})\s*{re.escape(name)}\s*(\d{{4}})", text)
        if match:
            y = int(match.group(2))
            if y >= 2400: y -= 543
            try: return pd.Timestamp(y, month, int(match.group(1)))
            except: return None
            
    # Try standard dd/mm/yyyy or yyyy/mm/dd
    match = re.search(r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})", text)
    if match:
        a, b, c = map(int, match.groups())
        if a >= 1000: y, m, d = a, b, c
        else: y, m, d = c, b, a
        if y < 100: y += 2000
        if y >= 2400: y -= 543
        try: return pd.Timestamp(y, m, d)
        except: pass
    return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_lottery_data(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content", re.I)) or soup
        rows = []
        
        # Parse table rows
        for row in content.find_all("tr"):
            text = " ".join([c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])])
            if not text: continue
            date = normalize_date(text)
            if not date: continue
            
            six = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
            three = re.findall(r"(?<!\d)\d{3}(?!\d)", text)
            two = re.findall(r"(?<!\d)\d{2}(?!\d)", text)
            
            if six and two:
                rows.append({"Date": date, "Result_6D": six[0], "Result_3D": six[0][-3:], "Result_2D": two[-1]})
            elif three and two:
                rows.append({"Date": date, "Result_6D": None, "Result_3D": three[0], "Result_2D": two[-1]})

        if not rows:
            raise Exception("ไม่พบข้อมูลสลาก (อาจมีการเปลี่ยนรูปแบบเว็บ)")
            
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Result_3D"] = df["Result_3D"].astype(str).str.zfill(3)
        df["Result_2D"] = df["Result_2D"].astype(str).str.zfill(2)
        if "Result_6D" in df.columns:
            df["Result_6D"] = df["Result_6D"].astype(str).str.extract(r"(\d{6})")[0]
            
        return df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    except Exception as exc:
        raise Exception(f"โหลดข้อมูลไม่สำเร็จ: {exc}")

def is_thai_6d(df):
    return "Result_6D" in df.columns and df["Result_6D"].notna().sum() >= 10

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def build_features(df, thai_6d=False):
    w = df.copy()
    
    # Target variables
    if thai_6d:
        six = w["Result_6D"].fillna("000000").astype(str).str.zfill(6)
        for i in range(6): w[f"H{i+1}"] = six.str[i].astype(np.int8)
    else:
        three = w["Result_3D"].astype(str).str.zfill(3)
        w["H"] = three.str[0].astype(np.int8)
        w["T"] = three.str[1].astype(np.int8)
        w["O"] = three.str[2].astype(np.int8)

    two = w["Result_2D"].astype(str).str.zfill(2)
    w["T2"] = two.str[0].astype(np.int8)
    w["O2"] = two.str[1].astype(np.int8)

    # Date features
    dt = w["Date"].dt
    w["DOW"], w["DAY"], w["MONTH"] = dt.dayofweek.astype(np.int8), dt.day.astype(np.int8), dt.month.astype(np.int8)
    w["DAY_OF_YEAR"] = dt.dayofyear.astype(np.int16)
    w["DOW_SIN"] = np.sin(2 * np.pi * w["DOW"] / 7).astype(np.float32)
    w["DOW_COS"] = np.cos(2 * np.pi * w["DOW"] / 7).astype(np.float32)

    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS
    
    # Engineered Features
    for pos in positions:
        s = w[pos]
        p = s.shift(1)
        
        for lag in (1, 2, 3, 5): w[f"{pos}_L{lag}"] = s.shift(lag)
        
        for window in (10, 20):
            r = p.rolling(window, min_periods=2)
            w[f"{pos}_M{window}"] = r.mean()
            w[f"{pos}_S{window}"] = r.std()
            for digit in (0, 5):
                w[f"{pos}_F{window}_{digit}"] = (p == digit).astype(np.float32).rolling(window, min_periods=2).mean()
                
        w[f"{pos}_D1"] = s.shift(1) - s.shift(2)
        w[f"{pos}_D2"] = s.shift(2) - s.shift(3)
        w[f"{pos}_ODD"] = p % 2
        w[f"{pos}_HIGH"] = (p >= 5).astype(np.float32)
        w[f"{pos}_MOD3"] = p % 3
        w[f"{pos}_SIN"] = np.sin(2 * np.pi * p / 10).astype(np.float32)
        w[f"{pos}_COS"] = np.cos(2 * np.pi * p / 10).astype(np.float32)
        w[f"{pos}_EWMA7"] = p.ewm(span=7, adjust=False).mean()
        w[f"{pos}_REPEAT"] = (p == s.shift(2)).astype(np.float32)

    return w.replace([np.inf, -np.inf], np.nan)

def get_features(thai_6d):
    base = ["DOW", "DAY", "MONTH", "DAY_OF_YEAR", "DOW_SIN", "DOW_COS"]
    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS
    for pos in positions:
        base.extend([
            f"{pos}_L1", f"{pos}_L2", f"{pos}_L3", f"{pos}_L5",
            f"{pos}_M10", f"{pos}_M20", f"{pos}_S10", f"{pos}_S20",
            f"{pos}_D1", f"{pos}_D2", f"{pos}_ODD", f"{pos}_HIGH", f"{pos}_MOD3",
            f"{pos}_SIN", f"{pos}_COS", f"{pos}_EWMA7", f"{pos}_REPEAT"
        ])
        for window in (10, 20):
            for digit in (0, 5):
                base.append(f"{pos}_F{window}_{digit}")
    return list(dict.fromkeys(base))

# ============================================================
# 5. MACHINE LEARNING
# ============================================================
def get_adaptive_config(n):
    # Tuned hyperparameters for better generalization
    if n >= 700: return {"min_train": 120, "trees": 100, "depth": 6, "leaf": 4, "selected_features": 20, "backtest_points": 6, "recent_decay": 0.985}
    if n >= 400: return {"min_train": 100, "trees": 80,  "depth": 5, "leaf": 4, "selected_features": 18, "backtest_points": 6, "recent_decay": 0.980}
    if n >= 200: return {"min_train": 80,  "trees": 60,  "depth": 5, "leaf": 3, "selected_features": 16, "backtest_points": 5, "recent_decay": 0.975}
    return              {"min_train": 50,  "trees": 40,  "depth": 4, "leaf": 2, "selected_features": 14, "backtest_points": 4, "recent_decay": 0.970}

def create_model(name, cfg, system="hot"):
    t, d, l = cfg["trees"], cfg["depth"], cfg["leaf"]
    
    if system == "hot":
        if name == "ExtraTrees":
            return ExtraTreesClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, max_features=0.6, class_weight="balanced", n_jobs=-1, random_state=42)
        return HistGradientBoostingClassifier(max_iter=t, max_leaf_nodes=15, learning_rate=0.05, min_samples_leaf=l, l2_regularization=3.0, random_state=42)
    
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=t, max_depth=max(3, d-1), min_samples_leaf=l+1, max_features=0.5, class_weight="balanced", n_jobs=-1, random_state=91)
    return HistGradientBoostingClassifier(max_iter=int(t*0.8), max_leaf_nodes=10, learning_rate=0.04, min_samples_leaf=l+1, l2_regularization=5.0, random_state=91)

def select_features_once(X, y, max_features, system="hot"):
    cols = list(X.columns)
    if len(cols) <= max_features: return cols
    valid = [c for c in cols if X[c].nunique(dropna=False) > 1]
    if len(valid) <= max_features: return valid
    
    Xi = X[valid].replace([np.inf, -np.inf], np.nan).astype(np.float32).fillna(0.0)
    selector = ExtraTreesClassifier(n_estimators=20, max_depth=4, min_samples_leaf=3, n_jobs=-1, random_state=123 if system=="hot" else 321)
    selector.fit(Xi, y)
    order = np.argsort(selector.feature_importances_)[::-1]
    return [valid[i] for i in order[:max_features]]

def normalize_probability(p):
    p = np.clip(np.nan_to_num(np.asarray(p, dtype=np.float32)), 1e-9, None)
    return p / p.sum() if p.sum() > 0 else np.ones(10, dtype=np.float32) / 10

def model_probability(X_train, y_train, X_test, cfg, selected, system):
    A = X_train[selected].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    B = X_test[selected].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    med = A.median()
    A, B = A.fillna(med).fillna(0.0), B.fillna(med).fillna(0.0)
    
    predictions = []
    for name in MODEL_NAMES:
        try:
            model = create_model(name, cfg, system)
            model.fit(A, y_train)
            raw = model.predict_proba(B)[0]
            out = np.zeros(10, dtype=np.float32)
            for cls, prob in zip(model.classes_, raw):
                if 0 <= int(cls) <= 9: out[int(cls)] = prob
            predictions.append(normalize_probability(out))
        except: continue
        
    return normalize_probability(np.mean(predictions, axis=0)) if predictions else np.ones(10)/10

def predict_system(X_train, y_train, X_test, cfg, system="hot"):
    selected = select_features_once(X_train, y_train, cfg["selected_features"], system)
    prob = model_probability(X_train, y_train, X_test, cfg, selected, system)
    
    if system == "hot":
        order = np.argsort(prob)[::-1]
        hot = [(int(n), float(prob[n])) for n in order[:3]]
        conf = float(prob[order[0]] - prob[order[1]]) if len(order) >= 2 else 0.0
        return {"probability": prob, "hot": hot, "confidence": conf, "top_cov": float(prob[order[:3]].sum())}
    else:
        dead_score = normalize_probability(1.0 - prob)
        order = np.argsort(dead_score)[::-1]
        dead = [(int(n), float(dead_score[n])) for n in order[:5]]
        return {"probability": prob, "dead_score": dead_score, "dead": dead, "top_cov": float(dead_score[order[:5]].sum())}

def final_prediction(df_feat, pos, features, cfg):
    X = df_feat[features].astype(np.float32)
    y = df_feat[pos].astype(np.int8)
    return {
        "hot": predict_system(X.iloc[:-1], y.iloc[:-1], X.iloc[[-1]], cfg, "hot"),
        "dead": predict_system(X.iloc[:-1], y.iloc[:-1], X.iloc[[-1]], cfg, "dead")
    }

# ============================================================
# 6. UI COMPONENTS
# ============================================================
def display_card(pos, result, is_hot=True):
    sys_key = "hot" if is_hot else "dead"
    data = result[sys_key][sys_key]
    
    nums_str = " - ".join([str(n) for n, _ in data])
    probs_str = " | ".join([f"{n}: {p*100:.1f}%" for n, p in data])
    
    card_class = "hot-card" if is_hot else "dead-card"
    num_class = "hot-num" if is_hot else "dead-num"
    title_icon = "🔥 เด่น TOP-3" if is_hot else "🛑 ดับ TOP-5"
    
    html = f"""
    <div class="card {card_class}">
        <div class="pos-title">{POSITION_LABELS[pos]}</div>
        <div class="num-display {num_class}">{nums_str}</div>
        <div class="prob-details">
            {title_icon} โอกาส: {probs_str}
        </div>
        <div class="meta-info">
            ความครอบคลุม: {result[sys_key]['top_cov']*100:.1f}% 
            {'| Gap: ' + str(round(result[sys_key].get('confidence', 0)*100, 1)) + '%' if is_hot else ''}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# 7. MAIN APP
# ============================================================
def main():
    inject_css()
    
    st.markdown("""
        <div class="main-title">🤖 LOTTO AI PRO V8.4.0</div>
        <div class="subtitle">Optimized Machine Learning System | Clean UI</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    lottery = col1.selectbox("🏷️ เลือกประเภทสลาก", list(LOTTERY_SOURCES.keys()))
    selected_day = col2.selectbox("📅 วันเป้าหมายงวดถัดไป", ["อัตโนมัติ"] + DOW_NAMES)

    if not st.button("🚀 เริ่มวิเคราะห์ด้วย AI", type="primary", use_container_width=True):
        return

    with st.spinner("📥 กำลังดึงข้อมูลสถิติย้อนหลัง..."):
        try:
            df = fetch_lottery_data(LOTTERY_SOURCES[lottery])
        except Exception as exc:
            st.error(str(exc))
            return

    if len(df) < 50:
        st.error(f"❌ มีข้อมูลเพียง {len(df)} งวด (ต้องการอย่างน้อย 50 งวดเพื่อเทรนโมเดล)")
        return

    thai_6d = (lottery == "หวยไทย" and is_thai_6d(df))
    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS
    
    # Calculate target date
    last_date = pd.Timestamp(df["Date"].iloc[-1])
    days_ahead = 7
    if selected_day == "อัตโนมัติ" and len(df) >= 2:
        days_ahead = max(int((df["Date"].iloc[-1] - df["Date"].iloc[-2]).days), 1)
    elif selected_day != "อัตโนมัติ":
        days_ahead = (DOW_NAMES.index(selected_day) - last_date.dayofweek) % 7 or 7
        
    target_date = last_date + timedelta(days=days_ahead)

    # Prepare data for prediction
    dummy = {"Date": target_date, "Result_3D": "000", "Result_2D": "00"}
    if thai_6d: dummy["Result_6D"] = "000000"
    ext = pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)

    with st.spinner("⚡ กำลังประมวลผล Feature Engineering และฝึกสอนโมเดล..."):
        feat = build_features(ext, thai_6d)
        features = get_features(thai_6d)
        cfg = get_adaptive_config(len(df))
        
        final = {}
        progress = st.progress(0)
        for i, pos in enumerate(positions):
            final[pos] = final_prediction(feat, pos, features, cfg)
            progress.progress(int(((i + 1) / len(positions)) * 100))
        progress.empty()

    # System Status Header
    mode_text = "6 หลัก + 2 หลัก" if thai_6d else "3 หลัก + 2 หลัก"
    st.markdown(f"""
        <div class="status-box">
            <h3 style="margin:0 0 10px 0;">📊 รายงานการวิเคราะห์ (Analysis Report)</h3>
            • <b>ฐานข้อมูล:</b> {len(df):,} งวดย้อนหลัง<br>
            • <b>วิเคราะห์สำหรับงวด:</b> {target_date.strftime('%d/%m/%Y')} ({mode_text})<br>
            • <b>AI Config:</b> {cfg['trees']} Trees / Depth {cfg['depth']} / Features {cfg['selected_features']}
        </div>
    """, unsafe_allow_html=True)

    # Display Results in Tabs
    t1, t2, t3 = st.tabs(["🔥 สรุปเลขเด่น (HOT)", "🛑 สรุปเลขดับ (DEAD)", "📋 ตารางภาพรวม"])
    
    with t1:
        st.subheader("🔥 โพยเลขเด่น TOP-3")
        cols = st.columns(3)
        for i, pos in enumerate(positions):
            with cols[i % 3]: display_card(pos, final[pos], is_hot=True)
            
    with t2:
        st.subheader("🛑 โพยเลขดับ (ความน่าจะเป็นต่ำ) TOP-5")
        cols = st.columns(3)
        for i, pos in enumerate(positions):
            with cols[i % 3]: display_card(pos, final[pos], is_hot=False)
            
    with t3:
        st.subheader("📋 ตารางสรุปภาพรวมทุกตำแหน่ง")
        data_summary = []
        for pos in positions:
            hot_data = final[pos]["hot"]["hot"]
            dead_data = final[pos]["dead"]["dead"]
            data_summary.append({
                "ตำแหน่ง": POSITION_LABELS[pos],
                "🔥 เด่นอันดับ 1": f"{hot_data[0][0]} ({hot_data[0][1]*100:.1f}%)",
                "🔥 เด่นอันดับ 2": f"{hot_data[1][0]} ({hot_data[1][1]*100:.1f}%)",
                "🔥 เด่นอันดับ 3": f"{hot_data[2][0]} ({hot_data[2][1]*100:.1f}%)",
                "🛑 ดับ (ควรเลี่ยง)": " - ".join([str(n) for n, _ in dead_data])
            })
        st.dataframe(pd.DataFrame(data_summary), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
