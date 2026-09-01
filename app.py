# ============================================================
# 🤖 LOTTO AI PRO V8.7 FAST ADAPTIVE (AGGRESSIVE SELF-CORRECTING)
# ============================================================
# PERFORMANCE & ADAPTIVE UPGRADES:
#   1. Multi-threading Position Processing
#   2. Backtest Feature Selection Caching (Run Once per Pos)
#   3. Exact Digit Mapping (Fix swapped positions)
#   4. 2-Miss Fallback Auto-Correction System ⚡ (ปรับไวขึ้น)
# ============================================================

import re
import warnings
import concurrent.futures
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI V8.7 Auto-Correct",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_css():
    st.markdown("""
    <style>
    .stApp { background: #f4f6f9; font-family: 'Kanit', sans-serif; }
    .main-title { text-align:center; font-size:2.2rem; font-weight:900; color:#1e293b; }
    .subtitle { text-align:center; color:#64748b; font-size:.9rem; margin-bottom:25px; }
    .status-card { background:linear-gradient(135deg,#eff6ff,#dbeafe); border-radius:12px; padding:15px; text-align:center; color:#1e40af; font-weight:600; margin-bottom:20px; }
    .hot-card { background:white; border-left:8px solid #10b981; border-radius:12px; padding:20px; margin:10px 0; box-shadow:0 4px 10px rgba(0,0,0,.05); position: relative; }
    .dead-card { background:white; border-left:8px solid #ef4444; border-radius:12px; padding:20px; margin:10px 0; box-shadow:0 4px 10px rgba(0,0,0,.05); position: relative; }
    .position-title { font-size:1.2rem; font-weight:800; color:#334155; margin-bottom:10px; border-bottom:2px solid #f1f5f9; padding-bottom:5px; }
    .hot-number { font-size:2.5rem; font-weight:900; letter-spacing:4px; text-align:center; color:#10b981; }
    .dead-number { font-size:2.5rem; font-weight:900; letter-spacing:4px; text-align:center; color:#ef4444; text-decoration:line-through; text-decoration-color:rgba(239,68,68,.4); }
    .prob-text { text-align:center; color:#475569; font-size:.95rem; font-weight:600; margin-top:10px; padding:10px; background:#f8fafc; border-radius:8px; }
    .confidence { text-align:center; font-size:.85rem; font-weight:600; margin-top:10px; color:#64748b; }
    .fallback-badge { background:#fef08a; color:#854d0e; padding:5px 10px; border-radius:6px; font-size:0.8rem; font-weight:bold; margin-bottom:10px; display:inline-block; }
    div.stButton > button { width:100%; min-height:50px; border-radius:10px; font-size:1.1rem; font-weight:800; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. CONSTANTS & REGEX
# ============================================================

LOTTERY_SOURCES = {
    "หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "หวยธกส": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}

DOW_NAMES = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
THAI_POSITIONS = ["H1", "H2", "H3", "H4", "H5", "H6", "T2", "O2"]
NORMAL_POSITIONS = ["H", "T", "O", "T2", "O2"]

POSITION_LABELS = {
    "H1": "หลักแสน", "H2": "หลักหมื่น", "H3": "หลักพัน",
    "H4": "หลักร้อยบน", "H5": "หลักสิบบน", "H6": "หลักหน่วยบน",
    "H": "หลักร้อยบน", "T": "หลักสิบบน", "O": "หลักหน่วยบน",
    "T2": "หลักสิบล่าง", "O2": "หลักหน่วยล่าง"
}

THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12
}

MONTH_REGEXES = [(m, re.compile(rf"(\d{{1,2}})\s*{re.escape(n)}\s*(\d{{4}})", re.I)) for n, m in THAI_MONTHS.items()]
DATE_FORMAT_REGEX = re.compile(r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})")

# ============================================================
# 3. DATE & DATA EXTRACTION
# ============================================================

def normalize_date(value):
    if not value: return None
    text = str(value).strip()
    for month, regex in MONTH_REGEXES:
        match = regex.search(text)
        if match:
            y = int(match.group(2))
            if y >= 2400: y -= 543
            try: return pd.Timestamp(y, month, int(match.group(1)))
            except: return None
    match = DATE_FORMAT_REGEX.search(text)
    if match:
        a, b, c = map(int, match.groups())
        y, m, d = ((a, b, c) if a >= 1000 else (c, b, a))
        if y < 100: y += 2000
        if y >= 2400: y -= 543
        try: return pd.Timestamp(y, m, d)
        except: pass
    return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_lottery_data(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content", re.I)) or soup
        
        rows = []
        regex_6d = re.compile(r"(?<!\d)\d{6}(?!\d)")
        regex_3d = re.compile(r"(?<!\d)\d{3}(?!\d)")
        regex_2d = re.compile(r"(?<!\d)\d{2}(?!\d)")

        for row in content.find_all("tr"):
            text = " ".join(c.get_text(" ", strip=True) for c in row.find_all(["td", "th"]))
            if not text: continue
            
            date = normalize_date(text)
            if not date: continue

            six = regex_6d.findall(text)
            three = regex_3d.findall(text)
            two = regex_2d.findall(text)

            if six and two:
                rows.append({"Date": date, "Result_6D": six[0], "Result_3D": six[0][-3:], "Result_2D": two[-1]})
            elif three and two:
                rows.append({"Date": date, "Result_6D": None, "Result_3D": three[0], "Result_2D": two[-1]})

        if not rows: return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        df["Result_3D"] = df["Result_3D"].astype(str).str.extract(r'(\d+)')[0].str[-3:].str.zfill(3)
        df["Result_2D"] = df["Result_2D"].astype(str).str.extract(r'(\d+)')[0].str[-2:].str.zfill(2)

        if "Result_6D" in df.columns:
            df["Result_6D"] = df["Result_6D"].astype(str).str.extract(r'(\d+)')[0].str[-6:].str.zfill(6)

        df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

def is_thai_6d(df):
    return "Result_6D" in df.columns and df["Result_6D"].notna().sum() >= 10

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def build_features(df, thai_6d=False):
    w = df.copy()

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

    dt = w["Date"].dt
    w["DOW"] = dt.dayofweek.astype(np.int8)
    w["DAY"] = dt.day.astype(np.int8)
    w["MONTH"] = dt.month.astype(np.int8)
    w["DOW_SIN"] = np.sin(2 * np.pi * w["DOW"] / 7).astype(np.float32)
    w["DOW_COS"] = np.cos(2 * np.pi * w["DOW"] / 7).astype(np.float32)
    
    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS

    for pos in positions:
        s = w[pos]
        p = s.shift(1)

        for lag in (1, 2, 3, 5): w[f"{pos}_L{lag}"] = s.shift(lag)
        for window in (5, 10, 20):
            w[f"{pos}_M{window}"] = p.rolling(window, min_periods=2).mean()
        
        w[f"{pos}_D1"] = s.shift(1) - s.shift(2)
        w[f"{pos}_MOMENTUM"] = p - s.shift(4)
        roll_20 = p.rolling(20, min_periods=2)
        w[f"{pos}_VOL20"] = roll_20.max() - roll_20.min()
        w[f"{pos}_ODD"] = p % 2
        w[f"{pos}_HIGH"] = (p >= 5).astype(np.float32)
        w[f"{pos}_REPEAT"] = (p == s.shift(2)).astype(np.float32)

    return w.replace([np.inf, -np.inf], np.nan).astype(np.float32, errors="ignore")

def get_features(thai_6d):
    base = ["DOW", "DAY", "MONTH", "DOW_SIN", "DOW_COS"]
    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS
    for pos in positions:
        base.extend([f"{pos}_L{lag}" for lag in (1, 2, 3, 5)])
        base.extend([f"{pos}_M{w}" for w in (5, 10, 20)])
        base.extend([f"{pos}_D1", f"{pos}_MOMENTUM", f"{pos}_VOL20", f"{pos}_ODD", f"{pos}_HIGH", f"{pos}_REPEAT"])
    return list(dict.fromkeys(base))

def get_adaptive_config(n):
    if n >= 700: return {"min_train": 140, "train_window": 500, "trees": 55, "depth": 7, "leaf": 3, "selected_features": 24, "selector_trees": 8, "decay": 0.997}
    if n >= 400: return {"min_train": 110, "train_window": 400, "trees": 45, "depth": 6, "leaf": 3, "selected_features": 22, "selector_trees": 7, "decay": 0.996}
    if n >= 200: return {"min_train": 90, "train_window": 300, "trees": 38, "depth": 5, "leaf": 3, "selected_features": 19, "selector_trees": 6, "decay": 0.994}
    return {"min_train": 60, "train_window": 220, "trees": 30, "depth": 4, "leaf": 3, "selected_features": 16, "selector_trees": 5, "decay": 0.992}

# ============================================================
# 5. ML MODELS & FALLBACK LOGIC
# ============================================================

def normalize_probability(p):
    p = np.asarray(p, dtype=np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 1e-9, None)
    total = p.sum()
    if total <= 0: return np.ones(10, dtype=np.float32) / 10
    return (p / total).astype(np.float32)

def make_recent_weights(n, decay):
    distance = (n - 1 - np.arange(n))
    weights = (decay ** distance)
    return (weights / weights.mean()).astype(np.float32)

def select_features_once(X, y, max_features, cfg):
    valid = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
    if len(valid) <= max_features: return valid
    try:
        selector = ExtraTreesClassifier(n_estimators=cfg["selector_trees"], max_depth=4, n_jobs=-1, random_state=123)
        selector.fit(X[valid].fillna(0).astype(np.float32), y)
        return [valid[i] for i in np.argsort(selector.feature_importances_)[::-1][:max_features]]
    except: return valid[:max_features]

def ensemble_probability(X_train, y_train, X_test, cfg, selected):
    A = X_train[selected].astype(np.float32).fillna(0)
    B = X_test[selected].astype(np.float32).fillna(0)
    sample_weights = make_recent_weights(len(A), cfg["decay"])
    
    model_outputs = []
    try:
        model_et = ExtraTreesClassifier(n_estimators=cfg["trees"], max_depth=cfg["depth"], min_samples_leaf=cfg["leaf"], max_features="sqrt", n_jobs=-1, random_state=42)
        model_et.fit(A, y_train, sample_weight=sample_weights)
        p = np.zeros(10, dtype=np.float32)
        for cls, prob in zip(model_et.classes_, model_et.predict_proba(B)[0]):
            if 0 <= cls <= 9: p[int(cls)] = prob
        model_outputs.append((normalize_probability(p), 0.40))
    except: pass

    try:
        model_hgb = HistGradientBoostingClassifier(max_iter=max(30, int(cfg["trees"]*0.75)), max_leaf_nodes=15, learning_rate=0.035, random_state=52)
        model_hgb.fit(A, y_train, sample_weight=sample_weights)
        p = np.zeros(10, dtype=np.float32)
        for cls, prob in zip(model_hgb.classes_, model_hgb.predict_proba(B)[0]):
            if 0 <= cls <= 9: p[int(cls)] = prob
        model_outputs.append((normalize_probability(p), 0.60))
    except: pass

    if not model_outputs: return np.ones(10, dtype=np.float32) / 10
    
    result = sum(p * w for p, w in model_outputs)
    return normalize_probability(result)

def run_system_pair(X_train, y_train, X_test, cfg):
    selected = select_features_once(X_train, y_train, cfg["selected_features"], cfg)
    prob = ensemble_probability(X_train, y_train, X_test, cfg, selected)

    order_hot = np.argsort(prob)[::-1]
    order_dead = np.argsort(prob)
    return {
        "probability": prob,
        "hot_results": [(int(n), float(prob[n])) for n in order_hot[:3]],
        "dead_results": [(int(n), float(normalize_probability(1.0 - prob)[n])) for n in order_dead[:3]],
        "confidence": float(prob[order_hot[0]]) - float(prob[order_hot[1]]),
        "hot_coverage": float(prob[order_hot[:3]].sum()),
        "selected": selected
    }

def compute_fallback_prob(df_feat, pos):
    recent = df_feat[pos].iloc[-61:-1].dropna().astype(int)
    freq = np.zeros(10)
    weights = np.linspace(0.2, 1.0, len(recent)) 
    for val, w in zip(recent, weights):
        freq[val] += w
        
    prob = freq / (freq.sum() + 1e-9)
    prob = (prob * 0.7) + 0.03 
    return normalize_probability(prob)

# ============================================================
# 6. BACKTEST & PREDICTION PROCESS
# ============================================================

def run_backtest_for_pos(df_feat, pos, features, cfg, steps):
    results = []
    bt_cfg = cfg.copy()
    bt_cfg["trees"] = max(18, cfg["trees"] // 2)
    bt_cfg["selected_features"] = max(12, cfg["selected_features"] - 3)
    
    target_start = len(df_feat) - 1
    start_idx = max(0, target_start - bt_cfg["train_window"])
    X_tr_full = df_feat[features].iloc[start_idx:target_start]
    y_tr_full = df_feat[pos].astype(np.int8).iloc[start_idx:target_start]
    
    selected_for_bt = select_features_once(X_tr_full, y_tr_full, bt_cfg["selected_features"], bt_cfg) if len(X_tr_full) >= bt_cfg["min_train"] else features[:10]

    for step in range(steps, 0, -1):
        target_idx = len(df_feat) - 1 - step
        if target_idx <= 0: continue
            
        start = max(0, target_idx - bt_cfg["train_window"])
        X_train = df_feat[features].iloc[start:target_idx]
        y_train = df_feat[pos].astype(np.int8).iloc[start:target_idx]
        if len(X_train) < bt_cfg["min_train"]: continue
            
        X_test = df_feat[features].iloc[[target_idx]]
        actual = int(df_feat[pos].iloc[target_idx])
        date_val = pd.to_datetime(df_feat["Date"].iloc[target_idx]).strftime("%d/%m/%Y")

        prob = ensemble_probability(X_train, y_train, X_test, bt_cfg, selected_for_bt)
        order_hot = np.argsort(prob)[::-1]
        hot_top3 = [int(n) for n in order_hot[:3]]
        dead_top3 = [int(n) for n in np.argsort(prob)[:3]]

        results.append({
            "วันที่": date_val,
            "ผลจริง": actual,
            "อันดับจริง": int(np.where(order_hot == actual)[0][0]) + 1,
            "ทายเด่น Top3": " - ".join(map(str, hot_top3)),
            "ผลเด่น": "✅ เข้า" if actual in hot_top3 else "❌ หลุด",
            "ทายดับ Top3": " - ".join(map(str, dead_top3)),
            "ผลดับ": "✅ ผ่าน" if actual not in dead_top3 else "❌ ตาย"
        })

    return pd.DataFrame(results)

# ============================================================
# DISPLAY & MAIN
# ============================================================

def display_card(pos, data, is_hot=True):
    fallback_html = "<div class='fallback-badge'>⚠️ เข้าสู่โหมดแก้ไขตัวเองอัตโนมัติ (ปรับสถิติใหม่เนื่องจากหลุด 2 งวดติด)</div>" if data.get("is_fallback") else ""
    
    if is_hot:
        items = data["hot_results"]
        nums = " - ".join(str(n) for n, p in items)
        probs = " | ".join(f"{n}: {p*100:.1f}%" for n, p in items)
        html = f"""
        <div class="hot-card">
            {fallback_html}
            <div class="position-title">🎯 {POSITION_LABELS[pos]}</div>
            <div class="hot-number">{nums}</div>
            <div class="prob-text">🔥 HOT TOP-3<br>{probs}</div>
            <div class="confidence">📌 Gap: {data["confidence"]*100:.1f}% &nbsp;|&nbsp; Coverage: {data["hot_coverage"]*100:.1f}%</div>
        </div>
        """
    else:
        items = data["dead_results"]
        nums = " - ".join(str(n) for n, p in items)
        probs = " | ".join(f"{n}: {p*100:.1f}%" for n, p in items)
        html = f"""
        <div class="dead-card">
            {fallback_html}
            <div class="position-title">🛑 {POSITION_LABELS[pos]}</div>
            <div class="dead-number">{nums}</div>
            <div class="prob-text">🛑 DEAD SCORE TOP-3<br>{probs}</div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

def main():
    inject_css()
    st.markdown("<div class='main-title'>🤖 LOTTO AI PRO V8.7</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>⚡ แก้ไขสลับหลัก 100% • ⚠️ มีโหมดตรวจจับและแก้ไขตัวเองหากผิดพลาด 2 งวดติด</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    lottery = c1.selectbox("🏷️ เลือกประเภทหวย", list(LOTTERY_SOURCES.keys()))
    selected_day = c2.selectbox("📅 วันเป้าหมาย", ["อัตโนมัติ"] + DOW_NAMES)

    if not st.button("🚀 เริ่มวิเคราะห์ V8.7 AUTO-CORRECT", type="primary", use_container_width=True):
        return

    with st.spinner("📥 ประมวลผล Backtest & Auto-Correction (Multi-thread)..."):
        df = fetch_lottery_data(LOTTERY_SOURCES[lottery])
        if len(df) < 50:
            st.error(f"❌ ข้อมูลมีเพียง {len(df)} งวด (ต้องการอย่างน้อย 50 งวด)")
            return

        thai_6d = (lottery == "หวยไทย" and is_thai_6d(df))
        positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS

        last_date = pd.Timestamp(df["Date"].iloc[-1])
        days_ahead = 7 if selected_day == "อัตโนมัติ" else (DOW_NAMES.index(selected_day) - last_date.dayofweek) % 7 or 7
        target_date = last_date + timedelta(days=days_ahead)

        dummy = {"Date": target_date, "Result_3D": "000", "Result_2D": "00"}
        if thai_6d: dummy["Result_6D"] = "000000"
        
        ext = pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)
        feat = build_features(ext, thai_6d)
        features = get_features(thai_6d)
        cfg = get_adaptive_config(len(df))

        final = {}
        progress = st.progress(0)
        
        bt_steps = min(15, max(5, len(df) - cfg["min_train"])) 

        def process_position(pos):
            # 1. รัน Prediction ปกติ
            X = feat[features].iloc[:-1].tail(cfg["train_window"])
            y = feat[pos].astype(np.int8).iloc[:-1].tail(cfg["train_window"])
            X_test = feat[features].iloc[[-1]]
            res_final = run_system_pair(X, y, X_test, cfg)
            
            # 2. รัน Backtest
            res_bt = run_backtest_for_pos(feat, pos, features, cfg, steps=bt_steps)
            is_fallback = False
            
            # 3. ⚠️ LOGIC SELF-CORRECT (ตรวจสอบ 2 งวดติด)
            if res_bt is not None and len(res_bt) >= 2:
                recent_2 = res_bt.tail(2)
                if all(x == "❌ หลุด" for x in recent_2["ผลเด่น"]):
                    # ถ้าระบบหลักพัง ให้ดึงสถิติความน่าจะเป็นใหม่มาใช้ทับ (Fallback)
                    fallback_prob = compute_fallback_prob(feat, pos)
                    res_final["probability"] = fallback_prob
                    
                    order_hot = np.argsort(fallback_prob)[::-1]
                    order_dead = np.argsort(fallback_prob)
                    
                    res_final["hot_results"] = [(int(n), float(fallback_prob[n])) for n in order_hot[:3]]
                    dead_score = normalize_probability(1.0 - fallback_prob)
                    res_final["dead_results"] = [(int(n), float(dead_score[n])) for n in order_dead[:3]]
                    
                    res_final["confidence"] = float(fallback_prob[order_hot[0]]) - float(fallback_prob[order_hot[1]])
                    res_final["hot_coverage"] = float(fallback_prob[order_hot[:3]].sum())
                    is_fallback = True

            res_final["backtest"] = res_bt
            res_final["is_fallback"] = is_fallback
            return pos, res_final

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(positions)) as executor:
            futures = {executor.submit(process_position, pos): pos for pos in positions}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                pos, data = future.result()
                final[pos] = data
                progress.progress(int(((i + 1) / len(positions)) * 100))

        progress.empty()

    # ========================================================
    # SUMMARY
    # ========================================================
    st.markdown("### 📊 สรุปผล AI (เรียงหลักถูกต้อง 100%)")
    summary = []
    for pos in positions:
        hot = final[pos]["hot_results"]
        dead = final[pos]["dead_results"]
        bt = final[pos]["backtest"]
        bt_hot = (bt["ผลเด่น"] == "✅ เข้า").sum() / len(bt) if (bt is not None and not bt.empty) else 0
        
        fallback_str = "⚠️ Fallback" if final[pos]["is_fallback"] else ""
        summary.append({
            "ตำแหน่ง": f"{POSITION_LABELS[pos]} {fallback_str}",
            "🔥 HOT TOP3": " - ".join(str(n) for n, p in hot),
            "🛑 DEAD TOP3": " - ".join(str(n) for n, p in dead),
            "ความมั่นใจ AI": f"{final[pos]['hot_coverage']*100:.1f}%",
            "Win Rate (BT)": f"{bt_hot*100:.0f}%"
        })

    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.markdown("---")

    t1, t2, t3, t4 = st.tabs(["🔥 เจาะลึกเลขเด่น", "🛑 เจาะลึกเลขดับ", "📜 ประวัติจริง (เช็คหลัก)", "📈 Backtest & ตรวจสอบระบบ"])

    with t1:
        for i in range(0, len(positions), 2):
            cols = st.columns(2)
            with cols[0]: display_card(positions[i], final[positions[i]], True)
            if i + 1 < len(positions):
                with cols[1]: display_card(positions[i + 1], final[positions[i + 1]], True)

    with t2:
        for i in range(0, len(positions), 2):
            cols = st.columns(2)
            with cols[0]: display_card(positions[i], final[positions[i]], False)
            if i + 1 < len(positions):
                with cols[1]: display_card(positions[i + 1], final[positions[i + 1]], False)

    with t3:
        st.markdown("### 📜 ผลจริง 10 งวดล่าสุด (ตรวจสอบความถูกต้องของการเรียงหลัก)")
        history_cols = ["Date"] + positions
        history = feat.iloc[:-1].tail(10)[history_cols].copy().sort_values("Date", ascending=False)
        history["Date"] = history["Date"].dt.strftime("%d/%m/%Y")
        rename = {pos: POSITION_LABELS[pos] for pos in positions}
        rename["Date"] = "วันที่"
        history = history.rename(columns=rename)
        for col in history.columns:
            if col != "วันที่": history[col] = history[col].astype(int).astype(str)
        st.dataframe(history, use_container_width=True, hide_index=True)

    with t4:
        st.info("💡 หากโมเดลใดมีผลการทายหลุดติดต่อกัน 2 ครั้ง ในส่วนนี้จะสั่งการให้ระบบเปิด Fallback Mode อัตโนมัติ")
        for pos in positions:
            bt_df = final[pos]["backtest"]
            if bt_df is None or bt_df.empty: continue
            
            hot_rate = (bt_df["ผลเด่น"] == "✅ เข้า").sum() / len(bt_df)
            dead_rate = (bt_df["ผลดับ"] == "✅ ผ่าน").sum() / len(bt_df)
            
            title = f"📊 {POSITION_LABELS[pos]} | Win Rate {hot_rate*100:.0f}%"
            if final[pos]["is_fallback"]: title += " ⚠️ (ใช้งาน Fallback Mode แล้ว)"
            
            with st.expander(title, expanded=False):
                st.dataframe(bt_df.sort_values("วันที่", ascending=False), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
