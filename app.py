# ============================================================
# LOTTO AI - ADAPTIVE EQUATION LOCK V2.0 (MONGODB EDITION)
# ============================================================
# FEATURES
# 1. ดึงข้อมูลจาก Blogspot พร้อมตัดคำและวันที่ด้วย Regex
# 2. ใช้ระบบ Cloud Database (MongoDB) เพื่อเก็บ State ถาวร
# 3. จัดการ Session State ไม่ให้โหลดข้อมูลซ้ำซ้อน
# 4. แสดงผลสถานะสูตรด้วยระบบสัญญาณไฟ (Traffic Light)
# 5. ระบบ Adaptive Lock เปลี่ยนสูตรอัตโนมัติเมื่อผิด 2 งวดติด
# ============================================================

import os
import re
import json
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# นำเข้า MongoDB
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "LOTTO AI - ADAPTIVE EQUATION LOCK"
HISTORY_FILE = "lotto_adaptive_history.json" # ไว้เป็น Fallback กรณีไม่ได้ต่อ MongoDB
LOOKBACK = 10
REQUEST_TIMEOUT = 20
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/128.0 Safari/537.36"
    )
}

# ============================================================
# LOTTERY URLS
# ============================================================
LOTTO_URLS = {
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

# ============================================================
# PAGE STYLE
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.markdown("""
<style>
.main-title { font-size: 30px; font-weight: 800; margin-bottom: 5px; }
.lock { background:#dff5df; padding:5px 10px; border-radius:8px; font-weight:bold; }
.change { background:#ffe0e0; padding:5px 10px; border-radius:8px; font-weight:bold; }
.good { color:green; font-weight:bold; }
.bad { color:red; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE SETUP (MONGODB + JSON FALLBACK)
# ============================================================
@st.cache_resource
def init_mongo_connection():
    if MONGO_AVAILABLE and "MONGO_URI" in st.secrets:
        return MongoClient(st.secrets["MONGO_URI"])
    return None

def get_mongo_collection():
    client = init_mongo_connection()
    if client:
        return client["lotto_ai_db"]["adaptive_state"]
    return None

def load_state():
    # ลองโหลดจาก MongoDB ก่อน
    collection = get_mongo_collection()
    if collection is not None:
        try:
            state = collection.find_one({"_id": "main_global_state"})
            if state:
                state.pop("_id", None)
                return state
            return {}
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถดึงข้อมูลจาก MongoDB ได้ (ใช้ Local JSON แทน): {e}")
    
    # Fallback กรณีไม่มี MongoDB
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_state(state):
    collection = get_mongo_collection()
    if collection is not None:
        try:
            collection.update_one({"_id": "main_global_state"}, {"$set": state}, upsert=True)
            return
        except Exception as e:
            st.warning(f"⚠️ เกิดข้อผิดพลาดในการบันทึก MongoDB (ใช้ Local JSON แทน): {e}")
            
    # Fallback กรณีไม่มี MongoDB
    tmp_file = HISTORY_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, HISTORY_FILE)

# ============================================================
# DOWNLOAD BLOGSPOT & PARSE DATA
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_page(url):
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def extract_post_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    
    containers = []
    selectors = ["div.post-body", "div.post-body.entry-content", "div.entry-content", "article", "main"]
    
    for selector in selectors:
        found = soup.select(selector)
        for x in found:
            txt = x.get_text("\n", strip=True)
            if len(txt) > 100: containers.append(txt)
            
    if containers:
        containers.sort(key=len, reverse=True)
        return containers[0]
    return soup.get_text("\n", strip=True)

def parse_lottery_text(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    records = []
    
    # Regex แบบควบรวม รองรับทั้ง 6 ตัว และ 3 ตัว (ดึงวันที่มาด้วย)
    pattern = r"\*?\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{3,6})\s*\|\s*(\d{2})"
    
    for idx, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            date_str = match.group(1)
            main_num = match.group(2)
            number2 = match.group(3)
            
            if len(main_num) == 6:
                number6 = main_num
                number3 = main_num[-3:]
            else:
                number6 = None
                number3 = main_num
                
            records.append({
                "source_line": idx,
                "date": date_str,
                "number6": number6,
                "number3": number3,
                "number2": number2,
                "raw": line
            })

    # ⚠️ สำคัญมาก: ต้อง Reverse ข้อมูลให้เรียงจาก เก่า -> ใหม่ สำหรับ Backtest
    records = records[::-1]
    return records

def deduplicate_records(records):
    result = []
    seen = set()
    for r in records:
        key = (r.get("date"), r.get("number6"), r.get("number3"), r.get("number2"))
        if key in seen: continue
        seen.add(key)
        result.append(r)
    return result

def select_last_draws(records):
    records = deduplicate_records(records)
    # ตัดเอาเฉพาะ LOOKBACK งวดล่าสุด
    return records[-LOOKBACK:]

# ============================================================
# CONVERT RESULT TO DIGITS
# ============================================================
def result_to_digits(record):
    if record.get("number6"):
        s = str(record["number6"]).zfill(6)
        return {
            "H1": int(s[0]), "H2": int(s[1]), "H3": int(s[2]), "H4": int(s[3]),
            "H5": int(s[4]), "H6": int(s[5]), "T": int(s[-3]), "O": int(s[-2]),
            "N": int(s[-1]), "T2": int(s[-2]), "O2": int(s[-1]),
        }
    if record.get("number3"):
        s = str(record["number3"]).zfill(3)
        return {"T": int(s[0]), "O": int(s[1]), "N": int(s[2])}
    if record.get("number2"):
        s = str(record["number2"]).zfill(2)
        return {"T2": int(s[0]), "O2": int(s[1])}
    return {}

# ============================================================
# FORMULA ENGINE
# ============================================================
class FormulaEngine:
    def __init__(self):
        self.formulas = [
            # Single lag
            ("L1", lambda d: d[-1]), ("L2", lambda d: d[-2]), ("L3", lambda d: d[-3]),
            ("L4", lambda d: d[-4]), ("L5", lambda d: d[-5]),
            # Sum
            ("L1+L2", lambda d: d[-1] + d[-2]), ("L1+L3", lambda d: d[-1] + d[-3]),
            ("L2+L3", lambda d: d[-2] + d[-3]), ("L3+L4", lambda d: d[-3] + d[-4]),
            # Difference
            ("L1-L2", lambda d: d[-1] - d[-2]), ("L2-L3", lambda d: d[-2] - d[-3]),
            # Multiplication
            ("L1*L2", lambda d: d[-1] * d[-2]), ("L2*L3", lambda d: d[-2] * d[-3]),
            # Weighted
            ("2L1+L2", lambda d: 2*d[-1] + d[-2]), ("L1+2L2", lambda d: d[-1] + 2*d[-2]),
            # Constants
            ("L1+1", lambda d: d[-1] + 1), ("L1+2", lambda d: d[-1] + 2), ("L1+3", lambda d: d[-1] + 3),
            ("L1-1", lambda d: d[-1] - 1), ("L1-2", lambda d: d[-1] - 2),
            # Combinations
            ("L1+L2+L3", lambda d: d[-1] + d[-2] + d[-3]),
            ("L1-L2+L3", lambda d: d[-1] - d[-2] + d[-3]),
        ]

    def predict(self, formula_name, history):
        for name, fn in self.formulas:
            if name == formula_name:
                try: return int(fn(history)) % 10
                except: return 0
        return 0

def backtest_formula(formula_name, values):
    engine = FormulaEngine()
    if len(values) < 6:
        return {"formula": formula_name, "hits": 0, "tests": 0, "rate": 0.0}
    hits, tests = 0, 0
    for i in range(5, len(values)):
        history = values[:i]
        actual = values[i]
        pred = engine.predict(formula_name, history)
        if pred == actual: hits += 1
        tests += 1
    return {"formula": formula_name, "hits": hits, "tests": tests, "rate": hits / tests if tests else 0}

def rank_formulas(values):
    engine = FormulaEngine()
    results = [backtest_formula(name, values) for name, _ in engine.formulas]
    df = pd.DataFrame(results)
    if df.empty: return df
    df["score"] = (df["rate"] * 100) + (df["hits"] * 0.5)
    df = df.sort_values(["score", "rate", "hits"], ascending=False).reset_index(drop=True)
    return df.head(3)

# ============================================================
# LOGIC & PREDICTION
# ============================================================
def detect_positions(records):
    positions = set()
    for record in records:
        positions.update(result_to_digits(record).keys())
    if all(p in positions for p in ["H1","H2","H3","H4","H5","H6"]): return ["H1","H2","H3","H4","H5","H6"]
    if all(p in positions for p in ["T", "O", "N"]): return ["T", "O", "N"]
    if all(p in positions for p in ["T2", "O2"]): return ["T2", "O2"]
    return sorted(list(positions))

def build_position_series(records, position):
    values = []
    for r in records:
        digits = result_to_digits(r)
        if position in digits: values.append(int(digits[position]))
    return values

def create_initial_position_state(values):
    top3 = rank_formulas(values)
    if top3.empty:
        return {"formula": "L1", "top3": [], "miss_streak": 0, "lock": True, "history": []}
    top3_records = top3.to_dict('records')
    return {"formula": top3_records[0]["formula"], "top3": top3_records, "miss_streak": 0, "lock": True, "history": []}

def refresh_position(values, old_state=None):
    top3 = rank_formulas(values)
    if top3.empty: return create_initial_position_state(values)
    top3_records = top3.to_dict('records')
    
    if old_state:
        old_formula = old_state.get("formula")
        candidates = [x["formula"] for x in top3_records]
        if old_formula in candidates and old_state.get("miss_streak", 0) < 2:
            selected = old_formula
        else:
            selected = top3_records[0]["formula"]
    else:
        selected = top3_records[0]["formula"]

    return {
        "formula": selected, "top3": top3_records, "miss_streak": 0, "lock": True,
        "history": old_state.get("history", []) if old_state else []
    }

def add_prediction_history(state, actual, prediction, formula, draw_id, date_str):
    hit = (int(actual) == int(prediction))
    state["miss_streak"] = 0 if hit else int(state.get("miss_streak", 0)) + 1

    history = state.get("history", [])
    if history and history[-1].get("draw") == draw_id and history[-1].get("formula") == formula:
        return state

    history.append({
        "draw": draw_id,
        "date": date_str,
        "actual": int(actual),
        "prediction": int(prediction),
        "formula": formula,
        "hit": bool(hit),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    state["history"] = history[-10:]
    return state

def process_lottery(lottery_name, records, global_state):
    records = select_last_draws(records)
    if len(records) < 6: return {"error": "ข้อมูลน้อยกว่า 6 งวด"}
    
    positions = detect_positions(records)
    if not positions: return {"error": "ไม่พบตำแหน่งเลข"}
    
    lottery_state = global_state.get(lottery_name, {})
    results = {}

    for idx, r in enumerate(records):
        # ใช้วันที่เป็น ID สำหรับอ้างอิงงวด
        r["draw_id"] = r.get("date", f"Draw-{idx}")

    for position in positions:
        values = build_position_series(records, position)
        if len(values) < 6: continue
        
        old_state = lottery_state.get(position)
        if not old_state:
            state = create_initial_position_state(values)
        else:
            state = old_state
            last_draw = records[-1]
            last_draw_id = last_draw["draw_id"]
            last_date = last_draw.get("date", "-")
            
            already_checked = any(x.get("draw") == last_draw_id for x in state.get("history", []))
            
            if not already_checked:
                formula = state.get("formula", "L1")
                history_values = values[:-1]
                if len(history_values) >= 5:
                    prediction = FormulaEngine().predict(formula, history_values)
                    actual = values[-1]
                    state = add_prediction_history(state, actual, prediction, formula, last_draw_id, last_date)
            
            if state.get("miss_streak", 0) >= 2:
                old_formula = state.get("formula")
                new_state = refresh_position(values, state)
                if len(new_state["top3"]) > 1:
                    candidates = [x["formula"] for x in new_state["top3"]]
                    if old_formula in candidates:
                        idx_old = candidates.index(old_formula)
                        if idx_old + 1 < len(candidates):
                            new_state["formula"] = candidates[idx_old + 1]
                new_state["miss_streak"] = 0
                new_state["lock"] = True
                state = new_state

        formula = state.get("formula", "L1")
        prediction = FormulaEngine().predict(formula, values)
        results[position] = {"state": state, "values": values, "prediction": prediction}
        lottery_state[position] = state
        
    global_state[lottery_name] = lottery_state
    return {"positions": positions, "results": results, "records": records}

# ============================================================
# UI RENDER COMPONENTS
# ============================================================
def display_position_history(state):
    history = state.get("history", [])
    if not history:
        st.info("ยังไม่มีประวัติการตรวจผล")
        return
    rows = []
    for x in history[::-1]:
        rows.append({
            "วันที่": x.get("date", "-"),
            "ทาย": x.get("prediction"),
            "ผลจริง": x.get("actual"),
            "สมการ": x.get("formula"),
            "ผล": "✅ ถูก" if x.get("hit") else "❌ ผิด",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# MAIN APP FLOW
# ============================================================
# ระบบ Session State ลดการอ่าน/เขียน DB พร่ำเพรื่อ
if "global_state" not in st.session_state:
    st.session_state.global_state = load_state()

global_state = st.session_state.global_state

st.sidebar.title("⚙️ ตั้งค่าระบบ")
selected_lotto = st.sidebar.selectbox("เลือกหวย", list(LOTTO_URLS.keys()))
st.sidebar.markdown(f"**ข้อมูลย้อนหลัง:** {LOOKBACK} งวด\n\n**กฎเปลี่ยนสูตร:** ผิด 2 งวดติด")

if st.sidebar.button("🔄 ล้าง Cache และโหลดใหม่", use_container_width=True):
    st.cache_data.clear()
    st.session_state.global_state = load_state()
    st.rerun()

st.markdown('<div class="main-title">🤖 LOTTO AI - ADAPTIVE EQUATION LOCK</div>', unsafe_allow_html=True)
url = LOTTO_URLS[selected_lotto]

with st.spinner(f"กำลังดึงข้อมูล {selected_lotto}..."):
    try:
        html = download_page(url)
        text = extract_post_text(html)
        records = parse_lottery_text(text)
    except Exception as e:
        st.error(f"ไม่สามารถดึงข้อมูลได้: {e}")
        st.stop()

records = select_last_draws(records)

col1, col2, col3 = st.columns(3)
with col1: st.metric("งวดที่พบ", len(records))
with col2: st.metric("ย้อนหลังที่ใช้", min(len(records), LOOKBACK))
with col3: st.metric("ตำแหน่ง", len(detect_positions(records)))

with st.expander("📋 ดูข้อมูลที่ดึงมา (ดิบ)"):
    raw_rows = []
    # เรียงให้งวดใหม่สุดอยู่บนเวลาโชว์ตาราง
    for i, r in enumerate(records[::-1], start=1):
        raw_rows.append({
            "วันที่": r.get("date"),
            "เลข 6 ตัว": r.get("number6", "-"),
            "เลข 3 ตัว": r.get("number3", "-"),
            "เลข 2 ตัว": r.get("number2", "-"),
        })
    st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)

if len(records) < 6:
    st.warning("ระบบต้องการอย่างน้อย 6 งวด เพื่อทดสอบสมการ")
    st.stop()

result = process_lottery(selected_lotto, records, global_state)
if "error" in result:
    st.error(result["error"])
    st.stop()

# อัปเดต Global State และบันทึก
st.session_state.global_state = global_state
save_state(global_state)

st.markdown("## 🎯 สรุปสถานะสูตรปัจจุบัน (Traffic Light)")
summary_rows = []
for position in result["positions"]:
    item = result["results"].get(position)
    if not item: continue
    
    state = item["state"]
    miss = state.get("miss_streak", 0)
    
    # ระบบสีแจ้งเตือน
    if miss == 0: status = "🟢 ปลอดภัย (LOCK)"
    elif miss == 1: status = "🟡 เฝ้าระวัง"
    else: status = "🔄 รอเปลี่ยนสูตร"
        
    summary_rows.append({
        "หลัก": position,
        "สมการปัจจุบัน": state.get("formula", "-"),
        "ทำนายงวดหน้า": item["prediction"],
        "ผิดติดกัน": miss,
        "สถานะ": status
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.markdown("## 🔮 ค่าทำนายงวดถัดไป")
pred_cols = st.columns(len(result["positions"]))
for col, position in zip(pred_cols, result["positions"]):
    item = result["results"].get(position)
    if not item: continue
    with col:
        st.metric(position, "-" if item["prediction"] is None else item["prediction"])
        st.caption(f"สูตร: {item['state'].get('formula', '-')}")

st.markdown("## 📚 ประวัติย้อนหลัง (ดูว่าสูตรหลุดหรือยัง)")
for position in result["positions"]:
    item = result["results"].get(position)
    if not item: continue
    state = item["state"]
    miss = state.get("miss_streak", 0)
    
    # ใส่ Emoji บอกสถานะที่ชื่อ Expander เลย
    icon = "🟢" if miss == 0 else "🟡" if miss == 1 else "🔴"
    with st.expander(f"{icon} หลัก {position} • สูตร {state.get('formula')} (ผิด {miss} งวด)"):
        display_position_history(state)
