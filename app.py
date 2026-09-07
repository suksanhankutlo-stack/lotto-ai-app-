# ============================================================
# LOTTO AI - ADAPTIVE EQUATION LOCK V3.0 (SUPER AI + MOMENTUM SCORING)
# ============================================================
# FEATURES
# 1. คลังสมการขนาดใหญ่ (รวมสูตรสถิติ, ผลต่าง, ค่าเฉลี่ย, ยกกำลัง)
# 2. ระบบให้คะแนนแบบใหม่ (Streak Bonus) ดันสูตรที่เพิ่งเข้าเป้าให้เป็นอันดับ 1
# 3. จำลองประวัติย้อนหลังอัตโนมัติ (Backfill 10 งวด)
# 4. บันทึก State ลง MongoDB 
# ============================================================

import os
import re
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "LOTTO AI - ADAPTIVE EQUATION LOCK V3"
HISTORY_FILE = "lotto_adaptive_history.json" 
LOOKBACK = 30 # จำนวนงวดที่ดึงมาคำนวณตั้งต้น
HISTORY_SHOW = 10 # จำนวนประวัติย้อนหลังที่แสดงผล
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
# DATABASE SETUP (MONGODB)
# ============================================================
@st.cache_resource
def init_mongo_connection():
    if MONGO_AVAILABLE:
        URI = "mongodb+srv://admin:%40Sscg789@cluster0.1o86fzh.mongodb.net/?appName=Cluster0"
        return MongoClient(URI)
    return None

def get_mongo_collection():
    client = init_mongo_connection()
    if client:
        return client["lotto_ai_db"]["adaptive_state"]
    return None

def load_state():
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
# SUPER FORMULA ENGINE (V3)
# ============================================================
class FormulaEngine:
    def __init__(self):
        self.formulas = [
            # 1. กลุ่มดึงค่าตรงๆ
            ("L1", lambda d: d[-1]), ("L2", lambda d: d[-2]), ("L3", lambda d: d[-3]),
            ("L4", lambda d: d[-4]), ("L5", lambda d: d[-5]),
            
            # 2. กลุ่มบวก/ลบ
            ("L1+L2", lambda d: d[-1] + d[-2]), ("L1+L3", lambda d: d[-1] + d[-3]),
            ("L2+L3", lambda d: d[-2] + d[-3]), ("L3+L4", lambda d: d[-3] + d[-4]),
            ("L1-L2", lambda d: d[-1] - d[-2]), ("L2-L3", lambda d: d[-2] - d[-3]),
            
            # 3. กลุ่มผลต่างสัมบูรณ์ (หาความห่างของตัวเลข)
            ("|L1-L2|", lambda d: abs(d[-1] - d[-2])),
            ("|L1-L3|", lambda d: abs(d[-1] - d[-3])),
            ("|L2-L3|", lambda d: abs(d[-2] - d[-3])),
            
            # 4. กลุ่มคูณ/ถ่วงน้ำหนัก
            ("L1*L2", lambda d: d[-1] * d[-2]), ("L2*L3", lambda d: d[-2] * d[-3]),
            ("2L1+L2", lambda d: 2*d[-1] + d[-2]), ("L1+2L2", lambda d: d[-1] + 2*d[-2]),
            ("3L1", lambda d: 3*d[-1]), ("4L1", lambda d: 4*d[-1]), ("5L1", lambda d: 5*d[-1]),
            
            # 5. กลุ่มบวกค่าคงที่ (เดินหน้า-ถอยหลัง)
            ("L1+1", lambda d: d[-1] + 1), ("L1+2", lambda d: d[-1] + 2), 
            ("L1+3", lambda d: d[-1] + 3), ("L1+4", lambda d: d[-1] + 4),
            ("L1+5", lambda d: d[-1] + 5), ("L1+7", lambda d: d[-1] + 7),
            ("L1-1", lambda d: d[-1] - 1), ("L1-2", lambda d: d[-1] - 2),
            ("L1-3", lambda d: d[-1] - 3),
            
            # 6. กลุ่มสถิติ (Max, Min, Avg)
            ("MAX(L1,L2)", lambda d: max(d[-1], d[-2])),
            ("MIN(L1,L2)", lambda d: min(d[-1], d[-2])),
            ("AVG(L1,L2)", lambda d: (d[-1] + d[-2]) // 2),
            
            # 7. กลุ่ม 3 งวด
            ("L1+L2+L3", lambda d: d[-1] + d[-2] + d[-3]),
            ("L1-L2+L3", lambda d: d[-1] - d[-2] + d[-3]),
            
            # 8. กลุ่มยกกำลัง (Pattern สวิง)
            ("L1^2", lambda d: d[-1] ** 2),
            ("L1^2+L2", lambda d: (d[-1] ** 2) + d[-2]),
        ]

    def predict(self, formula_name, history):
        for name, fn in self.formulas:
            if name == formula_name:
                try: 
                    return int(fn(history)) % 10
                except: 
                    return 0
        return 0

def backtest_formula(formula_name, values):
    engine = FormulaEngine()
    if len(values) < 6:
        return {"formula": formula_name, "hits": 0, "tests": 0, "rate": 0.0, "score": 0.0}
        
    hits, tests, recent_streak = 0, 0, 0
    
    for i in range(5, len(values)):
        history = values[:i]
        actual = values[i]
        pred = engine.predict(formula_name, history)
        
        if pred == actual: 
            hits += 1
            recent_streak += 1  # นับคอมโบต่อเนื่อง
        else:
            recent_streak = 0   # หลุดปุ๊บ รีเซ็ตคอมโบทันที
            
        tests += 1
        
    rate = hits / tests if tests else 0
    
    # 🧠 ระบบคิดคะแนนแบบใหม่ (AI Scoring):
    # - เอาเปอร์เซ็นต์ความแม่นยำ * 100
    # - บวกโบนัส "สูตรกำลังเดิน" เข้าไปอย่างหนัก (คูณ 15)
    score = (rate * 100) + (hits * 1.5) + (recent_streak * 15)
    
    return {
        "formula": formula_name, 
        "hits": hits, 
        "tests": tests, 
        "rate": rate, 
        "score": score
    }

def rank_formulas(values):
    engine = FormulaEngine()
    results = [backtest_formula(name, values) for name, _ in engine.formulas]
    df = pd.DataFrame(results)
    
    if df.empty: return df
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
    series = []
    for r in records:
        digits = result_to_digits(r)
        if position in digits: 
            series.append({
                "value": int(digits[position]),
                "draw_id": r.get("draw_id", "-"),
                "date": r.get("date", "-")
            })
    return series

def refresh_position(values, old_state=None):
    top3 = rank_formulas(values)
    if top3.empty: 
        return {"formula": "L1", "top3": [], "miss_streak": 0, "lock": True, "history": []}
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
    
    # จำกัดประวัติให้เหลือเฉพาะ HISTORY_SHOW (10 งวด)
    state["history"] = history[-HISTORY_SHOW:]
    return state

def process_lottery(lottery_name, records, global_state):
    records = select_last_draws(records)
    if len(records) < 6: return {"error": "ข้อมูลน้อยกว่า 6 งวด"}
    
    positions = detect_positions(records)
    if not positions: return {"error": "ไม่พบตำแหน่งเลข"}
    
    lottery_state = global_state.get(lottery_name, {})
    results = {}

    for idx, r in enumerate(records):
        r["draw_id"] = r.get("date", f"Draw-{idx}")

    for position in positions:
        series = build_position_series(records, position)
        if len(series) < 6: continue
        
        values = [x["value"] for x in series]
        state = lottery_state.get(position)
        if not state:
            state = {"formula": "L1", "top3": [], "miss_streak": 0, "lock": True, "history": []}
            
        # ระบบจำลองย้อนหลังอัตโนมัติ (Auto-Backfill History)
        for i in range(5, len(series)):
            draw_id = series[i]["draw_id"]
            draw_date = series[i]["date"]
            
            already_checked = any(x.get("draw") == draw_id for x in state.get("history", []))
            if already_checked:
                continue
                
            history_values = values[:i]
            
            if not state.get("top3"):
                temp_state = refresh_position(history_values, None)
                state["formula"] = temp_state["formula"]
                state["top3"] = temp_state["top3"]
                
            formula = state.get("formula", "L1")
            prediction = FormulaEngine().predict(formula, history_values)
            actual = values[i]
            
            state = add_prediction_history(state, actual, prediction, formula, draw_id, draw_date)
            
            # กฎล็อกและเปลี่ยนสูตร
            if state.get("miss_streak", 0) >= 2:
                current_values = values[:i+1]
                old_formula = state.get("formula")
                new_state = refresh_position(current_values, state)
                if len(new_state["top3"]) > 1:
                    candidates = [x["formula"] for x in new_state["top3"]]
                    if old_formula in candidates:
                        idx_old = candidates.index(old_formula)
                        if idx_old + 1 < len(candidates):
                            new_state["formula"] = candidates[idx_old + 1]
                    else:
                        new_state["formula"] = candidates[0]
                new_state["miss_streak"] = 0
                new_state["lock"] = True
                state = new_state
                
        # อัปเดตสูตร Top3 งวดปัจจุบันสุดเพื่อโชว์ UI
        state["top3"] = rank_formulas(values).to_dict('records')
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
if "global_state" not in st.session_state:
    st.session_state.global_state = load_state()

global_state = st.session_state.global_state

st.sidebar.title("⚙️ ตั้งค่าระบบ")
selected_lotto = st.sidebar.selectbox("เลือกหวย", list(LOTTO_URLS.keys()))
st.sidebar.markdown(f"**ข้อมูลย้อนหลัง (สำหรับจำลอง):** {LOOKBACK} งวด\n\n**แสดงประวัติ:** {HISTORY_SHOW} งวด\n\n**กฎเปลี่ยนสูตร:** ผิด 2 งวดติด")

# ปุ่มล้างข้อมูล (สำคัญมาก: เมื่ออัปเกรดโค้ดต้องกดปุ่มนี้เพื่อโละสูตรเก่า)
if st.sidebar.button("🔄 ล้าง Cache และบังคับหาชุดสูตรใหม่", use_container_width=True):
    st.cache_data.clear()
    
    # ล้างข้อมูลของหวยที่กำลังเลือกอยู่ทิ้งไปเลย เพื่อบังคับระบบให้คิดใหม่ตั้งแต่ศูนย์
    if selected_lotto in global_state:
        del global_state[selected_lotto]
        save_state(global_state)
        st.session_state.global_state = global_state
        
    st.rerun()

st.markdown('<div class="main-title">🤖 LOTTO AI - ADAPTIVE EQUATION V3</div>', unsafe_allow_html=True)
url = LOTTO_URLS[selected_lotto]

with st.spinner(f"กำลังดึงข้อมูลและคำนวณสมการ {selected_lotto}..."):
    try:
        html = download_page(url)
        text = extract_post_text(html)
        records = parse_lottery_text(text)
    except Exception as e:
        st.error(f"ไม่สามารถดึงข้อมูลได้: {e}")
        st.stop()

records = select_last_draws(records)

col1, col2, col3 = st.columns(3)
with col1: st.metric("งวดที่พบทั้งหมด", len(records))
with col2: st.metric("ประวัติจำลองย้อนหลัง", min(len(records), LOOKBACK))
with col3: st.metric("ตำแหน่ง", len(detect_positions(records)))

with st.expander("📋 ดูข้อมูลที่ดึงมา (ดิบ)"):
    raw_rows = []
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

st.session_state.global_state = global_state
save_state(global_state)

st.markdown("## 🎯 สรุปสถานะสูตรปัจจุบัน (Traffic Light)")
summary_rows = []
for position in result["positions"]:
    item = result["results"].get(position)
    if not item: continue
    
    state = item["state"]
    miss = state.get("miss_streak", 0)
    
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

st.markdown(f"## 📚 ประวัติย้อนหลัง {HISTORY_SHOW} งวด (ผลงานสูตรล่าสุด)")
for position in result["positions"]:
    item = result["results"].get(position)
    if not item: continue
    state = item["state"]
    miss = state.get("miss_streak", 0)
    
    icon = "🟢" if miss == 0 else "🟡" if miss == 1 else "🔴"
    with st.expander(f"{icon} หลัก {position} • สูตร {state.get('formula')} (ผิด {miss} งวด)"):
        display_position_history(state)
