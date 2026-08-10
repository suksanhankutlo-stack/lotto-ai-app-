import urllib.request
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import timedelta

# =========================================================
# 1. โหลด "สมองลับ" (ใช้ Cache เพื่อไม่ให้ดาวน์โหลดซ้ำ)
# =========================================================
@st.cache_resource
def load_secret_modules():
    url_dub = 'https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app-/refs/heads/main/secret_lotto_v4.py'
    url_den = 'https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app-/refs/heads/main/secret_lotto_den_v4.py'
    try:
        urllib.request.urlretrieve(url_dub, 'secret_lotto_v4.py')
        urllib.request.urlretrieve(url_den, 'secret_lotto_den_v4.py')
        return True
    except Exception as e:
        return False

# โหลดโมดูล
if not load_secret_modules():
    st.error("❌ ไม่สามารถดึงไฟล์ข้อมูลสูตรได้ กรุณาตรวจสอบลิงก์ GitHub")
    st.stop()

from secret_lotto_v4 import LotteryScraper as Scraper_Dub, OptimizedEliminationSystemV4
from secret_lotto_den_v4 import LOTTERY_SOURCES as Sources_Den, fetch_and_clean_data as fetch_den, EnsembleEngine

# =========================================================
# 2. ฟังก์ชันประมวลผลสำหรับ "เลขดับ" (Elimination)
# =========================================================
def get_dead_numbers(probs_array, k=7):
    return [(idx, probs_array[idx]) for idx in np.argsort(probs_array)[:k]]

def format_dead_output(dead_list):
    return " - ".join([str(num) for num, prob in dead_list])

def run_analysis_dub(target_lotto, dow_input_str):
    day_map = {'อัตโนมัติ (คำนวณจากงวดล่าสุด)': None, 'วันจันทร์': 0, 'วันอังคาร': 1, 'วันพุธ': 2, 'วันพฤหัสบดี': 3, 'วันศุกร์': 4, 'วันเสาร์': 5, 'วันอาทิตย์': 6}
    dow_input = day_map[dow_input_str]

    scraper = Scraper_Dub()
    df = scraper.fetch_data(target_lotto)

    if df is None or df.empty: return "<h3 style='color:red;'>❌ ขัดข้อง: ไม่สามารถดึงข้อมูลได้</h3>"

    sys_status = OptimizedEliminationSystemV4(df, 'hundred', target_lotto)
    last_date = df['date'].iloc[-1]

    if dow_input is not None:
        days_ahead = dow_input - last_date.dayofweek
        if days_ahead <= 0: days_ahead += 7
        target_date = last_date + timedelta(days=days_ahead)
        target_dow = dow_input
    else:
        days_ahead = 7 if len(df) <= 1 else (last_date - df['date'].iloc[-2]).days
        target_date = last_date + timedelta(days=days_ahead)
        target_dow = target_date.dayofweek

    dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
    
    # ใช้ HTML ครอบเพื่อขยายขนาด
    out = f"<div class='result-container result-dub'>"
    out += f"<div class='res-header'>🛑 สรุปเลขดับ<br>ประจำวัน{dow_names[target_dow]}ที่ {target_date.strftime('%d/%m/%Y')}</div>"
    out += f"<div class='res-sub'>(สเตตัสระบบ: {sys_status.mode_name})</div><hr style='border-color: #fca5a5;'>"

    positions = {'💯 3 ตัวบน (ร้อย)': 'hundred', '🔟 3 ตัวบน (สิบ)': 'ten', '1️⃣ 3 ตัวบน (หน่วย)': 'unit', '🔽 2 ตัวล่าง (สิบ)': 'bot_ten', '⬇️ 2 ตัวล่าง (หน่วย)': 'bot_unit'}
    
    for pos_th, col_en in positions.items():
        system = OptimizedEliminationSystemV4(df, col_en, target_lotto)
        results = system.analyze(target_dow)
        if not results: continue

        dead_final = get_dead_numbers(results['final'], 7)
        nums_final = format_dead_output(dead_final)
        out += f"<div class='res-pos'>{pos_th}</div>"
        out += f"<div class='res-num-box'>🌟 ดับฟันธง: <br><span class='dub-text'>{nums_final}</span></div>"

    out += "</div>"
    return out

# =========================================================
# 3. ฟังก์ชันประมวลผลสำหรับ "เลขเด่น" (Prediction)
# =========================================================
def run_prediction_den(selected_lotto, dow_input_str):
    day_map = {'อัตโนมัติ (คำนวณจากงวดล่าสุด)': None, 'วันจันทร์': 0, 'วันอังคาร': 1, 'วันพุธ': 2, 'วันพฤหัสบดี': 3, 'วันศุกร์': 4, 'วันเสาร์': 5, 'วันอาทิตย์': 6}
    target_dow = day_map[dow_input_str]

    den_map = {'หวยไทย': '1. หวยไทย', 'หวยธกส': '2. หวยธกส.', 'หวยออมสิน': '3. หวยออมสิน', 'หวยลาว': '4. หวยลาว', 'หวยฮานอย': '5. หวยฮานอย',
               'หวยมาเลย์': '6. หวยมาเลย์', 'หวยหุ้นไทยเย็น': '7. หวยหุ้นไทยเย็น', 'หวยหุ้นนิเคอิบ่าย': '8. หวยหุ้นนิเคอิบ่าย',
               'หวยหุ้นฮั่งเส็งบ่าย': '9. หวยหุ้นฮั่งเส็งบ่าย', 'หวยหุ้นจีนบ่าย': '10. หวยหุ้นจีนบ่าย'}

    url = Sources_Den[den_map[selected_lotto]]
    df_raw = fetch_den(url)

    if df_raw is None or df_raw.empty: return "<h3 style='color:red;'>❌ ขัดข้อง: ไม่สามารถดึงข้อมูลได้</h3>", None

    engine = EnsembleEngine(df_raw, den_map[selected_lotto], target_dow=target_dow)
    preds, next_date = engine.predict_all()

    dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
    labels = {'H': '💯 หลักร้อย (บน)', 'T': '🔟 หลักสิบ (บน)', 'O': '1️⃣ หลักหน่วย (บน)', 'T2': '🔽 หลักสิบ (ล่าง)', 'O2': '⬇️ หลักหน่วย (ล่าง)'}

    # ใช้ HTML ครอบเพื่อขยายขนาด
    out = f"<div class='result-container result-den'>"
    out += f"<div class='res-header'>🎯 ผลการวิเคราะห์เลขเด่น<br>ประจำวัน{dow_names[next_date.dayofweek]}ที่ {next_date.strftime('%d-%m-%Y')}</div>"
    out += f"<div class='res-sub'>(สเตตัสระบบ: {engine.mode_name})</div><hr style='border-color: #86efac;'>"

    for pos in ['H', 'T', 'O', 'T2', 'O2']:
        nums_final = " - ".join([str(num) for num, prob in preds[pos]['Final']])
        out += f"<div class='res-pos'>{labels[pos]}</div>"
        out += f"<div class='res-num-box'>🌟 เด่นฟันธง: <br><span class='den-text'>{nums_final}</span></div>"

    out += "</div>"

    fig = plt.figure(figsize=(10, 6))
    colors_list = ['#ef4444', '#f97316', '#22c55e', '#3b82f6', '#8b5cf6']
    fig.patch.set_facecolor('#f8fafc') 
    
    for idx, pos in enumerate(['H', 'T', 'O', 'T2', 'O2']):
        ax = plt.subplot(2, 3, idx + 1)
        ax.set_facecolor('#ffffff')
        top_5_items = preds[pos]['Final']
        ax.bar([str(x[0]) for x in top_5_items], [x[1]*100 for x in top_5_items], color=colors_list, edgecolor='white', linewidth=1.2)
        ax.set_title(labels[pos].split(' ')[1] + ' ' + labels[pos].split(' ')[2], fontsize=10, fontweight='bold', color='#334155')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
    plt.tight_layout()

    return out, fig

# =========================================================
# 4. ตกแต่ง UI ด้วย Custom CSS
# =========================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* ตกแต่งพื้นหลังแอป */
    .stApp { background-color: #f8fafc; }
    
    /* ตกแต่ง Header แบบ Gradient */
    .title-text {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #2563eb, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    
    /* ตกแต่งคำบรรยาย (Subtitle) */
    .subtitle-text {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: -10px;
        margin-bottom: 30px;
        padding: 15px;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    /* กล่องแสดงผลการวิเคราะห์ใหม่ (ใหญ่ชัดเจน) */
    .result-container {
        padding: 25px;
        border-radius: 15px;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    .result-dub {
        background-color: #fef2f2;
        border-left: 8px solid #ef4444;
        color: #7f1d1d;
    }
    .result-den {
        background-color: #f0fdf4;
        border-left: 8px solid #22c55e;
        color: #14532d;
    }
    .res-header {
        font-size: 2.2rem !important;
        font-weight: 900;
        margin-bottom: 5px;
        line-height: 1.3;
    }
    .res-sub {
        font-size: 1.1rem;
        font-style: italic;
        opacity: 0.8;
        margin-bottom: 15px;
    }
    .res-pos {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 10px;
        color: #334155;
    }
    .res-num-box {
        font-size: 1.3rem;
        font-weight: bold;
        background-color: rgba(255,255,255,0.85);
        padding: 15px 20px;
        border-radius: 10px;
        border: 2px dashed #cbd5e1;
        text-align: center;
    }
    
    /* ขนาดตัวเลขเฉพาะเจาะจง ให้ใหญ่สะใจ */
    .dub-text {
        font-size: 2.6rem;
        font-weight: 900;
        color: #dc2626;
        letter-spacing: 3px;
        display: block;
        margin-top: 5px;
    }
    .den-text {
        font-size: 2.6rem;
        font-weight: 900;
        color: #16a34a;
        letter-spacing: 3px;
        display: block;
        margin-top: 5px;
    }

    div[data-testid="stSelectbox"] > div > div {
        background-color: #eff6ff !important;
        border: 2px solid #93c5fd !important;
        border-radius: 8px;
    }
    div[data-testid="stSelectbox"] > div > div:hover {
        border: 2px solid #3b82f6 !important;
    }
    div.stButton > button {
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        padding: 0.6rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 5. หน้าจอ UI (Streamlit)
# =========================================================
def main():
    st.set_page_config(page_title="Lotto AI All-in-One", page_icon="🎯", layout="wide")
    
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = None
    if 'result_text' not in st.session_state:
        st.session_state.result_text = None
    if 'result_fig' not in st.session_state:
        st.session_state.result_fig = None
    
    inject_custom_css()
    
    st.markdown('<h1 class="title-text">✨ สูตรคำนวณ AI 🤖</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="subtitle-text">
        🛑 <b>ระบบวิเคราะห์เลขดับ (Candidate Elimination)</b> - PRO V4 (Adaptive) <br>
        🎯 <b>ระบบวิเคราะห์เลขเด่น Ultimate Ensemble</b> (Optimized Fast Mode)
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        lotto = st.selectbox(
            "🏷️ เลือกประเภทหวย",
            ["หวยไทย","หวยธกส","หวยออมสิน","หวยลาว","หวยฮานอย","หวยมาเลย์","หวยหุ้นไทยเย็น","หวยหุ้นนิเคอิบ่าย","หวยหุ้นฮั่งเส็งบ่าย","หวยหุ้นจีนบ่าย"]
        )
    with c2:
        day = st.selectbox(
            "📅 เลือกวัน",
            ["อัตโนมัติ (คำนวณจากงวดล่าสุด)", "วันจันทร์","วันอังคาร","วันพุธ","วันพฤหัสบดี","วันศุกร์","วันเสาร์","วันอาทิตย์"]
        )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        btn_dub = st.button("🛑 เริ่มวิเคราะห์เลขดับ", type="primary", use_container_width=True)
    with col2:
        btn_den = st.button("🎯 เริ่มวิเคราะห์เลขเด่น", type="primary", use_container_width=True)

    # =========================================================
    # 6. พื้นที่ประมวลผลและแสดงผล
    # =========================================================
    bottom_area = st.container() 
    
    with bottom_area:
        if btn_dub:
            with st.spinner("⏳ กำลังประมวลผลเลขดับ... กรุณารอสักครู่"):
                result_dub = run_analysis_dub(lotto, day)
                st.session_state.analysis_mode = 'dub'
                st.session_state.result_text = result_dub
                st.session_state.result_fig = None
                
        elif btn_den:
            with st.spinner("⏳ กำลังประมวลผลเลขเด่น... กรุณารอสักครู่"):
                text, fig = run_prediction_den(lotto, day)
                st.session_state.analysis_mode = 'den'
                st.session_state.result_text = text
                st.session_state.result_fig = fig

        if st.session_state.analysis_mode is not None:
            st.write("") 
            with st.expander("✨ เปิด/ปิด พื้นที่แสดงผลการวิเคราะห์", expanded=True):
                # ยกเลิกการใช้ st.error/st.success และใช้ st.markdown ส่ง HTML เพื่อให้ตัวใหญ่คมชัดขึ้น
                st.markdown(st.session_state.result_text, unsafe_allow_html=True)
                
                # แสดงกราฟ (ถ้ามี)
                if st.session_state.analysis_mode == 'den' and st.session_state.result_fig:
                    st.pyplot(st.session_state.result_fig)

if __name__ == "__main__":
    main()
