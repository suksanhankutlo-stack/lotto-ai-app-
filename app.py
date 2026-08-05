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

    if df is None or df.empty: return "### ❌ ขัดข้อง: ไม่สามารถดึงข้อมูลได้"

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
    out = f"## 🛑 สรุปเลขดับ ประจำวัน{dow_names[target_dow]}ที่ {target_date.strftime('%d/%m/%Y')}\n"
    out += f"*(สเตตัสระบบ: {sys_status.mode_name})*\n\n---\n"

    positions = {'💯 3 ตัวบน (ร้อย)': 'hundred', '🔟 3 ตัวบน (สิบ)': 'ten', '1️⃣ 3 ตัวบน (หน่วย)': 'unit', '🔽 2 ตัวล่าง (สิบ)': 'bot_ten', '⬇️ 2 ตัวล่าง (หน่วย)': 'bot_unit'}
    
    for pos_th, col_en in positions.items():
        system = OptimizedEliminationSystemV4(df, col_en, target_lotto)
        results = system.analyze(target_dow)
        if not results: continue

        dead_final = get_dead_numbers(results['final'], 7)
        out += f"### {pos_th}\n"
        out += f"> 🌟 **ดับฟันธง:** **`{format_dead_output(dead_final)}`**\n\n"

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

    if df_raw is None or df_raw.empty: return "### ❌ ขัดข้อง: ไม่สามารถดึงข้อมูลได้", None

    engine = EnsembleEngine(df_raw, den_map[selected_lotto], target_dow=target_dow)
    preds, next_date = engine.predict_all()

    dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
    labels = {'H': '💯 หลักร้อย (บน)', 'T': '🔟 หลักสิบ (บน)', 'O': '1️⃣ หลักหน่วย (บน)', 'T2': '🔽 หลักสิบ (ล่าง)', 'O2': '⬇️ หลักหน่วย (ล่าง)'}

    out = f"## 🎯 ผลการวิเคราะห์เลขเด่น ประจำวัน{dow_names[next_date.dayofweek]}ที่ {next_date.strftime('%d-%m-%Y')}\n"
    out += f"*(สเตตัสระบบ: {engine.mode_name})*\n\n---\n"

    for pos in ['H', 'T', 'O', 'T2', 'O2']:
        nums_final = " - ".join([str(num) for num, prob in preds[pos]['Final']])
        out += f"### {labels[pos]}\n"
        out += f"> 🌟 **เด่นฟันธง:** **`{nums_final}`**\n\n"

    fig = plt.figure(figsize=(10, 6))
    colors_list = ['#ef4444', '#f97316', '#22c55e', '#3b82f6', '#8b5cf6']
    for idx, pos in enumerate(['H', 'T', 'O', 'T2', 'O2']):
        ax = plt.subplot(2, 3, idx + 1)
        top_5_items = preds[pos]['Final']
        ax.bar([str(x[0]) for x in top_5_items], [x[1]*100 for x in top_5_items], color=colors_list)
        ax.set_title(labels[pos].split(' ')[1] + ' ' + labels[pos].split(' ')[2], fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    return out, fig

# =========================================================
# 4. หน้าจอ UI (Streamlit)
# =========================================================
def main():
    st.set_page_config(page_title="Lotto AI All-in-One", page_icon="🎯", layout="wide")
    st.title("สูตรคำนวณ AI ")
    
    st.markdown("""
    **ระบบวิเคราะห์เลขดับ (Candidate Elimination) - PRO V4 (Adaptive)** | 
    **ระบบวิเคราะห์เลขเด่น Ultimate Ensemble (Optimized Fast Mode)**
    """)

    lotto = st.selectbox(
        "เลือกประเภทหวย",
        ["หวยไทย","หวยธกส","หวยออมสิน","หวยลาว","หวยฮานอย","หวยมาเลย์","หวยหุ้นไทยเย็น","หวยหุ้นนิเคอิบ่าย","หวยหุ้นฮั่งเส็งบ่าย","หวยหุ้นจีนบ่าย"]
    )

    day = st.selectbox(
        "เลือกวัน",
        ["อัตโนมัติ (คำนวณจากงวดล่าสุด)", "วันจันทร์","วันอังคาร","วันพุธ","วันพฤหัสบดี","วันศุกร์","วันเสาร์","วันอาทิตย์"]
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🛑 วิเคราะห์เลขดับ", use_container_width=True):
            with st.spinner("กำลังประมวลผลเลขดับ..."):
                st.markdown(run_analysis_dub(lotto, day))

    with col2:
        if st.button("🎯 วิเคราะห์เลขเด่น", use_container_width=True):
            with st.spinner("กำลังประมวลผลเลขเด่น..."):
                text, fig = run_prediction_den(lotto, day)
                st.markdown(text)
                if fig:
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
