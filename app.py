import urllib.request
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import timedelta


# =========================================================
# 1. โหลด "สมองลับ"
#    PRO V7 AI-ONLY – Mobile Accuracy Edition
# =========================================================

@st.cache_resource
def load_secret_modules():

    url_dub = (
        "https://raw.githubusercontent.com/"
        "suksanhankutlo-stack/lotto-ai-app-/"
        "refs/heads/main/secret_lotto_v4.py"
    )

    url_v7 = (
        "https://raw.githubusercontent.com/"
        "suksanhankutlo-stack/lotto-ai-app-/"
        "refs/heads/main/secret_lotto_v7_ai.py"
    )

    try:

        urllib.request.urlretrieve(
            url_dub,
            "secret_lotto_v4.py"
        )

        urllib.request.urlretrieve(
            url_v7,
            "secret_lotto_v7_ai.py"
        )

        return True

    except Exception as e:

        st.error(
            "❌ ไม่สามารถดาวน์โหลดโมดูล AI ได้\n\n"
            f"รายละเอียด: {e}"
        )

        return False


# =========================================================
# โหลดโมดูล
# =========================================================

if not load_secret_modules():

    st.error(
        "❌ ไม่สามารถดึงไฟล์ PRO V7 AI-ONLY จาก GitHub ได้"
    )

    st.stop()


# =========================================================
# Reload module ป้องกัน Streamlit ใช้ไฟล์เก่า
# =========================================================

if "secret_lotto_v7_ai" in sys.modules:
    del sys.modules["secret_lotto_v7_ai"]


# =========================================================
# Import ระบบเลขดับ V4
# =========================================================

from secret_lotto_v4 import (
    LotteryScraper as Scraper_Dub,
    OptimizedEliminationSystemV4
)


# =========================================================
# Import ระบบเลขเด่น PRO V7 AI-ONLY
# =========================================================

from secret_lotto_v7_ai import (
    LOTTERY_SOURCES as Sources_Den,
    fetch_and_clean_data as fetch_den,
    PROV7AIOnly
)


# =========================================================
# 2. Helper – เลขดับ
# =========================================================

def get_dead_numbers(probs_array, k=7):

    probs_array = np.asarray(
        probs_array,
        dtype=float
    )

    return [
        (int(idx), float(probs_array[idx]))
        for idx in np.argsort(probs_array)[:k]
    ]


def format_dead_output(dead_list):

    return " - ".join(
        str(num)
        for num, prob in dead_list
    )


# =========================================================
# 3. วิเคราะห์เลขดับ
# =========================================================

def run_analysis_dub(
    target_lotto,
    dow_input_str
):

    day_map = {

        "อัตโนมัติ (คำนวณจากงวดล่าสุด)": None,

        "วันจันทร์": 0,
        "วันอังคาร": 1,
        "วันพุธ": 2,
        "วันพฤหัสบดี": 3,
        "วันศุกร์": 4,
        "วันเสาร์": 5,
        "วันอาทิตย์": 6

    }

    dow_input = day_map[dow_input_str]


    try:

        scraper = Scraper_Dub()

        df = scraper.fetch_data(
            target_lotto
        )

    except Exception as e:

        return (
            "<h3 style='color:red;'>"
            "❌ ระบบเลขดับโหลดข้อมูลไม่ได้<br>"
            f"{e}"
            "</h3>"
        )


    if df is None or df.empty:

        return (
            "<h3 style='color:red;'>"
            "❌ ไม่สามารถดึงข้อมูลเลขดับได้"
            "</h3>"
        )


    try:

        sys_status = OptimizedEliminationSystemV4(
            df,
            "hundred",
            target_lotto
        )

        last_date = df["date"].iloc[-1]


        if dow_input is not None:

            days_ahead = (
                dow_input -
                last_date.dayofweek
            )

            if days_ahead <= 0:
                days_ahead += 7

            target_date = (
                last_date +
                timedelta(days=days_ahead)
            )

            target_dow = dow_input

        else:

            if len(df) <= 1:

                days_ahead = 7

            else:

                days_ahead = (
                    last_date -
                    df["date"].iloc[-2]
                ).days

                days_ahead = max(
                    1,
                    days_ahead
                )

            target_date = (
                last_date +
                timedelta(days=days_ahead)
            )

            target_dow = (
                target_date.dayofweek
            )


        dow_names = [
            "จันทร์",
            "อังคาร",
            "พุธ",
            "พฤหัสบดี",
            "ศุกร์",
            "เสาร์",
            "อาทิตย์"
        ]


        out = (
            "<div class='result-container result-dub'>"
        )

        out += (
            "<div class='res-header'>"
            "🛑 สรุปเลขดับ<br>"
            f"ประจำวัน{dow_names[target_dow]}"
            f"ที่ {target_date.strftime('%d/%m/%Y')}"
            "</div>"
        )

        out += (
            "<div class='res-sub'>"
            f"(สเตตัสระบบ: "
            f"{sys_status.mode_name})"
            "</div>"
        )

        out += (
            "<hr style='border-color:#fca5a5;'>"
        )


        positions = {

            "💯 3 ตัวบน (ร้อย)": "hundred",

            "🔟 3 ตัวบน (สิบ)": "ten",

            "1️⃣ 3 ตัวบน (หน่วย)": "unit",

            "🔽 2 ตัวล่าง (สิบ)": "bot_ten",

            "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit"

        }


        for pos_th, col_en in positions.items():

            system = OptimizedEliminationSystemV4(
                df,
                col_en,
                target_lotto
            )

            results = system.analyze(
                target_dow
            )

            if not results:
                continue


            dead_final = get_dead_numbers(
                results["final"],
                7
            )

            nums_final = (
                format_dead_output(
                    dead_final
                )
            )


            out += (
                f"<div class='res-pos'>"
                f"{pos_th}"
                "</div>"
            )

            out += (
                "<div class='res-num-box'>"
                "🌟 ดับฟันธง:<br>"
                f"<span class='dub-text'>"
                f"{nums_final}"
                "</span>"
                "</div>"
            )


        out += "</div>"

        return out


    except Exception as e:

        return (
            "<h3 style='color:red;'>"
            "❌ ระบบเลขดับเกิดข้อผิดพลาด<br>"
            f"{e}"
            "</h3>"
        )


# =========================================================
# 4. วิเคราะห์เลขเด่น
#    PRO V7 AI-ONLY
# =========================================================

def run_prediction_den(
    selected_lotto,
    dow_input_str
):

    day_map = {

        "อัตโนมัติ (คำนวณจากงวดล่าสุด)": None,

        "วันจันทร์": 0,
        "วันอังคาร": 1,
        "วันพุธ": 2,
        "วันพฤหัสบดี": 3,
        "วันศุกร์": 4,
        "วันเสาร์": 5,
        "วันอาทิตย์": 6

    }

    target_dow = day_map[
        dow_input_str
    ]


    den_map = {

        "หวยไทย":
            "1. หวยไทย",

        "หวยธกส":
            "2. หวยธกส.",

        "หวยออมสิน":
            "3. หวยออมสิน",

        "หวยลาว":
            "4. หวยลาว",

        "หวยฮานอย":
            "5. หวยฮานอย",

        "หวยมาเลย์":
            "6. หวยมาเลย์",

        "หวยหุ้นไทยเย็น":
            "7. หวยหุ้นไทยเย็น",

        "หวยหุ้นนิเคอิบ่าย":
            "8. หวยหุ้นนิเคอิบ่าย",

        "หวยหุ้นฮั่งเส็งบ่าย":
            "9. หวยหุ้นฮั่งเส็งบ่าย",

        "หวยหุ้นจีนบ่าย":
            "10. หวยหุ้นจีนบ่าย"

    }


    lottery_key = den_map[
        selected_lotto
    ]


    if lottery_key not in Sources_Den:

        return (
            "<h3 style='color:red;'>"
            "❌ ไม่พบแหล่งข้อมูลหวยนี้ใน PRO V7"
            "</h3>",
            None
        )


    url = Sources_Den[
        lottery_key
    ]


    # =====================================================
    # ดึงข้อมูลจริง
    # =====================================================

    try:

        df_raw = fetch_den(
            url
        )

    except Exception as e:

        return (
            "<h3 style='color:red;'>"
            "❌ PRO V7 ไม่สามารถดึงข้อมูลจริงได้<br>"
            f"{e}"
            "</h3>",
            None
        )


    if df_raw is None or df_raw.empty:

        return (
            "<h3 style='color:red;'>"
            "❌ ไม่พบข้อมูลจริงจากแหล่งข้อมูล"
            "</h3>",
            None
        )


    # =====================================================
    # สร้าง PRO V7 AI-ONLY Engine
    # =====================================================

    try:

        engine = PROV7AIOnly(
            df_raw,
            lottery_key,
            target_dow=target_dow
        )

        preds, next_date = (
            engine.predict_all()
        )

    except Exception as e:

        return (
            "<h3 style='color:red;'>"
            "❌ PRO V7 AI-ONLY ประมวลผลไม่ได้<br>"
            f"{e}"
            "</h3>",
            None
        )


    dow_names = [

        "จันทร์",
        "อังคาร",
        "พุธ",
        "พฤหัสบดี",
        "ศุกร์",
        "เสาร์",
        "อาทิตย์"

    ]


    labels = {

        "H":
            "💯 หลักร้อย (บน)",

        "T":
            "🔟 หลักสิบ (บน)",

        "O":
            "1️⃣ หลักหน่วย (บน)",

        "T2":
            "🔽 หลักสิบ (ล่าง)",

        "O2":
            "⬇️ หลักหน่วย (ล่าง)"

    }


    # =====================================================
    # HTML RESULT
    # =====================================================

    out = (
        "<div class='result-container result-den'>"
    )


    out += (
        "<div class='res-header'>"
        "🤖 PRO V7 AI-ONLY<br>"
        "🎯 ผลการวิเคราะห์เลขเด่น<br>"
        f"ประจำวัน{dow_names[next_date.dayofweek]}"
        f"ที่ {next_date.strftime('%d-%m-%Y')}"
        "</div>"
    )


    out += (
        "<div class='res-sub'>"
        f"(สเตตัสระบบ: {engine.mode_name})"
        "<br>"
        f"ข้อมูลที่ใช้: {len(df_raw):,} งวด"
        "</div>"
    )


    out += (
        "<hr style='border-color:#86efac;'>"
    )


    # =====================================================
    # แสดง Top 5 จาก AI Ensemble
    # =====================================================

    for pos in [
        "H",
        "T",
        "O",
        "T2",
        "O2"
    ]:

        if pos not in preds:
            continue


        top5 = preds[pos].get(
            "AI_Ensemble",
            []
        )


        nums_final = " - ".join(
            str(num)
            for num, prob in top5
        )


        out += (
            f"<div class='res-pos'>"
            f"{labels[pos]}"
            "</div>"
        )


        out += (
            "<div class='res-num-box'>"
            "🤖 AI TOP 5:<br>"
            f"<span class='den-text'>"
            f"{nums_final}"
            "</span>"
            "</div>"
        )


        # แสดง AI Top 3 เพิ่มเติม

        top3 = preds[pos].get(
            "Top3",
            []
        )


        top3_text = " - ".join(
            str(x)
            for x in top3
        )


        out += (
            "<div class='ai-top3'>"
            f"🔥 AI TOP 3: "
            f"<b>{top3_text}</b>"
            "</div>"
        )


    out += "</div>"


    # =====================================================
    # GRAPH
    # =====================================================

    fig = plt.figure(
        figsize=(10, 6)
    )


    for idx, pos in enumerate(
        ["H", "T", "O", "T2", "O2"]
    ):

        ax = plt.subplot(
            2,
            3,
            idx + 1
        )


        top5 = preds[pos].get(
            "AI_Ensemble",
            []
        )


        numbers = [
            str(item[0])
            for item in top5
        ]


        probabilities = [
            float(item[1]) * 100
            for item in top5
        ]


        ax.bar(
            numbers,
            probabilities
        )


        ax.set_title(
            labels[pos],
            fontsize=10,
            fontweight="bold"
        )


        ax.set_ylabel(
            "%"
        )


        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )


    plt.tight_layout()


    return out, fig


# =========================================================
# 5. Custom CSS
# =========================================================

def inject_custom_css():

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #f8fafc;
        }

        .title-text {
            text-align: center;
            font-size: 3.5rem;
            font-weight: 900;

            background:
                -webkit-linear-gradient(
                    45deg,
                    #2563eb,
                    #db2777
                );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;

            margin-bottom: 0;
            padding-bottom: 10px;
        }

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

            box-shadow:
                0 4px 6px -1px
                rgb(0 0 0 / 0.1);
        }

        .result-container {
            padding: 25px;

            border-radius: 15px;

            margin-top: 15px;
            margin-bottom: 25px;

            box-shadow:
                0 4px 10px
                rgba(0,0,0,0.08);
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

            background-color:
                rgba(255,255,255,0.85);

            padding: 15px 20px;

            border-radius: 10px;

            border:
                2px dashed #cbd5e1;

            text-align: center;
        }

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

        .ai-top3 {
            margin-top: 8px;

            padding: 8px;

            text-align: center;

            font-size: 1.1rem;

            background:
                rgba(255,255,255,0.7);

            border-radius: 8px;
        }

        div[data-testid="stSelectbox"]
        > div > div {

            background-color:
                #eff6ff !important;

            border:
                2px solid #93c5fd !important;

            border-radius: 8px;
        }

        div[data-testid="stSelectbox"]
        > div > div:hover {

            border:
                2px solid #3b82f6 !important;
        }

        div.stButton > button {

            border-radius: 8px;

            font-size: 18px;

            font-weight: bold;

            padding: 0.6rem;

            border: none;

            transition: all 0.3s ease;

            box-shadow:
                0 4px 6px -1px
                rgb(0 0 0 / 0.1);
        }

        div.stButton > button:hover {

            transform:
                translateY(-2px);

            box-shadow:
                0 10px 15px -3px
                rgb(0 0 0 / 0.1);
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 6. MAIN STREAMLIT
# =========================================================

def main():

    st.set_page_config(
        page_title="Lotto AI PRO V7",
        page_icon="🤖",
        layout="wide"
    )


    # =====================================================
    # Session State
    # =====================================================

    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = None

    if "result_text" not in st.session_state:
        st.session_state.result_text = None

    if "result_fig" not in st.session_state:
        st.session_state.result_fig = None


    inject_custom_css()


    # =====================================================
    # Header
    # =====================================================

    st.markdown(
        '<h1 class="title-text">'
        '🤖 PRO V7 AI-ONLY'
        '</h1>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle-text">

        🛑 <b>ระบบวิเคราะห์เลขดับ</b>
        - PRO V4 Adaptive

        <br>

        🤖 <b>ระบบวิเคราะห์เลขเด่น</b>
        - PRO V7 AI-ONLY
        - Mobile Accuracy Edition

        <br>

        🧠 AI Ensemble +
        Walk-Forward Backtest +
        Dynamic Weight

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # เลือกหวย / วัน
    # =====================================================

    c1, c2 = st.columns(2)


    with c1:

        lotto = st.selectbox(
            "🏷️ เลือกประเภทหวย",

            [
                "หวยไทย",
                "หวยธกส",
                "หวยออมสิน",
                "หวยลาว",
                "หวยฮานอย",
                "หวยมาเลย์",
                "หวยหุ้นไทยเย็น",
                "หวยหุ้นนิเคอิบ่าย",
                "หวยหุ้นฮั่งเส็งบ่าย",
                "หวยหุ้นจีนบ่าย"
            ]
        )


    with c2:

        day = st.selectbox(
            "📅 เลือกวัน",

            [
                "อัตโนมัติ (คำนวณจากงวดล่าสุด)",
                "วันจันทร์",
                "วันอังคาร",
                "วันพุธ",
                "วันพฤหัสบดี",
                "วันศุกร์",
                "วันเสาร์",
                "วันอาทิตย์"
            ]
        )


    st.markdown("---")


    # =====================================================
    # Buttons
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        btn_dub = st.button(
            "🛑 เริ่มวิเคราะห์เลขดับ",
            type="primary",
            use_container_width=True
        )


    with col2:

        btn_den = st.button(
            "🤖 เริ่มวิเคราะห์ PRO V7 AI",
            type="primary",
            use_container_width=True
        )


    # =====================================================
    # PROCESS
    # =====================================================

    bottom_area = st.container()


    with bottom_area:


        # =================================================
        # เลขดับ
        # =================================================

        if btn_dub:

            with st.spinner(
                "⏳ กำลังวิเคราะห์เลขดับ..."
            ):

                result_dub = run_analysis_dub(
                    lotto,
                    day
                )


                st.session_state.analysis_mode = (
                    "dub"
                )

                st.session_state.result_text = (
                    result_dub
                )

                st.session_state.result_fig = None


        # =================================================
        # PRO V7 AI
        # =================================================

        elif btn_den:

            with st.spinner(
                "🤖 PRO V7 AI-ONLY กำลังวิเคราะห์..."
            ):

                text, fig = run_prediction_den(
                    lotto,
                    day
                )


                st.session_state.analysis_mode = (
                    "den"
                )

                st.session_state.result_text = (
                    text
                )

                st.session_state.result_fig = (
                    fig
                )


        # =================================================
        # แสดงผล
        # =================================================

        if (
            st.session_state.analysis_mode
            is not None
        ):

            st.write("")


            with st.expander(
                "✨ เปิด/ปิด พื้นที่แสดงผลการวิเคราะห์",
                expanded=True
            ):

                st.markdown(
                    st.session_state.result_text,
                    unsafe_allow_html=True
                )


                if (
                    st.session_state.analysis_mode
                    == "den"
                    and
                    st.session_state.result_fig
                    is not None
                ):

                    st.pyplot(
                        st.session_state.result_fig,
                        clear_figure=True
                    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
