# ============================================================
# 🚀 LOTTO AI PRO V7 AI-ONLY + CANDIDATE ELIMINATION
# Mobile Accuracy Edition
#
# 🎯 เลขเด่น  : PRO V7 AI-ONLY
# 🛑 เลขดับ   : Candidate Elimination
#
# สำหรับ Streamlit / GitHub / Streamlit Cloud
# ============================================================

import os
import sys
import hashlib
import urllib.request
import importlib.util
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 0. CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI PRO V7",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 1. URL ของสมองระบบ
# ============================================================

URL_V7_AI = (
    "https://raw.githubusercontent.com/"
    "suksanhankutlo-stack/lotto-ai-app-/"
    "refs/heads/main/secret_lotto_v7_ai.py"
)

URL_DUB = (
    "https://raw.githubusercontent.com/"
    "suksanhankutlo-stack/lotto-ai-app-/"
    "refs/heads/main/secret_lotto_v4.py"
)

V7_FILE = "secret_lotto_v7_ai.py"
DUB_FILE = "secret_lotto_v4.py"


# ============================================================
# 2. DOWNLOAD MODULE
# ============================================================

@st.cache_resource(show_spinner=False)
def download_module(url, filename):

    try:
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            urllib.request.urlretrieve(url, filename)

        return filename

    except Exception as e:
        st.error(f"❌ ดาวน์โหลด {filename} ไม่สำเร็จ\n\n{e}")
        return None


# ============================================================
# 3. LOAD PYTHON MODULE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_python_module(filepath, module_name):

    try:

        spec = importlib.util.spec_from_file_location(
            module_name,
            filepath
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"ไม่สามารถสร้าง module spec: {filepath}"
            )

        module = importlib.util.module_from_spec(spec)

        sys.modules[module_name] = module

        spec.loader.exec_module(module)

        return module

    except Exception as e:

        st.error(
            f"❌ โหลดระบบ {module_name} ไม่สำเร็จ\n\n"
            f"{type(e).__name__}: {e}"
        )

        return None


# ============================================================
# 4. LOAD V7 AI + DUB SYSTEM
# ============================================================

v7_path = download_module(URL_V7_AI, V7_FILE)
dub_path = download_module(URL_DUB, DUB_FILE)

if v7_path is None or dub_path is None:
    st.stop()

V7 = load_python_module(
    v7_path,
    "secret_lotto_v7_ai"
)

DUB = load_python_module(
    dub_path,
    "secret_lotto_v4"
)

if V7 is None or DUB is None:
    st.stop()


# ============================================================
# 5. FIND V7 ENGINE
# ============================================================

# รองรับชื่อ Class ที่อาจต่างกันใน secret_lotto_v7_ai.py

V7_ENGINE = None

for class_name in [
    "PROV7AIOnly",
    "PROV7AIOnlyEngine",
    "AIOnlyEngine",
    "EnsembleEngine",
]:

    if hasattr(V7, class_name):

        V7_ENGINE = getattr(V7, class_name)
        break


# ============================================================
# 6. FIND DATA FUNCTIONS
# ============================================================

V7_SOURCES = getattr(
    V7,
    "LOTTERY_SOURCES",
    {}
)

V7_FETCH = getattr(
    V7,
    "fetch_and_clean_data",
    None
)

# ระบบเลขดับ
DUB_SCRAPER = getattr(
    DUB,
    "LotteryScraper",
    None
)

DUB_ENGINE = getattr(
    DUB,
    "OptimizedEliminationSystemV4",
    None
)


# ============================================================
# 7. LOTTERY MAP
# ============================================================

LOTTERY_LIST = [
    "หวยไทย",
    "หวยธกส",
    "หวยออมสิน",
    "หวยลาว",
    "หวยฮานอย",
    "หวยมาเลย์",
    "หวยหุ้นไทยเย็น",
    "หวยหุ้นนิเคอิบ่าย",
    "หวยหุ้นฮั่งเส็งบ่าย",
    "หวยหุ้นจีนบ่าย",
]

LOTTERY_MAP = {
    "หวยไทย": "1. หวยไทย",
    "หวยธกส": "2. หวยธกส.",
    "หวยออมสิน": "3. หวยออมสิน",
    "หวยลาว": "4. หวยลาว",
    "หวยฮานอย": "5. หวยฮานอย",
    "หวยมาเลย์": "6. หวยมาเลย์",
    "หวยหุ้นไทยเย็น": "7. หวยหุ้นไทยเย็น",
    "หวยหุ้นนิเคอิบ่าย": "8. หวยหุ้นนิเคอิบ่าย",
    "หวยหุ้นฮั่งเส็งบ่าย": "9. หวยหุ้นฮั่งเส็งบ่าย",
    "หวยหุ้นจีนบ่าย": "10. หวยหุ้นจีนบ่าย",
}

DAY_MAP = {
    "อัตโนมัติ (คำนวณจากงวดล่าสุด)": None,
    "วันจันทร์": 0,
    "วันอังคาร": 1,
    "วันพุธ": 2,
    "วันพฤหัสบดี": 3,
    "วันศุกร์": 4,
    "วันเสาร์": 5,
    "วันอาทิตย์": 6,
}

DAY_NAMES = [
    "จันทร์",
    "อังคาร",
    "พุธ",
    "พฤหัสบดี",
    "ศุกร์",
    "เสาร์",
    "อาทิตย์",
]


# ============================================================
# 8. COMMON HELPERS
# ============================================================

def safe_probability_array(values):
    """
    ทำ probability ให้เป็น numpy array 10 ค่า
    """

    arr = np.asarray(values, dtype=float).flatten()

    if len(arr) == 10:
        pass

    elif len(arr) > 10:
        arr = arr[:10]

    else:
        temp = np.zeros(10)
        temp[:len(arr)] = arr
        arr = temp

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    total = arr.sum()

    if total <= 0:
        return np.ones(10) / 10.0

    return arr / total


def top_numbers(probabilities, n=5):

    probs = safe_probability_array(probabilities)

    idx = np.argsort(probs)[::-1][:n]

    return [
        (int(i), float(probs[i]))
        for i in idx
    ]


def dead_numbers(probabilities, n=7):

    probs = safe_probability_array(probabilities)

    idx = np.argsort(probs)[:n]

    return [
        (int(i), float(probs[i]))
        for i in idx
    ]


def format_numbers(items):

    return " - ".join(
        str(number)
        for number, _ in items
    )


def get_target_date(last_date, target_dow):

    last_date = pd.to_datetime(last_date)

    if target_dow is None:

        return last_date + timedelta(days=7)

    days_ahead = (
        target_dow -
        last_date.dayofweek
    )

    if days_ahead <= 0:
        days_ahead += 7

    return last_date + timedelta(
        days=days_ahead
    )


# ============================================================
# 9. LOAD V7 DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_v7_data(lottery_key):

    if V7_FETCH is None:
        raise RuntimeError(
            "ไม่พบ fetch_and_clean_data ใน V7"
        )

    url = V7_SOURCES.get(lottery_key)

    if not url:
        raise RuntimeError(
            f"ไม่พบ URL ของ {lottery_key}"
        )

    df = V7_FETCH(url)

    if df is None or df.empty:
        raise RuntimeError(
            "ไม่พบข้อมูลหวย"
        )

    return df.copy()


# ============================================================
# 10. LOAD DUB DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_dub_data(lottery_name):

    if DUB_SCRAPER is None:
        raise RuntimeError(
            "ไม่พบ LotteryScraper ใน V4"
        )

    scraper = DUB_SCRAPER()

    df = scraper.fetch_data(
        lottery_name
    )

    if df is None or df.empty:
        raise RuntimeError(
            "ไม่สามารถดึงข้อมูลเลขดับได้"
        )

    return df.copy()


# ============================================================
# 11. FIND V7 PREDICT METHOD
# ============================================================

def run_v7_engine(
    df,
    lottery_key,
    target_dow
):

    if V7_ENGINE is None:

        raise RuntimeError(
            "ไม่พบ AI Engine ใน secret_lotto_v7_ai.py\n"
            "ตรวจสอบชื่อ Class ของ PRO V7"
        )

    # --------------------------------------------------------
    # พยายามสร้าง Engine ตามรูปแบบต่าง ๆ
    # --------------------------------------------------------

    engine = None

    constructor_errors = []

    constructors = [
        lambda: V7_ENGINE(
            df,
            lottery_key,
            target_dow=target_dow
        ),
        lambda: V7_ENGINE(
            df,
            lottery_key,
            target_dow
        ),
        lambda: V7_ENGINE(
            df,
            lottery_key
        ),
        lambda: V7_ENGINE(df),
    ]

    for constructor in constructors:

        try:
            engine = constructor()
            break

        except Exception as e:
            constructor_errors.append(
                str(e)
            )

    if engine is None:

        raise RuntimeError(
            "สร้าง V7 AI Engine ไม่สำเร็จ\n\n"
            + "\n".join(constructor_errors[-3:])
        )

    # --------------------------------------------------------
    # หา predict method
    # --------------------------------------------------------

    prediction = None

    for method_name in [
        "predict_all",
        "predict",
        "analyze",
        "run",
    ]:

        if hasattr(engine, method_name):

            method = getattr(
                engine,
                method_name
            )

            try:

                if method_name == "predict_all":
                    prediction = method()

                elif method_name == "predict":
                    prediction = method()

                elif method_name == "analyze":
                    prediction = method()

                else:
                    prediction = method()

                break

            except TypeError:

                try:
                    prediction = method(
                        target_dow
                    )
                    break

                except Exception:
                    continue

    if prediction is None:

        raise RuntimeError(
            "ไม่พบฟังก์ชัน Prediction ของ PRO V7 AI"
        )

    # --------------------------------------------------------
    # รองรับทั้ง
    #
    # predictions
    #
    # หรือ
    #
    # predictions, next_date
    # --------------------------------------------------------

    if isinstance(prediction, tuple):

        predictions = prediction[0]

        if len(prediction) >= 2:
            next_date = prediction[1]
        else:
            next_date = None

    else:

        predictions = prediction
        next_date = None

    return engine, predictions, next_date


# ============================================================
# 12. NORMALIZE V7 RESULTS
# ============================================================

def extract_final_probs(position_result):

    if position_result is None:
        return np.ones(10) / 10

    # กรณีเป็น dict
    if isinstance(position_result, dict):

        for key in [
            "Final",
            "final",
            "AI",
            "ai",
            "Probs",
            "probs",
            "probabilities",
        ]:

            if key in position_result:

                value = position_result[key]

                # ถ้าเป็น list ของ (digit, prob)
                if (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], (tuple, list))
                ):

                    probs = np.zeros(10)

                    for item in value:

                        if len(item) >= 2:

                            digit = int(item[0])
                            prob = float(item[1])

                            if 0 <= digit <= 9:
                                probs[digit] = prob

                    return safe_probability_array(
                        probs
                    )

                return safe_probability_array(
                    value
                )

    # กรณีเป็น array
    try:

        return safe_probability_array(
            position_result
        )

    except Exception:

        return np.ones(10) / 10


# ============================================================
# 13. FORMAT V7 POSITIONS
# ============================================================

POSITION_LABELS = {
    "H": "💯 หลักร้อย (บน)",
    "T": "🔟 หลักสิบ (บน)",
    "O": "1️⃣ หลักหน่วย (บน)",
    "T2": "🔽 หลักสิบ (ล่าง)",
    "O2": "⬇️ หลักหน่วย (ล่าง)",
}


def get_v7_position(
    predictions,
    position
):

    if not isinstance(
        predictions,
        dict
    ):
        return np.ones(10) / 10

    value = predictions.get(
        position
    )

    if value is None:

        # รองรับชื่ออื่น
        aliases = {
            "H": ["hundred", "top_hundred"],
            "T": ["ten", "top_ten"],
            "O": ["unit", "top_unit"],
            "T2": ["bot_ten", "bottom_ten"],
            "O2": ["bot_unit", "bottom_unit"],
        }

        for alias in aliases.get(
            position,
            []
        ):

            if alias in predictions:

                value = predictions[
                    alias
                ]

                break

    return extract_final_probs(
        value
    )


# ============================================================
# 14. PRO V7 AI-ONLY ANALYSIS
# ============================================================

def run_prediction_v7(
    selected_lotto,
    day_input
):

    lottery_key = LOTTERY_MAP[
        selected_lotto
    ]

    target_dow = DAY_MAP[
        day_input
    ]

    df = load_v7_data(
        lottery_key
    )

    engine, predictions, next_date = (
        run_v7_engine(
            df,
            lottery_key,
            target_dow
        )
    )

    if next_date is None:

        if "Date" in df.columns:

            next_date = get_target_date(
                df["Date"].iloc[-1],
                target_dow
            )

        else:

            next_date = pd.Timestamp.now()

    result = {}

    for position in POSITION_LABELS:

        probs = get_v7_position(
            predictions,
            position
        )

        result[position] = {
            "probs": probs,
            "top5": top_numbers(
                probs,
                5
            )
        }

    return (
        result,
        next_date,
        engine,
        len(df)
    )


# ============================================================
# 15. CANDIDATE ELIMINATION
# ============================================================

def run_candidate_elimination(
    selected_lotto,
    day_input
):

    target_dow = DAY_MAP[
        day_input
    ]

    df = load_dub_data(
        selected_lotto
    )

    if DUB_ENGINE is None:
        raise RuntimeError(
            "ไม่พบ OptimizedEliminationSystemV4"
        )

    # --------------------------------------------------------
    # คำนวณวันเป้าหมาย
    # --------------------------------------------------------

    last_date = pd.to_datetime(
        df["date"].iloc[-1]
    )

    target_date = get_target_date(
        last_date,
        target_dow
    )

    actual_dow = target_date.dayofweek

    # --------------------------------------------------------
    # สถานะระบบ
    # --------------------------------------------------------

    try:

        status_engine = DUB_ENGINE(
            df,
            "hundred",
            selected_lotto
        )

        mode_name = getattr(
            status_engine,
            "mode_name",
            "Candidate Elimination"
        )

    except Exception:

        mode_name = (
            "Candidate Elimination"
        )

    # --------------------------------------------------------
    # ตำแหน่ง
    # --------------------------------------------------------

    positions = {
        "💯 3 ตัวบน (ร้อย)": "hundred",
        "🔟 3 ตัวบน (สิบ)": "ten",
        "1️⃣ 3 ตัวบน (หน่วย)": "unit",
        "🔽 2 ตัวล่าง (สิบ)": "bot_ten",
        "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit",
    }

    results = {}

    for label, column in positions.items():

        try:

            system = DUB_ENGINE(
                df,
                column,
                selected_lotto
            )

            output = system.analyze(
                actual_dow
            )

            if not output:
                continue

            final_probs = output.get(
                "final"
            )

            if final_probs is None:

                final_probs = output.get(
                    "Final"
                )

            if final_probs is None:
                continue

            final_probs = safe_probability_array(
                final_probs
            )

            results[label] = {
                "dead": dead_numbers(
                    final_probs,
                    7
                ),
                "probs": final_probs,
            }

        except Exception:
            continue

    return (
        results,
        target_date,
        mode_name,
        len(df)
    )


# ============================================================
# 16. CSS
# ============================================================

def inject_css():

    st.markdown(
        """
        <style>

        .stApp {
            background:
                linear-gradient(
                    180deg,
                    #f8fafc 0%,
                    #eef2ff 100%
                );
        }

        .main-title {
            text-align:center;
            font-size:2.8rem;
            font-weight:900;
            margin-bottom:5px;
            background:
                linear-gradient(
                    90deg,
                    #2563eb,
                    #7c3aed,
                    #db2777
                );
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
        }

        .sub-title {
            text-align:center;
            color:#475569;
            font-size:1rem;
            margin-bottom:20px;
        }

        .panel {
            padding:20px;
            border-radius:18px;
            margin-top:15px;
            margin-bottom:20px;
            box-shadow:
                0 5px 15px
                rgba(15,23,42,0.08);
        }

        .panel-ai {
            background:#f0fdf4;
            border-left:8px solid #22c55e;
        }

        .panel-dub {
            background:#fef2f2;
            border-left:8px solid #ef4444;
        }

        .result-header {
            font-size:1.9rem;
            font-weight:900;
            margin-bottom:5px;
        }

        .result-sub {
            font-size:0.95rem;
            color:#64748b;
            margin-bottom:15px;
        }

        .position {
            font-size:1.35rem;
            font-weight:800;
            margin-top:18px;
            margin-bottom:7px;
        }

        .number-box {
            background:white;
            border-radius:12px;
            padding:13px;
            text-align:center;
            border:2px dashed #cbd5e1;
        }

        .ai-number {
            font-size:2.2rem;
            font-weight:900;
            color:#16a34a;
            letter-spacing:3px;
        }

        .dead-number {
            font-size:2.2rem;
            font-weight:900;
            color:#dc2626;
            letter-spacing:3px;
        }

        .badge {
            display:inline-block;
            padding:5px 10px;
            border-radius:20px;
            font-size:0.85rem;
            font-weight:700;
            margin-top:5px;
        }

        .badge-ai {
            background:#dcfce7;
            color:#166534;
        }

        .badge-dub {
            background:#fee2e2;
            color:#991b1b;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 17. RENDER AI RESULT
# ============================================================

def render_ai_result(
    result,
    next_date,
    engine,
    data_count
):

    engine_name = engine.__class__.__name__

    html = f"""
    <div class="panel panel-ai">

        <div class="result-header">
            🎯 PRO V7 AI-ONLY
        </div>

        <div class="result-sub">
            วันที่เป้าหมาย:
            {next_date.strftime("%d/%m/%Y")}
            |
            ข้อมูล:
            {data_count} งวด
            |
            Engine:
            {engine_name}
        </div>

        <span class="badge badge-ai">
            🤖 AI-ONLY
        </span>
    """

    for position, label in POSITION_LABELS.items():

        items = result[
            position
        ]["top5"]

        numbers = format_numbers(
            items
        )

        html += f"""
        <div class="position">
            {label}
        </div>

        <div class="number-box">

            🌟 <b>เด่น AI ฟันธง</b><br>

            <span class="ai-number">
                {numbers}
            </span>

        </div>
        """

    html += "</div>"

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 18. RENDER DUB RESULT
# ============================================================

def render_dub_result(
    results,
    target_date,
    mode_name,
    data_count
):

    html = f"""
    <div class="panel panel-dub">

        <div class="result-header">
            🛑 PRO V7 Candidate Elimination
        </div>

        <div class="result-sub">
            วันที่เป้าหมาย:
            {target_date.strftime("%d/%m/%Y")}
            |
            ข้อมูล:
            {data_count} งวด
            |
            {mode_name}
        </div>

        <span class="badge badge-dub">
            🛑 เลขดับ 7 ตัว
        </span>
    """

    for label, data in results.items():

        dead = data["dead"]

        numbers = format_numbers(
            dead
        )

        html += f"""
        <div class="position">
            {label}
        </div>

        <div class="number-box">

            🛑 <b>ดับฟันธง 7 ตัว</b><br>

            <span class="dead-number">
                {numbers}
            </span>

        </div>
        """

    html += "</div>"

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 19. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(
        """
        <div class="main-title">
            ✨ LOTTO AI PRO V7
        </div>

        <div class="sub-title">
            🤖 AI-ONLY Mobile Accuracy Edition
            &nbsp; | &nbsp;
            🛑 Candidate Elimination
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Selectors
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        lotto = st.selectbox(
            "🏷️ เลือกประเภทหวย",
            LOTTERY_LIST
        )

    with c2:

        day = st.selectbox(
            "📅 วันเป้าหมาย",
            list(DAY_MAP.keys())
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        btn_ai = st.button(
            "🎯 วิเคราะห์เลขเด่น PRO V7 AI",
            type="primary",
            use_container_width=True
        )

    with c2:

        btn_dub = st.button(
            "🛑 วิเคราะห์เลขดับ 7 ตัว",
            type="secondary",
            use_container_width=True
        )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    if btn_ai:

        with st.spinner(
            "🤖 PRO V7 AI กำลังวิเคราะห์..."
        ):

            try:

                result, next_date, engine, count = (
                    run_prediction_v7(
                        lotto,
                        day
                    )
                )

                st.session_state[
                    "last_mode"
                ] = "ai"

                st.session_state[
                    "ai_result"
                ] = (
                    result,
                    next_date,
                    engine,
                    count
                )

            except Exception as e:

                st.error(
                    "❌ PRO V7 AI วิเคราะห์ไม่สำเร็จ\n\n"
                    f"{type(e).__name__}: {e}"
                )

    # --------------------------------------------------------
    # DUB
    # --------------------------------------------------------

    if btn_dub:

        with st.spinner(
            "🛑 Candidate Elimination กำลังคำนวณ..."
        ):

            try:

                result, target_date, mode, count = (
                    run_candidate_elimination(
                        lotto,
                        day
                    )
                )

                if not result:

                    st.warning(
                        "⚠️ ระบบไม่สามารถสร้างผลเลขดับได้"
                    )

                else:

                    st.session_state[
                        "last_mode"
                    ] = "dub"

                    st.session_state[
                        "dub_result"
                    ] = (
                        result,
                        target_date,
                        mode,
                        count
                    )

            except Exception as e:

                st.error(
                    "❌ ระบบเลขดับทำงานไม่สำเร็จ\n\n"
                    f"{type(e).__name__}: {e}"
                )

    # --------------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------------

    mode = st.session_state.get(
        "last_mode"
    )

    if mode == "ai":

        data = st.session_state.get(
            "ai_result"
        )

        if data:

            render_ai_result(
                *data
            )

    elif mode == "dub":

        data = st.session_state.get(
            "dub_result"
        )

        if data:

            render_dub_result(
                *data
            )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#94a3b8;
            font-size:0.8rem;
            margin-top:30px;
        ">
            LOTTO AI PRO V7
            • AI-ONLY + Candidate Elimination
            • Mobile Edition
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 20. START
# ============================================================

if __name__ == "__main__":
    main()
