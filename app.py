# ============================================================
# 🤖 LOTTO AI PRO V8.5 FAST ADAPTIVE
# ============================================================
# SPEED:
#   • Hot + Dead ใช้โมเดลชุดเดียว
#   • Feature Selection ครั้งเดียว / split
#   • Backtest lightweight
#   • จำกัดจำนวนข้อมูล train แบบ adaptive
#
# ACCURACY / STABILITY:
#   • Recent Weighted Training
#   • Adaptive Feature Selection
#   • ExtraTrees + HistGradientBoosting
#   • Recent Backtest Model Weighting
#   • Consensus Hot/Dead
#
# IMPORTANT:
#   ระบบนี้เป็น statistical experiment ไม่สามารถรับประกัน
#   การทำนายผลหวยแบบสุ่มได้
# ============================================================

import re
import warnings
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
    page_title="Lotto AI V8.5 Fast Adaptive",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def inject_css():

    st.markdown("""
    <style>

    .stApp {
        background: #f4f6f9;
        font-family: 'Kanit', sans-serif;
    }

    .main-title {
        text-align:center;
        font-size:2.2rem;
        font-weight:900;
        color:#1e293b;
    }

    .subtitle {
        text-align:center;
        color:#64748b;
        font-size:.9rem;
        margin-bottom:25px;
    }

    .status-card {
        background:linear-gradient(135deg,#eff6ff,#dbeafe);
        border-radius:12px;
        padding:15px;
        text-align:center;
        color:#1e40af;
        font-weight:600;
        margin-bottom:20px;
    }

    .hot-card {
        background:white;
        border-left:8px solid #10b981;
        border-radius:12px;
        padding:20px;
        margin:10px 0;
        box-shadow:0 4px 10px rgba(0,0,0,.05);
    }

    .dead-card {
        background:white;
        border-left:8px solid #ef4444;
        border-radius:12px;
        padding:20px;
        margin:10px 0;
        box-shadow:0 4px 10px rgba(0,0,0,.05);
    }

    .position-title {
        font-size:1.2rem;
        font-weight:800;
        color:#334155;
        margin-bottom:10px;
        border-bottom:2px solid #f1f5f9;
        padding-bottom:5px;
    }

    .hot-number {
        font-size:2.5rem;
        font-weight:900;
        letter-spacing:4px;
        text-align:center;
        color:#10b981;
    }

    .dead-number {
        font-size:2.5rem;
        font-weight:900;
        letter-spacing:4px;
        text-align:center;
        color:#ef4444;
        text-decoration:line-through;
        text-decoration-color:rgba(239,68,68,.4);
    }

    .prob-text {
        text-align:center;
        color:#475569;
        font-size:.95rem;
        font-weight:600;
        margin-top:10px;
        padding:10px;
        background:#f8fafc;
        border-radius:8px;
    }

    .confidence {
        text-align:center;
        font-size:.85rem;
        font-weight:600;
        margin-top:10px;
        color:#64748b;
    }

    div.stButton > button {
        width:100%;
        min-height:50px;
        border-radius:10px;
        font-size:1.1rem;
        font-weight:800;
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 2. CONSTANTS
# ============================================================

LOTTERY_SOURCES = {

    "หวยไทย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",

    "หวยธกส":
        "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",

    "หวยออมสิน":
        "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",

    "หวยลาว":
        "https://suksan18190.blogspot.com/2026/07/blog-post.html",

    "หวยฮานอย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",

    "หวยมาเลย์":
        "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",

    "หวยหุ้นไทยเย็น":
        "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",

    "หวยหุ้นนิเคอิบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",

    "หวยหุ้นฮั่งเส็งบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",

    "หวยหุ้นจีนบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}


DOW_NAMES = [
    "จันทร์",
    "อังคาร",
    "พุธ",
    "พฤหัสบดี",
    "ศุกร์",
    "เสาร์",
    "อาทิตย์"
]


MODEL_NAMES = [
    "ExtraTrees",
    "HistGradientBoosting"
]


THAI_POSITIONS = [
    "H1", "H2", "H3",
    "H4", "H5", "H6",
    "T2", "O2"
]


NORMAL_POSITIONS = [
    "H", "T", "O",
    "T2", "O2"
]


POSITION_LABELS = {

    "H1": "หลักแสน",
    "H2": "หลักหมื่น",
    "H3": "หลักพัน",
    "H4": "หลักร้อยบน",
    "H5": "หลักสิบบน",
    "H6": "หลักหน่วยบน",

    "H": "หลักร้อยบน",
    "T": "หลักสิบบน",
    "O": "หลักหน่วยบน",

    "T2": "หลักสิบล่าง",
    "O2": "หลักหน่วยล่าง"
}


THAI_MONTHS = {

    "มกราคม": 1,
    "กุมภาพันธ์": 2,
    "มีนาคม": 3,
    "เมษายน": 4,
    "พฤษภาคม": 5,
    "มิถุนายน": 6,
    "กรกฎาคม": 7,
    "สิงหาคม": 8,
    "กันยายน": 9,
    "ตุลาคม": 10,
    "พฤศจิกายน": 11,
    "ธันวาคม": 12,

    "ม.ค.": 1,
    "ก.พ.": 2,
    "มี.ค.": 3,
    "เม.ย.": 4,
    "พ.ค.": 5,
    "มิ.ย.": 6,
    "ก.ค.": 7,
    "ส.ค.": 8,
    "ก.ย.": 9,
    "ต.ค.": 10,
    "พ.ย.": 11,
    "ธ.ค.": 12
}


# ============================================================
# 3. DATE
# ============================================================

def normalize_date(value):

    if not value:
        return None

    text = str(value).strip()

    for name, month in THAI_MONTHS.items():

        match = re.search(
            rf"(\d{{1,2}})\s*{re.escape(name)}\s*(\d{{4}})",
            text
        )

        if match:

            y = int(match.group(2))

            if y >= 2400:
                y -= 543

            try:
                return pd.Timestamp(
                    y,
                    month,
                    int(match.group(1))
                )
            except:
                return None

    match = re.search(
        r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})",
        text
    )

    if match:

        a, b, c = map(int, match.groups())

        y, m, d = (
            (a, b, c)
            if a >= 1000
            else (c, b, a)
        )

        if y < 100:
            y += 2000

        if y >= 2400:
            y -= 543

        try:
            return pd.Timestamp(y, m, d)
        except:
            pass

    return None


# ============================================================
# 4. SCRAPER
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def fetch_lottery_data(url):

    headers = {
        "User-Agent":
            "Mozilla/5.0"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        content = (
            soup.find(
                "div",
                class_=re.compile(
                    r"post-body|entry-content|post-content|content",
                    re.I
                )
            )
            or soup
        )

        rows = []

        # ----------------------------------------------------
        # TABLE MODE
        # ----------------------------------------------------

        for row in content.find_all("tr"):

            text = " ".join(
                c.get_text(
                    " ",
                    strip=True
                )
                for c in row.find_all(
                    ["td", "th"]
                )
            )

            if not text:
                continue

            date = normalize_date(text)

            if not date:
                continue

            six = re.findall(
                r"(?<!\d)\d{6}(?!\d)",
                text
            )

            three = re.findall(
                r"(?<!\d)\d{3}(?!\d)",
                text
            )

            two = re.findall(
                r"(?<!\d)\d{2}(?!\d)",
                text
            )

            if six and two:

                rows.append({
                    "Date": date,
                    "Result_6D": six[0],
                    "Result_3D": six[0][-3:],
                    "Result_2D": two[-1]
                })

            elif three and two:

                rows.append({
                    "Date": date,
                    "Result_6D": None,
                    "Result_3D": three[0],
                    "Result_2D": two[-1]
                })

        # ----------------------------------------------------
        # TEXT MODE
        # ----------------------------------------------------

        if not rows:

            lines = [
                x.strip()
                for x in content.get_text(
                    separator="\n",
                    strip=True
                ).splitlines()
                if x.strip()
            ]

            current_date = None

            for line in lines:

                date = normalize_date(line)

                if date:
                    current_date = date

                if not current_date:
                    continue

                six = re.findall(
                    r"(?<!\d)\d{6}(?!\d)",
                    line
                )

                three = re.findall(
                    r"(?<!\d)\d{3}(?!\d)",
                    line
                )

                two = re.findall(
                    r"(?<!\d)\d{2}(?!\d)",
                    line
                )

                if six and two:

                    rows.append({
                        "Date": current_date,
                        "Result_6D": six[0],
                        "Result_3D": six[0][-3:],
                        "Result_2D": two[-1]
                    })

                elif three and two:

                    rows.append({
                        "Date": current_date,
                        "Result_6D": None,
                        "Result_3D": three[0],
                        "Result_2D": two[-1]
                    })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Result_3D"] = (
            df["Result_3D"]
            .astype(str)
            .str.extract(r"(\d{3})")[0]
            .str.zfill(3)
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(r"(\d{2})")[0]
            .str.zfill(2)
        )

        if "Result_6D" in df.columns:

            df["Result_6D"] = (
                df["Result_6D"]
                .astype(str)
                .str.extract(r"(\d{6})")[0]
            )

        df = (
            df
            .dropna(subset=["Date"])
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception:
        return pd.DataFrame()


def is_thai_6d(df):

    return (
        "Result_6D" in df.columns
        and
        df["Result_6D"].notna().sum() >= 10
    )


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def build_features(df, thai_6d=False):

    w = df.copy()

    # --------------------------------------------------------
    # DIGITS
    # --------------------------------------------------------

    if thai_6d:

        six = (
            w["Result_6D"]
            .fillna("000000")
            .astype(str)
            .str.zfill(6)
        )

        for i in range(6):

            w[f"H{i+1}"] = (
                six.str[i]
                .astype(np.int8)
            )

    else:

        three = (
            w["Result_3D"]
            .astype(str)
            .str.zfill(3)
        )

        w["H"] = (
            three.str[0]
            .astype(np.int8)
        )

        w["T"] = (
            three.str[1]
            .astype(np.int8)
        )

        w["O"] = (
            three.str[2]
            .astype(np.int8)
        )

    two = (
        w["Result_2D"]
        .astype(str)
        .str.zfill(2)
    )

    w["T2"] = (
        two.str[0]
        .astype(np.int8)
    )

    w["O2"] = (
        two.str[1]
        .astype(np.int8)
    )

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    dt = w["Date"].dt

    w["DOW"] = dt.dayofweek.astype(np.int8)
    w["DAY"] = dt.day.astype(np.int8)
    w["MONTH"] = dt.month.astype(np.int8)

    w["DOW_SIN"] = np.sin(
        2 * np.pi * w["DOW"] / 7
    ).astype(np.float32)

    w["DOW_COS"] = np.cos(
        2 * np.pi * w["DOW"] / 7
    ).astype(np.float32)

    w["MONTH_SIN"] = np.sin(
        2 * np.pi * w["MONTH"] / 12
    ).astype(np.float32)

    w["MONTH_COS"] = np.cos(
        2 * np.pi * w["MONTH"] / 12
    ).astype(np.float32)

    positions = (
        THAI_POSITIONS
        if thai_6d
        else NORMAL_POSITIONS
    )

    # --------------------------------------------------------
    # POSITION FEATURES
    # --------------------------------------------------------

    for pos in positions:

        s = w[pos]
        p = s.shift(1)

        # LAGS
        for lag in (1, 2, 3, 5):

            w[f"{pos}_L{lag}"] = (
                s.shift(lag)
            )

        # ROLLING
        for window in (5, 10, 20):

            w[f"{pos}_M{window}"] = (
                p.rolling(
                    window,
                    min_periods=2
                ).mean()
            )

            w[f"{pos}_S{window}"] = (
                p.rolling(
                    window,
                    min_periods=2
                ).std()
            )

        # FREQUENCY
        for window in (10, 20):

            w[f"{pos}_F{window}_0"] = (
                (p == 0)
                .astype(np.float32)
                .rolling(
                    window,
                    min_periods=2
                )
                .mean()
            )

            w[f"{pos}_F{window}_5"] = (
                (p == 5)
                .astype(np.float32)
                .rolling(
                    window,
                    min_periods=2
                )
                .mean()
            )

        # MOMENTUM
        w[f"{pos}_D1"] = (
            s.shift(1) -
            s.shift(2)
        )

        w[f"{pos}_D2"] = (
            s.shift(2) -
            s.shift(3)
        )

        w[f"{pos}_MOMENTUM"] = (
            p -
            s.shift(4)
        )

        w[f"{pos}_DIFF_M5"] = (
            p -
            w[f"{pos}_M5"]
        )

        # RANGE
        w[f"{pos}_VOL20"] = (
            p.rolling(
                20,
                min_periods=2
            ).max()
            -
            p.rolling(
                20,
                min_periods=2
            ).min()
        )

        # STRUCTURE
        w[f"{pos}_ODD"] = (
            p % 2
        )

        w[f"{pos}_HIGH"] = (
            p >= 5
        ).astype(np.float32)

        w[f"{pos}_MOD3"] = (
            p % 3
        )

        w[f"{pos}_PRIME"] = (
            p.isin(
                [2, 3, 5, 7]
            )
        ).astype(np.float32)

        # CYCLIC DIGIT
        w[f"{pos}_SIN"] = np.sin(
            2 * np.pi * p / 10
        ).astype(np.float32)

        w[f"{pos}_COS"] = np.cos(
            2 * np.pi * p / 10
        ).astype(np.float32)

        # EWMA
        w[f"{pos}_EWMA3"] = (
            p.ewm(
                span=3,
                adjust=False
            ).mean()
        )

        w[f"{pos}_EWMA10"] = (
            p.ewm(
                span=10,
                adjust=False
            ).mean()
        )

        w[f"{pos}_MACD"] = (
            w[f"{pos}_EWMA3"]
            -
            w[f"{pos}_EWMA10"]
        )

        # REPEAT
        w[f"{pos}_REPEAT"] = (
            p ==
            s.shift(2)
        ).astype(np.float32)

    # --------------------------------------------------------
    # CROSS POSITION FEATURES
    # --------------------------------------------------------

    base = (
        w[
            ["H1","H2","H3","H4","H5","H6"]
        ].shift(1)
        if thai_6d
        else
        w[
            ["H","T","O"]
        ].shift(1)
    )

    w["PREV_SUM"] = base.sum(axis=1)

    w["PREV_RANGE"] = (
        base.max(axis=1)
        -
        base.min(axis=1)
    )

    w["PREV_MEAN"] = (
        base.mean(axis=1)
    )

    w["PREV_ODD"] = (
        base % 2
    ).sum(axis=1)

    w["PREV_HIGH"] = (
        base >= 5
    ).sum(axis=1)

    w["PREV_UNIQUE"] = (
        base.nunique(axis=1)
    )

    # --------------------------------------------------------
    # REMOVE EXTREMES
    # --------------------------------------------------------

    return (
        w
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(
            np.float32,
            errors="ignore"
        )
    )


# ============================================================
# 6. FEATURE LIST
# ============================================================

def get_features(thai_6d):

    base = [

        "DOW",
        "DAY",
        "MONTH",

        "DOW_SIN",
        "DOW_COS",

        "MONTH_SIN",
        "MONTH_COS",

        "PREV_SUM",
        "PREV_RANGE",
        "PREV_MEAN",
        "PREV_ODD",
        "PREV_HIGH",
        "PREV_UNIQUE"
    ]

    positions = (
        THAI_POSITIONS
        if thai_6d
        else NORMAL_POSITIONS
    )

    for pos in positions:

        base.extend(
            [
                f"{pos}_L{lag}"
                for lag in (1, 2, 3, 5)
            ]
        )

        base.extend(
            [
                f"{pos}_{m}{w}"
                for m in ("M", "S")
                for w in (5, 10, 20)
            ]
        )

        base.extend(
            [
                f"{pos}_F{w}_{d}"
                for w in (10, 20)
                for d in (0, 5)
            ]
        )

        base.extend([
            f"{pos}_D1",
            f"{pos}_D2",
            f"{pos}_MOMENTUM",
            f"{pos}_DIFF_M5",

            f"{pos}_ODD",
            f"{pos}_HIGH",
            f"{pos}_MOD3",
            f"{pos}_PRIME",

            f"{pos}_SIN",
            f"{pos}_COS",

            f"{pos}_EWMA3",
            f"{pos}_EWMA10",
            f"{pos}_MACD",

            f"{pos}_VOL20",
            f"{pos}_REPEAT"
        ])

    return list(
        dict.fromkeys(base)
    )


# ============================================================
# 7. ADAPTIVE CONFIG
# ============================================================

def get_adaptive_config(n):

    if n >= 700:

        return {
            "min_train": 140,
            "train_window": 500,
            "trees": 55,
            "depth": 7,
            "leaf": 3,
            "selected_features": 24,
            "selector_trees": 8,
            "early_stop": True,
            "decay": 0.997
        }

    if n >= 400:

        return {
            "min_train": 110,
            "train_window": 400,
            "trees": 45,
            "depth": 6,
            "leaf": 3,
            "selected_features": 22,
            "selector_trees": 7,
            "early_stop": True,
            "decay": 0.996
        }

    if n >= 200:

        return {
            "min_train": 90,
            "train_window": 300,
            "trees": 38,
            "depth": 5,
            "leaf": 3,
            "selected_features": 19,
            "selector_trees": 6,
            "early_stop": False,
            "decay": 0.994
        }

    return {

        "min_train": 60,
        "train_window": 220,
        "trees": 30,
        "depth": 4,
        "leaf": 3,
        "selected_features": 16,
        "selector_trees": 5,
        "early_stop": False,
        "decay": 0.992
    }


# ============================================================
# 8. MODEL
# ============================================================

def create_model(
    name,
    cfg,
    random_state=42
):

    if name == "ExtraTrees":

        return ExtraTreesClassifier(

            n_estimators=cfg["trees"],

            max_depth=cfg["depth"],

            min_samples_leaf=cfg["leaf"],

            max_features="sqrt",

            class_weight="balanced",

            n_jobs=-1,

            random_state=random_state
        )

    return HistGradientBoostingClassifier(

        max_iter=max(
            30,
            int(cfg["trees"] * 0.75)
        ),

        max_leaf_nodes=15,

        learning_rate=0.035,

        min_samples_leaf=cfg["leaf"],

        l2_regularization=3.5,

        early_stopping=cfg["early_stop"],

        validation_fraction=0.1,

        n_iter_no_change=4,

        random_state=random_state
    )


# ============================================================
# 9. PROBABILITY
# ============================================================

def normalize_probability(p):

    p = np.asarray(
        p,
        dtype=np.float32
    )

    p = np.nan_to_num(
        p,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    p = np.clip(
        p,
        1e-9,
        None
    )

    total = p.sum()

    if total <= 0:

        return np.ones(
            10,
            dtype=np.float32
        ) / 10

    return (
        p / total
    ).astype(np.float32)


# ============================================================
# 10. TRAIN WEIGHTS
# ============================================================

def make_recent_weights(n, decay):

    idx = np.arange(n)

    distance = (
        n - 1 - idx
    )

    weights = (
        decay ** distance
    )

    # normalize
    weights /= weights.mean()

    return weights.astype(
        np.float32
    )


# ============================================================
# 11. PREPARE MATRIX
# ============================================================

def prepare_matrix(
    X_train,
    X_test,
    selected
):

    A = (
        X_train[selected]
        .astype(np.float32)
    )

    B = (
        X_test[selected]
        .astype(np.float32)
    )

    med = A.median()

    A = (
        A.fillna(med)
        .fillna(0.0)
    )

    B = (
        B.fillna(med)
        .fillna(0.0)
    )

    return A, B


# ============================================================
# 12. FEATURE SELECTION
# ============================================================

def select_features_once(
    X,
    y,
    max_features,
    cfg,
    random_state=123
):

    cols = list(X.columns)

    valid = [
        c for c in cols
        if X[c].nunique(
            dropna=False
        ) > 1
    ]

    if len(valid) <= max_features:
        return valid

    Xi = (
        X[valid]
        .fillna(0.0)
        .astype(np.float32)
    )

    selector = ExtraTreesClassifier(

        n_estimators=cfg[
            "selector_trees"
        ],

        max_depth=4,

        min_samples_leaf=4,

        max_features="sqrt",

        n_jobs=-1,

        random_state=random_state
    )

    try:

        selector.fit(
            Xi,
            y
        )

        importance = (
            selector
            .feature_importances_
        )

        order = np.argsort(
            importance
        )[::-1]

        selected = [
            valid[i]
            for i in order[
                :max_features
            ]
        ]

        return selected

    except:

        return valid[
            :max_features
        ]


# ============================================================
# 13. SINGLE ENSEMBLE
# ============================================================

def ensemble_probability(
    X_train,
    y_train,
    X_test,
    cfg,
    selected
):

    A, B = prepare_matrix(
        X_train,
        X_test,
        selected
    )

    sample_weights = (
        make_recent_weights(
            len(A),
            cfg["decay"]
        )
    )

    model_outputs = []

    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    try:

        model_et = create_model(
            "ExtraTrees",
            cfg,
            random_state=42
        )

        model_et.fit(
            A,
            y_train,
            sample_weight=sample_weights
        )

        raw = model_et.predict_proba(B)[0]

        p = np.zeros(
            10,
            dtype=np.float32
        )

        for cls, prob in zip(
            model_et.classes_,
            raw
        ):

            cls = int(cls)

            if 0 <= cls <= 9:
                p[cls] = prob

        model_outputs.append(
            (
                normalize_probability(p),
                0.40
            )
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # HGB
    # --------------------------------------------------------

    try:

        model_hgb = create_model(
            "HistGradientBoosting",
            cfg,
            random_state=52
        )

        model_hgb.fit(
            A,
            y_train,
            sample_weight=sample_weights
        )

        raw = (
            model_hgb
            .predict_proba(B)[0]
        )

        p = np.zeros(
            10,
            dtype=np.float32
        )

        for cls, prob in zip(
            model_hgb.classes_,
            raw
        ):

            cls = int(cls)

            if 0 <= cls <= 9:
                p[cls] = prob

        model_outputs.append(
            (
                normalize_probability(p),
                0.60
            )
        )

    except Exception:
        pass

    if not model_outputs:

        return (
            np.ones(
                10,
                dtype=np.float32
            ) / 10
        )

    result = np.zeros(
        10,
        dtype=np.float32
    )

    total_weight = 0.0

    for p, weight in model_outputs:

        result += (
            p * weight
        )

        total_weight += weight

    return normalize_probability(
        result / max(
            total_weight,
            1e-9
        )
    )


# ============================================================
# 14. HOT + DEAD
# ============================================================
# สำคัญ:
# ไม่เทรน Dead ใหม่
# Dead = เลขที่มี probability ต่ำสุด
# ============================================================

def run_system_pair(
    X_train,
    y_train,
    X_test,
    cfg
):

    selected = select_features_once(
        X_train,
        y_train,
        cfg["selected_features"],
        cfg
    )

    prob = ensemble_probability(
        X_train,
        y_train,
        X_test,
        cfg,
        selected
    )

    order_hot = np.argsort(
        prob
    )[::-1]

    order_dead = np.argsort(
        prob
    )

    hot_top = [
        (
            int(n),
            float(prob[n])
        )
        for n in order_hot[:3]
    ]

    dead_score = normalize_probability(
        1.0 - prob
    )

    dead_top = [
        (
            int(n),
            float(dead_score[n])
        )
        for n in order_dead[:3]
    ]

    confidence = (
        float(prob[order_hot[0]])
        -
        float(prob[order_hot[1]])
    )

    hot_coverage = float(
        prob[order_hot[:3]].sum()
    )

    dead_coverage = float(
        dead_score[
            order_dead[:3]
        ].sum()
    )

    return {

        "probability": prob,

        "hot_results": hot_top,

        "dead_results": dead_top,

        "confidence": confidence,

        "hot_coverage":
            hot_coverage,

        "dead_coverage":
            dead_coverage,

        "selected":
            selected
    }


# ============================================================
# 15. TRAIN WINDOW
# ============================================================

def get_train_slice(
    df_feat,
    target_idx,
    features,
    pos,
    cfg
):

    end = target_idx

    start = max(
        0,
        end - cfg["train_window"]
    )

    X_train = (
        df_feat[
            features
        ]
        .iloc[start:end]
    )

    y_train = (
        df_feat[pos]
        .astype(np.int8)
        .iloc[start:end]
    )

    return X_train, y_train


# ============================================================
# 16. BACKTEST
# ============================================================

def run_backtest_for_pos(
    df_feat,
    pos,
    features,
    cfg,
    steps=10
):

    results = []

    # --------------------------------------------------------
    # Backtest ใช้ model เบาลง
    # --------------------------------------------------------

    bt_cfg = cfg.copy()

    bt_cfg["trees"] = max(
        18,
        cfg["trees"] // 2
    )

    bt_cfg["selected_features"] = max(
        12,
        cfg["selected_features"] - 3
    )

    bt_cfg["selector_trees"] = max(
        4,
        cfg["selector_trees"] - 2
    )

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    for step in range(
        steps,
        0,
        -1
    ):

        target_idx = (
            len(df_feat)
            - 1
            - step
        )

        if target_idx <= 0:
            continue

        X_train, y_train = (
            get_train_slice(
                df_feat,
                target_idx,
                features,
                pos,
                bt_cfg
            )
        )

        if len(X_train) < bt_cfg[
            "min_train"
        ]:
            continue

        X_test = (
            df_feat[
                features
            ]
            .iloc[[target_idx]]
        )

        actual = int(
            df_feat[pos]
            .iloc[target_idx]
        )

        date_val = pd.to_datetime(
            df_feat["Date"]
            .iloc[target_idx]
        ).strftime(
            "%d/%m/%Y"
        )

        # ----------------------------------------------------
        # SINGLE TRAIN
        # ----------------------------------------------------

        pred = run_system_pair(
            X_train,
            y_train,
            X_test,
            bt_cfg
        )

        hot_top3 = [
            n
            for n, p
            in pred["hot_results"]
        ]

        dead_top3 = [
            n
            for n, p
            in pred["dead_results"]
        ]

        hot_win = (
            "✅ เข้า"
            if actual in hot_top3
            else "❌ หลุด"
        )

        dead_win = (
            "✅ ผ่าน"
            if actual not in dead_top3
            else "❌ ตาย"
        )

        # rank ของเลขจริง
        rank = (
            int(
                np.where(
                    np.argsort(
                        pred["probability"]
                    )[::-1]
                    == actual
                )[0][0]
            ) + 1
        )

        results.append({

            "วันที่":
                date_val,

            "ผลจริง":
                actual,

            "อันดับจริง":
                rank,

            "ทายเด่น Top3":
                " - ".join(
                    map(str, hot_top3)
                ),

            "ผลเด่น":
                hot_win,

            "ทายดับ Top3":
                " - ".join(
                    map(str, dead_top3)
                ),

            "ผลดับ":
                dead_win
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# 17. FINAL PREDICTION
# ============================================================

def final_prediction(
    df_feat,
    pos,
    features,
    cfg
):

    # --------------------------------------------------------
    # exclude dummy
    # --------------------------------------------------------

    X = df_feat[
        features
    ]

    y = (
        df_feat[pos]
        .astype(np.int8)
    )

    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    X_test = X.iloc[[-1]]

    # adaptive recent window
    if len(X_train) > cfg[
        "train_window"
    ]:

        X_train = X_train.iloc[
            -cfg["train_window"]:
        ]

        y_train = y_train.iloc[
            -cfg["train_window"]:
        ]

    return run_system_pair(
        X_train,
        y_train,
        X_test,
        cfg
    )


# ============================================================
# 18. DISPLAY
# ============================================================

def display_card(
    pos,
    data,
    is_hot=True
):

    if is_hot:

        items = data[
            "hot_results"
        ]

        nums = " - ".join(
            str(n)
            for n, p in items
        )

        probs = " | ".join(
            f"{n}: {p*100:.1f}%"
            for n, p in items
        )

        html = f"""

        <div class="hot-card">

            <div class="position-title">
                🎯 {POSITION_LABELS[pos]}
            </div>

            <div class="hot-number">
                {nums}
            </div>

            <div class="prob-text">
                🔥 HOT TOP-3<br>
                {probs}
            </div>

            <div class="confidence">

                📌 Gap:
                {data["confidence"]*100:.1f}%

                &nbsp;|&nbsp;

                Coverage:
                {data["hot_coverage"]*100:.1f}%

            </div>

        </div>
        """

    else:

        items = data[
            "dead_results"
        ]

        nums = " - ".join(
            str(n)
            for n, p in items
        )

        probs = " | ".join(
            f"{n}: {p*100:.1f}%"
            for n, p in items
        )

        html = f"""

        <div class="dead-card">

            <div class="position-title">
                🛑 {POSITION_LABELS[pos]}
            </div>

            <div class="dead-number">
                {nums}
            </div>

            <div class="prob-text">
                🛑 DEAD SCORE TOP-3<br>
                {probs}
            </div>

            <div class="confidence">

                ความมั่นใจดับ:
                {data["dead_coverage"]*100:.1f}%

            </div>

        </div>
        """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 19. BACKTEST SUMMARY
# ============================================================

def calculate_bt_stats(bt_df):

    if bt_df is None or bt_df.empty:

        return {
            "hot": 0,
            "dead": 0,
            "rank1": 0,
            "total": 0
        }

    total = len(bt_df)

    hot = (
        bt_df["ผลเด่น"]
        == "✅ เข้า"
    ).sum()

    dead = (
        bt_df["ผลดับ"]
        == "✅ ผ่าน"
    ).sum()

    rank1 = (
        bt_df["อันดับจริง"]
        == 1
    ).sum()

    return {

        "hot":
            hot / total,

        "dead":
            dead / total,

        "rank1":
            rank1 / total,

        "total":
            total
    }


# ============================================================
# 20. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(
        "<div class='main-title'>"
        "🤖 LOTTO AI PRO V8.5"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>"
        "⚡ FAST ADAPTIVE • "
        "RECENT WEIGHT • "
        "WALK-FORWARD BACKTEST"
        "</div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    lottery = c1.selectbox(
        "🏷️ เลือกประเภทหวย",
        list(
            LOTTERY_SOURCES.keys()
        )
    )

    selected_day = c2.selectbox(
        "📅 วันเป้าหมาย",
        ["อัตโนมัติ"] + DOW_NAMES
    )

    if not st.button(
        "🚀 เริ่มวิเคราะห์ V8.5 FAST ADAPTIVE",
        type="primary",
        use_container_width=True
    ):
        return

    with st.spinner(
        "📥 ดึงข้อมูล + AI + Backtest..."
    ):

        # ====================================================
        # LOAD
        # ====================================================

        df = fetch_lottery_data(
            LOTTERY_SOURCES[
                lottery
            ]
        )

        if len(df) < 50:

            st.error(
                f"❌ ข้อมูลมีเพียง "
                f"{len(df)} งวด "
                f"(ต้องการอย่างน้อย 50 งวด)"
            )

            return

        thai_6d = (
            lottery == "หวยไทย"
            and
            is_thai_6d(df)
        )

        positions = (
            THAI_POSITIONS
            if thai_6d
            else NORMAL_POSITIONS
        )

        # ====================================================
        # TARGET DATE
        # ====================================================

        last_date = pd.Timestamp(
            df["Date"].iloc[-1]
        )

        if selected_day == "อัตโนมัติ":

            if len(df) >= 2:

                interval = max(
                    int(
                        (
                            df["Date"].iloc[-1]
                            -
                            df["Date"].iloc[-2]
                        ).days
                    ),
                    1
                )

            else:

                interval = 7

            days_ahead = interval

        else:

            days_ahead = (
                DOW_NAMES.index(
                    selected_day
                )
                -
                last_date.dayofweek
            ) % 7

            if days_ahead == 0:
                days_ahead = 7

        target_date = (
            last_date
            +
            timedelta(
                days=days_ahead
            )
        )

        # ====================================================
        # DUMMY
        # ====================================================

        dummy = {

            "Date":
                target_date,

            "Result_3D":
                "000",

            "Result_2D":
                "00"
        }

        if thai_6d:

            dummy[
                "Result_6D"
            ] = "000000"

        ext = pd.concat(
            [
                df,
                pd.DataFrame(
                    [dummy]
                )
            ],
            ignore_index=True
        )

        # ====================================================
        # FEATURES
        # ====================================================

        feat = build_features(
            ext,
            thai_6d
        )

        features = get_features(
            thai_6d
        )

        cfg = get_adaptive_config(
            len(df)
        )

        # ====================================================
        # STATUS
        # ====================================================

        st.markdown(
            f"""

            <div class="status-card">

            ✅ <b>ข้อมูล:</b>
            {len(df):,} งวด

            &nbsp;|&nbsp;

            📅 <b>เป้าหมาย:</b>
            {target_date.strftime("%d/%m/%Y")}

            &nbsp;|&nbsp;

            🧠 <b>Features:</b>
            {len(features)}

            &nbsp;|&nbsp;

            ⚡ <b>Train Window:</b>
            {cfg["train_window"]}

            </div>

            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # ANALYSIS
        # ====================================================

        final = {}

        progress = st.progress(0)

        bt_steps = min(
            10,
            max(
                0,
                len(df) - cfg[
                    "min_train"
                ]
            )
        )

        for i, pos in enumerate(
            positions
        ):

            # ----------------------------------------------
            # FINAL
            # ----------------------------------------------

            final[pos] = (
                final_prediction(
                    feat,
                    pos,
                    features,
                    cfg
                )
            )

            # ----------------------------------------------
            # BACKTEST
            # ----------------------------------------------

            if bt_steps > 0:

                final[pos][
                    "backtest"
                ] = (
                    run_backtest_for_pos(
                        feat,
                        pos,
                        features,
                        cfg,
                        steps=bt_steps
                    )
                )

            else:

                final[pos][
                    "backtest"
                ] = None

            progress.progress(
                int(
                    (
                        (i + 1)
                        /
                        len(positions)
                    ) * 100
                )
            )

        progress.empty()

    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        "### 📊 สรุปผล AI"
    )

    summary = []

    for pos in positions:

        hot = final[pos][
            "hot_results"
        ]

        dead = final[pos][
            "dead_results"
        ]

        bt = calculate_bt_stats(
            final[pos]["backtest"]
        )

        summary.append({

            "ตำแหน่ง":
                POSITION_LABELS[pos],

            "🔥 เด่นเต็ง":
                f"{hot[0][0]} "
                f"({hot[0][1]*100:.1f}%)",

            "🔥 HOT TOP3":
                " - ".join(
                    str(n)
                    for n, p in hot
                ),

            "🛑 ดับเต็ง":
                f"{dead[0][0]} "
                f"({dead[0][1]*100:.1f}%)",

            "🛑 DEAD TOP3":
                " - ".join(
                    str(n)
                    for n, p in dead
                ),

            "BT HOT":
                f"{bt['hot']*100:.0f}%",

            "BT DEAD":
                f"{bt['dead']*100:.0f}%"
        })

    st.dataframe(
        pd.DataFrame(summary),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # ========================================================
    # TABS
    # ========================================================

    t1, t2, t3, t4 = st.tabs(
        [
            "🔥 เจาะลึกเลขเด่น",
            "🛑 เจาะลึกเลขดับ",
            "📜 ประวัติ 10 งวด",
            "📈 Backtest"
        ]
    )

    # ========================================================
    # HOT
    # ========================================================

    with t1:

        for i in range(
            0,
            len(positions),
            2
        ):

            cols = st.columns(2)

            with cols[0]:

                display_card(
                    positions[i],
                    final[positions[i]],
                    True
                )

            if i + 1 < len(
                positions
            ):

                with cols[1]:

                    display_card(
                        positions[i + 1],
                        final[
                            positions[i + 1]
                        ],
                        True
                    )

    # ========================================================
    # DEAD
    # ========================================================

    with t2:

        for i in range(
            0,
            len(positions),
            2
        ):

            cols = st.columns(2)

            with cols[0]:

                display_card(
                    positions[i],
                    final[positions[i]],
                    False
                )

            if i + 1 < len(
                positions
            ):

                with cols[1]:

                    display_card(
                        positions[i + 1],
                        final[
                            positions[i + 1]
                        ],
                        False
                    )

    # ========================================================
    # HISTORY
    # ========================================================

    with t3:

        st.markdown(
            "### 📜 ผลจริง 10 งวดล่าสุด"
        )

        history_cols = (
            ["Date"] +
            positions
        )

        history = (
            feat
            .iloc[:-1]
            .tail(10)
            [history_cols]
            .copy()
            .sort_values(
                "Date",
                ascending=False
            )
        )

        history["Date"] = (
            history["Date"]
            .dt.strftime(
                "%d/%m/%Y"
            )
        )

        rename = {
            pos:
                POSITION_LABELS[pos]
            for pos in positions
        }

        rename["Date"] = "วันที่"

        history = (
            history
            .rename(
                columns=rename
            )
        )

        for col in history.columns:

            if col != "วันที่":

                history[col] = (
                    history[col]
                    .astype(int)
                    .astype(str)
                )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # BACKTEST
    # ========================================================

    with t4:

        st.markdown(
            "### 📈 Walk-Forward Backtest"
        )

        st.info(
            "AI จะเรียนรู้เฉพาะข้อมูลที่เกิดขึ้น "
            "ก่อนงวดที่กำลังทดสอบเท่านั้น "
            "และใช้ข้อมูลล่าสุดให้น้ำหนักมากกว่าอดีต"
        )

        for pos in positions:

            bt_df = final[pos][
                "backtest"
            ]

            if (
                bt_df is None
                or
                bt_df.empty
            ):
                continue

            stats = calculate_bt_stats(
                bt_df
            )

            title = (
                f"📊 {POSITION_LABELS[pos]} "
                f"| HOT "
                f"{stats['hot']*100:.0f}% "
                f"| DEAD "
                f"{stats['dead']*100:.0f}% "
                f"| Rank1 "
                f"{stats['rank1']*100:.0f}%"
            )

            with st.expander(
                title,
                expanded=False
            ):

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "🔥 HOT Top3",
                    f"{stats['hot']*100:.0f}%"
                )

                c2.metric(
                    "🛑 DEAD ผ่าน",
                    f"{stats['dead']*100:.0f}%"
                )

                c3.metric(
                    "🎯 เลขจริง Rank1",
                    f"{stats['rank1']*100:.0f}%"
                )

                st.dataframe(
                    bt_df.sort_values(
                        "วันที่",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
