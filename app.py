# ============================================================
# 🤖 LOTTO AI PRO V8.3.1 TURBO EXTREME
# ============================================================
# STRICT WALK-FORWARD
# LEAKAGE SAFE
# NO PERSISTENT MODEL
# THAI LOTTERY 6D + 2D
# NORMAL LOTTERY 3D + 2D
#
# ============================================================
# V8.3.1 MAJOR UPDATE
# ============================================================
#
# 🔥 HOT AI SYSTEM
# ------------------------------------------------------------
# ✅ Independent Feature Selection
# ✅ ExtraTrees + HistGradientBoosting
# ✅ HOT TOP-3
# ✅ Hot Probability
# ✅ Hot Confidence
# ✅ Hot Walk-Forward Backtest
#
# 🛑 DEAD AI SYSTEM
# ------------------------------------------------------------
# ✅ Independent Feature Selection
# ✅ Separate Dead Scoring Pipeline
# ✅ ExtraTrees + HistGradientBoosting
# ✅ DEAD TOP-5
# ✅ Dead Score
# ✅ Dead Walk-Forward Backtest
#
# ⚡ TURBO
# ------------------------------------------------------------
# ✅ Float32
# ✅ Lightweight Feature Selection
# ✅ Limited Walk-Forward
# ✅ Adaptive Configuration
# ✅ No Persistent Model
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
# 1. STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lotto AI V8.3.1 Turbo Extreme",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. DATA SOURCES
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


# ============================================================
# 3. CONSTANTS
# ============================================================

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
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "T2",
    "O2"
]


NORMAL_POSITIONS = [
    "H",
    "T",
    "O",
    "T2",
    "O2"
]


POSITION_LABELS = {

    "H1":
        "💯 หลักแสน 6 ตัว",

    "H2":
        "🔢 หลักหมื่น 6 ตัว",

    "H3":
        "🔢 หลักพัน 6 ตัว",

    "H4":
        "🔢 หลักร้อย 6 ตัว",

    "H5":
        "🔟 หลักสิบ 6 ตัว",

    "H6":
        "1️⃣ หลักหน่วย 6 ตัว",

    "H":
        "💯 หลักร้อย 3 ตัวบน",

    "T":
        "🔟 หลักสิบ 3 ตัวบน",

    "O":
        "1️⃣ หลักหน่วย 3 ตัวบน",

    "T2":
        "🔽 หลักสิบ 2 ตัวล่าง",

    "O2":
        "⬇️ หลักหน่วย 2 ตัวล่าง",
}


# ============================================================
# 4. CSS
# ============================================================

def inject_css():

    st.markdown(
        """
        <style>

        .stApp {
            background:#f8fafc;
        }

        .main-title {
            text-align:center;
            font-size:2.15rem;
            font-weight:900;
        }

        .subtitle {
            text-align:center;
            color:#64748b;
            font-size:.88rem;
            margin-bottom:15px;
        }

        .status-card {
            background:#eff6ff;
            border:1px solid #bfdbfe;
            border-radius:14px;
            padding:13px;
            text-align:center;
            color:#1e40af;
            font-weight:700;
            line-height:1.7;
        }

        .hot-card {
            background:#f0fdf4;
            border-left:7px solid #16a34a;
            border-radius:14px;
            padding:14px;
            margin:8px 0;
        }

        .dead-card {
            background:#fef2f2;
            border-left:7px solid #dc2626;
            border-radius:14px;
            padding:14px;
            margin:8px 0;
        }

        .position-title {
            font-size:1.05rem;
            font-weight:900;
            color:#334155;
        }

        .hot-number,
        .dead-number {
            font-size:2.05rem;
            font-weight:900;
            letter-spacing:3px;
            text-align:center;
        }

        .hot-number {
            color:#16a34a;
        }

        .dead-number {
            color:#dc2626;
        }

        .prob-text {
            text-align:center;
            color:#64748b;
            font-size:.8rem;
            line-height:1.7;
        }

        .model-badge {
            text-align:center;
            background:white;
            border-radius:9px;
            padding:6px;
            margin-top:7px;
            color:#475569;
            font-weight:700;
        }

        .confidence {
            text-align:center;
            font-size:.85rem;
            font-weight:800;
            margin-top:5px;
        }

        .system-title {
            font-size:1.2rem;
            font-weight:900;
            margin-top:8px;
            margin-bottom:4px;
        }

        div.stButton > button {
            width:100%;
            min-height:48px;
            border-radius:10px;
            font-size:16px;
            font-weight:800;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 5. THAI MONTH
# ============================================================

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
# 6. DATE NORMALIZER
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
            except Exception:
                return None

    match = re.search(
        r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})",
        text
    )

    if match:

        a, b, c = map(
            int,
            match.groups()
        )

        if a >= 1000:

            y = a
            m = b
            d = c

        else:

            y = c
            m = b
            d = a

        if y < 100:
            y += 2000

        if y >= 2400:
            y -= 543

        try:
            return pd.Timestamp(
                y,
                m,
                d
            )
        except Exception:
            pass

    return None


# ============================================================
# 7. SCRAPER
# ============================================================

class ScrapingError(Exception):
    pass


@st.cache_data(
    ttl=600,
    show_spinner=False
)
def fetch_lottery_data(url):

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Linux; Android 10) "
            "AppleWebKit/537.36 "
            "Chrome/120 Mobile Safari/537.36",

        "Accept-Language":
            "th-TH,th;q=0.9,en;q=0.8"
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

        # ====================================================
        # TABLE
        # ====================================================

        for row in content.find_all("tr"):

            cells = [
                c.get_text(
                    " ",
                    strip=True
                )
                for c in row.find_all(
                    ["td", "th"]
                )
            ]

            text = " ".join(cells)

            if not text:
                continue

            date = normalize_date(text)

            if date is None:
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

                    "Date":
                        date,

                    "Result_6D":
                        six[0],

                    "Result_3D":
                        six[0][-3:],

                    "Result_2D":
                        two[-1]
                })

            elif three and two:

                rows.append({

                    "Date":
                        date,

                    "Result_6D":
                        None,

                    "Result_3D":
                        three[0],

                    "Result_2D":
                        two[-1]
                })

        # ====================================================
        # TEXT FALLBACK
        # ====================================================

        if not rows:

            text = content.get_text(
                separator="\n",
                strip=True
            )

            lines = [
                x.strip()
                for x in text.splitlines()
                if x.strip()
            ]

            current_date = None

            for line in lines:

                date = normalize_date(line)

                if date is not None:
                    current_date = date

                if current_date is None:
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

                        "Date":
                            current_date,

                        "Result_6D":
                            six[0],

                        "Result_3D":
                            six[0][-3:],

                        "Result_2D":
                            two[-1]
                    })

                elif three and two:

                    rows.append({

                        "Date":
                            current_date,

                        "Result_6D":
                            None,

                        "Result_3D":
                            three[0],

                        "Result_2D":
                            two[-1]
                    })

        if not rows:

            raise ScrapingError(
                "ไม่พบข้อมูลหวย"
            )

        df = pd.DataFrame(rows)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Result_3D"] = (
            df["Result_3D"]
            .astype(str)
            .str.extract(
                r"(\d{3})"
            )[0]
            .str.zfill(3)
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(
                r"(\d{2})"
            )[0]
            .str.zfill(2)
        )

        if "Result_6D" in df.columns:

            df["Result_6D"] = (
                df["Result_6D"]
                .astype(str)
                .str.extract(
                    r"(\d{6})"
                )[0]
            )

        df = (
            df
            .dropna(
                subset=["Date"]
            )
            .drop_duplicates(
                subset=["Date"]
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception as exc:

        raise ScrapingError(
            f"โหลดข้อมูลไม่สำเร็จ: {exc}"
        )


# ============================================================
# 8. FORMAT DETECTION
# ============================================================

def is_thai_6d(df):

    if "Result_6D" not in df.columns:
        return False

    return (
        df["Result_6D"]
        .notna()
        .sum()
        >= 10
    )


# ============================================================
# 9. FAST FEATURE ENGINEERING
# ============================================================

def build_features(
    df,
    thai_6d=False
):

    w = df.copy()

    # ========================================================
    # DIGITS
    # ========================================================

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

    # ========================================================
    # DATE
    # ========================================================

    dt = w["Date"].dt

    w["DOW"] = dt.dayofweek.astype(np.int8)
    w["DAY"] = dt.day.astype(np.int8)
    w["MONTH"] = dt.month.astype(np.int8)
    w["DAY_OF_YEAR"] = dt.dayofyear.astype(np.int16)

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

    # ========================================================
    # POSITION FEATURES
    # ========================================================

    for pos in positions:

        s = w[pos]
        p = s.shift(1)

        # ----------------------------------------------------
        # LAGS
        # ----------------------------------------------------

        for lag in (
            1,
            2,
            3,
            5
        ):

            w[f"{pos}_L{lag}"] = (
                s.shift(lag)
            )

        # ----------------------------------------------------
        # ROLLING
        # ----------------------------------------------------

        for window in (
            10,
            20
        ):

            r = p.rolling(
                window,
                min_periods=2
            )

            w[f"{pos}_M{window}"] = (
                r.mean()
            )

            w[f"{pos}_S{window}"] = (
                r.std()
            )

            for digit in (
                0,
                5
            ):

                w[
                    f"{pos}_F{window}_{digit}"
                ] = (
                    (p == digit)
                    .astype(np.float32)
                    .rolling(
                        window,
                        min_periods=2
                    )
                    .mean()
                )

        # ----------------------------------------------------
        # MOMENTUM
        # ----------------------------------------------------

        w[f"{pos}_D1"] = (
            s.shift(1)
            -
            s.shift(2)
        )

        w[f"{pos}_D2"] = (
            s.shift(2)
            -
            s.shift(3)
        )

        # ----------------------------------------------------
        # STATES
        # ----------------------------------------------------

        w[f"{pos}_ODD"] = (
            p % 2
        )

        w[f"{pos}_HIGH"] = (
            p >= 5
        ).astype(np.float32)

        w[f"{pos}_MOD3"] = (
            p % 3
        )

        # ----------------------------------------------------
        # CYCLIC DIGIT
        # ----------------------------------------------------

        w[f"{pos}_SIN"] = np.sin(
            2 * np.pi * p / 10
        ).astype(np.float32)

        w[f"{pos}_COS"] = np.cos(
            2 * np.pi * p / 10
        ).astype(np.float32)

        # ----------------------------------------------------
        # EWMA
        # ----------------------------------------------------

        w[f"{pos}_EWMA7"] = (
            p.ewm(
                span=7,
                adjust=False
            ).mean()
        )

        # ----------------------------------------------------
        # REPEAT
        # ----------------------------------------------------

        w[f"{pos}_REPEAT"] = (
            p == s.shift(2)
        ).astype(np.float32)

    # ========================================================
    # PREVIOUS DRAW AGGREGATES
    # ========================================================

    if thai_6d:

        base = w[
            [
                "H1",
                "H2",
                "H3",
                "H4",
                "H5",
                "H6"
            ]
        ].shift(1)

    else:

        base = w[
            [
                "H",
                "T",
                "O"
            ]
        ].shift(1)

    w["PREV_SUM"] = base.sum(axis=1)

    w["PREV_RANGE"] = (
        base.max(axis=1)
        -
        base.min(axis=1)
    )

    w["PREV_MEAN"] = base.mean(axis=1)

    w["PREV_ODD"] = (
        base % 2
    ).sum(axis=1)

    w["PREV_HIGH"] = (
        base >= 5
    ).sum(axis=1)

    w["PREV_UNIQUE"] = (
        base.nunique(axis=1)
    )

    return (
        w
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


# ============================================================
# 10. FEATURE LIST
# ============================================================

def get_features(thai_6d):

    base = [

        "DOW",
        "DAY",
        "MONTH",
        "DAY_OF_YEAR",

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

        base.extend([

            f"{pos}_L1",
            f"{pos}_L2",
            f"{pos}_L3",
            f"{pos}_L5",

            f"{pos}_M10",
            f"{pos}_M20",

            f"{pos}_S10",
            f"{pos}_S20",

            f"{pos}_D1",
            f"{pos}_D2",

            f"{pos}_ODD",
            f"{pos}_HIGH",

            f"{pos}_MOD3",

            f"{pos}_SIN",
            f"{pos}_COS",

            f"{pos}_EWMA7",

            f"{pos}_REPEAT"
        ])

        for window in (
            10,
            20
        ):

            for digit in (
                0,
                5
            ):

                base.append(
                    f"{pos}_F{window}_{digit}"
                )

    return list(
        dict.fromkeys(base)
    )


# ============================================================
# 11. ADAPTIVE CONFIG
# ============================================================

def get_adaptive_config(n):

    if n >= 700:

        return {
            "min_train": 120,
            "trees": 50,
            "depth": 7,
            "leaf": 2,
            "selected_features": 16,
            "backtest_points": 6,
            "recent_decay": 0.985
        }

    if n >= 400:

        return {
            "min_train": 100,
            "trees": 42,
            "depth": 6,
            "leaf": 2,
            "selected_features": 16,
            "backtest_points": 6,
            "recent_decay": 0.980
        }

    if n >= 200:

        return {
            "min_train": 80,
            "trees": 35,
            "depth": 6,
            "leaf": 2,
            "selected_features": 14,
            "backtest_points": 5,
            "recent_decay": 0.975
        }

    return {
        "min_train": 50,
        "trees": 28,
        "depth": 5,
        "leaf": 2,
        "selected_features": 14,
        "backtest_points": 4,
        "recent_decay": 0.970
    }


# ============================================================
# 12. MODEL FACTORY
# ============================================================

def create_model(
    name,
    cfg,
    system="hot"
):

    t = cfg["trees"]
    d = cfg["depth"]
    l = cfg["leaf"]

    # ========================================================
    # HOT MODEL
    # ========================================================

    if system == "hot":

        if name == "ExtraTrees":

            return ExtraTreesClassifier(

                n_estimators=t,

                max_depth=d,

                min_samples_leaf=l,

                max_features=0.65,

                class_weight="balanced",

                n_jobs=-1,

                random_state=42
            )

        return HistGradientBoostingClassifier(

            max_iter=max(
                25,
                int(t * 0.55)
            ),

            max_leaf_nodes=12,

            learning_rate=0.06,

            min_samples_leaf=l,

            l2_regularization=2,

            random_state=42
        )

    # ========================================================
    # DEAD MODEL
    # ========================================================
    #
    # ใช้ hyperparameter แตกต่างจาก HOT
    # เพื่อไม่ให้เป็นเพียงการ copy ระบบเด่น
    # ========================================================

    if name == "ExtraTrees":

        return ExtraTreesClassifier(

            n_estimators=max(
                20,
                int(t * 0.85)
            ),

            max_depth=max(
                4,
                d - 1
            ),

            min_samples_leaf=max(
                2,
                l
            ),

            max_features=0.45,

            class_weight="balanced",

            n_jobs=-1,

            random_state=91
        )

    return HistGradientBoostingClassifier(

        max_iter=max(
            20,
            int(t * 0.45)
        ),

        max_leaf_nodes=8,

        learning_rate=0.05,

        min_samples_leaf=max(
            2,
            l
        ),

        l2_regularization=4,

        random_state=91
    )


# ============================================================
# 13. FEATURE SELECTION
# ============================================================

def select_features_once(
    X,
    y,
    max_features,
    system="hot"
):

    cols = list(
        X.columns
    )

    if len(cols) <= max_features:
        return cols

    valid = [

        c

        for c in cols

        if X[c].nunique(
            dropna=False
        ) > 1
    ]

    if len(valid) <= max_features:
        return valid

    Xi = (
        X[valid]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(np.float32)
        .fillna(0.0)
    )

    if system == "hot":

        seed = 123

        trees = 8

        depth = 4

    else:

        seed = 321

        trees = 7

        depth = 3

    selector = ExtraTreesClassifier(

        n_estimators=trees,

        max_depth=depth,

        min_samples_leaf=3,

        max_features=0.6,

        n_jobs=-1,

        random_state=seed
    )

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

    return [
        valid[i]
        for i in order[
            :max_features
        ]
    ]


# ============================================================
# 14. NORMALIZE
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

        return (
            np.ones(
                10,
                dtype=np.float32
            )
            / 10
        )

    return (
        p / total
    ).astype(
        np.float32
    )


# ============================================================
# 15. MATRIX
# ============================================================

def prepare_matrix(
    X_train,
    X_test,
    selected
):

    A = (
        X_train[selected]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(np.float32)
    )

    B = (
        X_test[selected]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(np.float32)
    )

    med = A.median()

    A = (
        A
        .fillna(med)
        .fillna(0.0)
    )

    B = (
        B
        .fillna(med)
        .fillna(0.0)
    )

    return A, B


# ============================================================
# 16. MODEL PROBABILITY
# ============================================================

def model_probability(
    X_train,
    y_train,
    X_test,
    cfg,
    selected,
    system
):

    A, B = prepare_matrix(
        X_train,
        X_test,
        selected
    )

    predictions = []

    for name in MODEL_NAMES:

        try:

            model = create_model(
                name,
                cfg,
                system
            )

            model.fit(
                A,
                y_train
            )

            raw = (
                model
                .predict_proba(B)[0]
            )

            out = np.zeros(
                10,
                dtype=np.float32
            )

            for cls, prob in zip(
                model.classes_,
                raw
            ):

                cls = int(cls)

                if 0 <= cls <= 9:

                    out[cls] = (
                        prob
                    )

            predictions.append(
                normalize_probability(
                    out
                )
            )

        except Exception:

            continue

    if not predictions:

        return (
            np.ones(
                10,
                dtype=np.float32
            )
            / 10
        )

    return normalize_probability(
        np.mean(
            predictions,
            axis=0
        )
    )


# ============================================================
# 17. HOT SYSTEM
# ============================================================

def hot_system(
    X_train,
    y_train,
    X_test,
    cfg
):

    selected = select_features_once(

        X_train,

        y_train,

        cfg["selected_features"],

        system="hot"
    )

    probability = model_probability(

        X_train,

        y_train,

        X_test,

        cfg,

        selected,

        system="hot"
    )

    order = np.argsort(
        probability
    )[::-1]

    hot = [

        (
            int(n),

            float(
                probability[n]
            )
        )

        for n in order[:3]
    ]

    if len(order) >= 2:

        confidence = (
            probability[order[0]]
            -
            probability[order[1]]
        )

    else:

        confidence = 0.0

    return {

        "probability":
            probability,

        "hot":
            hot,

        "confidence":
            float(
                confidence
            ),

        "top3":
            float(
                probability[
                    order[:3]
                ].sum()
            ),

        "selected_features":
            selected
    }


# ============================================================
# 18. DEAD SYSTEM
# ============================================================

def dead_system(
    X_train,
    y_train,
    X_test,
    cfg
):

    selected = select_features_once(

        X_train,

        y_train,

        cfg["selected_features"],

        system="dead"
    )

    probability = model_probability(

        X_train,

        y_train,

        X_test,

        cfg,

        selected,

        system="dead"
    )

    # ========================================================
    # DEAD SCORE
    # ========================================================
    #
    # probability ต่ำ
    # =
    # Dead Score สูง
    #
    # ========================================================

    dead_score = (
        1.0
        -
        probability
    ).astype(
        np.float32
    )

    dead_score = normalize_probability(
        dead_score
    )

    order = np.argsort(
        dead_score
    )[::-1]

    dead = [

        (
            int(n),

            float(
                dead_score[n]
            )
        )

        for n in order[:5]
    ]

    return {

        "probability":
            probability,

        "dead_score":
            dead_score,

        "dead":
            dead,

        "top5":
            float(
                dead_score[
                    order[:5]
                ].sum()
            ),

        "selected_features":
            selected
    }


# ============================================================
# 19. HOT WALK-FORWARD
# ============================================================

def hot_walk_forward(
    df_feat,
    pos,
    features,
    cfg
):

    return walk_forward_system(

        df_feat,

        pos,

        features,

        cfg,

        system="hot"
    )


# ============================================================
# 20. DEAD WALK-FORWARD
# ============================================================

def dead_walk_forward(
    df_feat,
    pos,
    features,
    cfg
):

    return walk_forward_system(

        df_feat,

        pos,

        features,

        cfg,

        system="dead"
    )


# ============================================================
# 21. GENERIC WALK-FORWARD
# ============================================================

def walk_forward_system(
    df_feat,
    pos,
    features,
    cfg,
    system="hot"
):

    X = (
        df_feat[features]
        .astype(np.float32)
    )

    y = (
        df_feat[pos]
        .astype(np.int8)
    )

    n = len(df_feat)

    start = cfg["min_train"]

    if n <= start + 2:

        return {
            "tests": 0,
            "scores": {}
        }

    available = n - start

    tests = min(
        cfg["backtest_points"],
        available
    )

    test_indices = np.linspace(

        start,

        n - 1,

        tests,

        dtype=int
    )

    test_indices = np.unique(
        test_indices
    )

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    selection_end = int(
        test_indices[0]
    )

    if (
        y.iloc[
            :selection_end
        ].nunique()
        < 2
    ):

        return {
            "tests": 0,
            "scores": {}
        }

    selected = select_features_once(

        X.iloc[:selection_end],

        y.iloc[:selection_end],

        cfg["selected_features"],

        system=system
    )

    records = []

    # ========================================================
    # TEST
    # ========================================================

    for idx in test_indices:

        if idx < start:
            continue

        y_train = y.iloc[:idx]

        if y_train.nunique() < 2:
            continue

        try:

            probs = model_probability(

                X.iloc[:idx],

                y_train,

                X.iloc[[idx]],

                cfg,

                selected,

                system
            )

        except Exception:

            continue

        actual = int(
            y.iloc[idx]
        )

        order = np.argsort(
            probs
        )[::-1]

        # ====================================================
        # HOT SYSTEM
        # ====================================================

        if system == "hot":

            records.append({

                "top1":
                    int(
                        actual
                        ==
                        order[0]
                    ),

                "top3":
                    int(
                        actual
                        in
                        order[:3]
                    ),

                "top5":
                    int(
                        actual
                        in
                        order[:5]
                    ),

                "logloss":
                    -np.log(
                        max(
                            float(
                                probs[
                                    actual
                                ]
                            ),
                            1e-9
                        )
                    )
            })

        # ====================================================
        # DEAD SYSTEM
        # ====================================================

        else:

            dead_score = (
                1.0
                -
                probs
            )

            dead_order = np.argsort(
                dead_score
            )[::-1]

            records.append({

                "dead5":
                    int(
                        actual
                        in
                        dead_order[:5]
                    ),

                "not_dead5":
                    int(
                        actual
                        not in
                        dead_order[:5]
                    ),

                "dead7":
                    int(
                        actual
                        in
                        dead_order[:7]
                    ),

                "logloss":
                    -np.log(
                        max(
                            float(
                                probs[
                                    actual
                                ]
                            ),
                            1e-9
                        )
                    )
            })

    if not records:

        return {
            "tests": 0,
            "scores": {}
        }

    h = pd.DataFrame(
        records
    )

    # ========================================================
    # RECENCY WEIGHT
    # ========================================================

    decay = (

        cfg["recent_decay"]
        **
        (
            len(h)
            -
            np.arange(
                len(h)
            )
            - 1
        )
    )

    decay /= decay.sum()

    # ========================================================
    # HOT SCORE
    # ========================================================

    if system == "hot":

        scores = {

            "top1":
                float(
                    np.sum(
                        h["top1"]
                        *
                        decay
                    )
                ),

            "top3":
                float(
                    np.sum(
                        h["top3"]
                        *
                        decay
                    )
                ),

            "top5":
                float(
                    np.sum(
                        h["top5"]
                        *
                        decay
                    )
                ),

            "logloss":
                float(
                    np.sum(
                        h["logloss"]
                        *
                        decay
                    )
                )
        }

        scores["score"] = (

            0.45
            *
            scores["top3"]

            +

            0.35
            *
            scores["top1"]

            +

            0.20
            *
            (
                1
                /
                (
                    1
                    +
                    scores["logloss"]
                )
            )
        )

    # ========================================================
    # DEAD SCORE
    # ========================================================

    else:

        scores = {

            "dead5":
                float(
                    np.sum(
                        h["dead5"]
                        *
                        decay
                    )
                ),

            "not_dead5":
                float(
                    np.sum(
                        h["not_dead5"]
                        *
                        decay
                    )
                ),

            "dead7":
                float(
                    np.sum(
                        h["dead7"]
                        *
                        decay
                    )
                ),

            "logloss":
                float(
                    np.sum(
                        h["logloss"]
                        *
                        decay
                    )
                )
        }

        scores["score"] = (

            0.55
            *
            scores["dead5"]

            +

            0.30
            *
            scores["not_dead5"]

            +

            0.15
            *
            scores["dead7"]
        )

    return {

        "tests":
            len(h),

        "scores":
            scores
    }


# ============================================================
# 22. FINAL PREDICTION
# ============================================================

def final_prediction(
    df_feat,
    pos,
    features,
    cfg
):

    X = (
        df_feat[features]
        .astype(np.float32)
    )

    y = (
        df_feat[pos]
        .astype(np.int8)
    )

    X_train = X.iloc[:-1]

    y_train = y.iloc[:-1]

    X_test = X.iloc[[-1]]

    # ========================================================
    # HOT
    # ========================================================

    hot = hot_system(

        X_train,

        y_train,

        X_test,

        cfg
    )

    # ========================================================
    # DEAD
    # ========================================================

    dead = dead_system(

        X_train,

        y_train,

        X_test,

        cfg
    )

    return {

        "hot":
            hot,

        "dead":
            dead
    }


# ============================================================
# 23. DISPLAY HOT
# ============================================================

def display_hot_card(
    pos,
    result
):

    data = result["hot"]["hot"]

    nums = " - ".join(
        str(n)
        for n, _ in data
    )

    probs = " | ".join(
        f"{n}: {p*100:.1f}%"
        for n, p in data
    )

    html = f"""

    <div class="hot-card">

        <div class="position-title">
            {POSITION_LABELS[pos]}
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
            {result["hot"]["confidence"]*100:.1f}%

            &nbsp;|&nbsp;

            Top-3 Coverage:
            {result["hot"]["top3"]*100:.1f}%

        </div>

        <div class="model-badge">

            🔥 HOT AI SYSTEM<br>

            ExtraTrees +
            HistGradientBoosting

        </div>

    </div>

    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 24. DISPLAY DEAD
# ============================================================

def display_dead_card(
    pos,
    result
):

    data = result["dead"]["dead"]

    nums = " - ".join(
        str(n)
        for n, _ in data
    )

    scores = " | ".join(
        f"{n}: {p*100:.1f}%"
        for n, p in data
    )

    html = f"""

    <div class="dead-card">

        <div class="position-title">
            {POSITION_LABELS[pos]}
        </div>

        <div class="dead-number">
            {nums}
        </div>

        <div class="prob-text">

            🛑 DEAD TOP-5<br>

            {scores}

        </div>

        <div class="confidence">

            🛑 Dead Score Coverage:
            {result["dead"]["top5"]*100:.1f}%

        </div>

        <div class="model-badge">

            🛑 DEAD AI SYSTEM<br>

            ExtraTrees +
            HistGradientBoosting

        </div>

    </div>

    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 25. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(

        """
        <div class="main-title">

            🤖 LOTTO AI PRO V8.3.1
            TURBO EXTREME

        </div>

        <div class="subtitle">

            🔥 HOT TOP-3
            &nbsp;|&nbsp;
            🛑 DEAD TOP-5
            &nbsp;|&nbsp;
            STRICT WALK-FORWARD
            &nbsp;|&nbsp;
            LEAKAGE SAFE

        </div>
        """,

        unsafe_allow_html=True
    )

    # ========================================================
    # SELECT
    # ========================================================

    c1, c2 = st.columns(2)

    lottery = c1.selectbox(

        "🏷️ เลือกประเภทหวย",

        list(
            LOTTERY_SOURCES.keys()
        )
    )

    selected_day = c2.selectbox(

        "📅 วันเป้าหมาย",

        [
            "อัตโนมัติ"
        ]
        +
        DOW_NAMES
    )

    # ========================================================
    # START
    # ========================================================

    if not st.button(

        "🚀 เริ่มวิเคราะห์ V8.3.1 TURBO EXTREME",

        type="primary",

        use_container_width=True
    ):

        return

    # ========================================================
    # LOAD
    # ========================================================

    with st.spinner(
        "📥 โหลดข้อมูล..."
    ):

        try:

            df = fetch_lottery_data(

                LOTTERY_SOURCES[
                    lottery
                ]
            )

        except Exception as exc:

            st.error(
                str(exc)
            )

            return

    # ========================================================
    # VALIDATE
    # ========================================================

    if len(df) < 50:

        st.error(

            f"❌ มีข้อมูล "
            f"{len(df)} งวด "
            "ต้องมีอย่างน้อย 50 งวด"
        )

        return

    # ========================================================
    # FORMAT
    # ========================================================

    thai_6d = (

        lottery == "หวยไทย"

        and

        is_thai_6d(df)
    )

    positions = (

        THAI_POSITIONS

        if thai_6d

        else

        NORMAL_POSITIONS
    )

    # ========================================================
    # TARGET DATE
    # ========================================================

    last_date = pd.Timestamp(
        df["Date"].iloc[-1]
    )

    if selected_day == "อัตโนมัติ":

        if len(df) >= 2:

            gap = (

                df["Date"].iloc[-1]

                -

                df["Date"].iloc[-2]
            ).days

            days_ahead = max(
                int(gap),
                1
            )

        else:

            days_ahead = 7

    else:

        target_dow = (
            DOW_NAMES.index(
                selected_day
            )
        )

        days_ahead = (

            target_dow

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

    # ========================================================
    # DUMMY TARGET
    # ========================================================

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

    # ========================================================
    # FEATURES
    # ========================================================

    with st.spinner(

        "⚡ สร้าง Turbo Feature Matrix..."
    ):

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

    # ========================================================
    # INFO
    # ========================================================

    st.info(

        f"""

        ⚡ V8.3.1 TURBO EXTREME |

        ข้อมูล {len(df):,} งวด |

        {"หวยไทย 6 หลัก + 2 หลัก"
         if thai_6d
         else
         "หวย 3 หลัก + 2 หลัก"} |

        Features {len(features)} |

        Hot Top-3 |

        Dead Top-5 |

        Selected ≤
        {cfg["selected_features"]} |

        Trees
        {cfg["trees"]} |

        WF
        {cfg["backtest_points"]}

        """
    )

    # ========================================================
    # RESULT
    # ========================================================

    hot_backtest = {}
    dead_backtest = {}
    final = {}

    progress = st.progress(0)

    status = st.empty()

    total = len(
        positions
    )

    for i, pos in enumerate(
        positions
    ):

        status.caption(

            f"⚡ V8.3.1: "
            f"{POSITION_LABELS[pos]}"
        )

        # ====================================================
        # HOT BACKTEST
        # ====================================================

        hot_backtest[pos] = (
            hot_walk_forward(

                feat.iloc[:-1],

                pos,

                features,

                cfg
            )
        )

        # ====================================================
        # DEAD BACKTEST
        # ====================================================

        dead_backtest[pos] = (
            dead_walk_forward(

                feat.iloc[:-1],

                pos,

                features,

                cfg
            )
        )

        # ====================================================
        # FINAL
        # ====================================================

        final[pos] = (
            final_prediction(

                feat,

                pos,

                features,

                cfg
            )
        )

        progress.progress(

            int(
                (
                    (i + 1)
                    /
                    total
                )
                *
                100
            )
        )

    progress.empty()

    status.empty()

    # ========================================================
    # TEST INFO
    # ========================================================

    hot_tests = [

        x["tests"]

        for x in
        hot_backtest.values()

        if x.get(
            "tests",
            0
        ) > 0
    ]

    dead_tests = [

        x["tests"]

        for x in
        dead_backtest.values()

        if x.get(
            "tests",
            0
        ) > 0
    ]

    if hot_tests:

        hot_test_text = (
            f"{min(hot_tests)}–"
            f"{max(hot_tests)} จุด"
        )

    else:

        hot_test_text = "ไม่มี"

    if dead_tests:

        dead_test_text = (
            f"{min(dead_tests)}–"
            f"{max(dead_tests)} จุด"
        )

    else:

        dead_test_text = "ไม่มี"

    # ========================================================
    # STATUS
    # ========================================================

    st.markdown(

        f"""

        <div class="status-card">

        🤖 <b>
        LOTTO AI PRO V8.3.1
        TURBO EXTREME
        </b><br>

        📊 ข้อมูล:
        {len(df):,} งวด<br>

        📅 เป้าหมาย:
        {target_date.strftime("%d/%m/%Y")}<br>

        🎯 Mode:
        {"6D + 2D"
         if thai_6d
         else
         "3D + 2D"}<br>

        🔥 HOT:
        Top-3 |
        WF {hot_test_text}<br>

        🛑 DEAD:
        Top-5 |
        WF {dead_test_text}<br>

        🧠 Features:
        {len(features)}<br>

        🎯 Selected:
        ≤{cfg["selected_features"]}<br>

        🌳 Trees:
        {cfg["trees"]}<br>

        🔒 No Persistent Model

        </div>

        <br>

        """,

        unsafe_allow_html=True
    )

    # ========================================================
    # TABS
    # ========================================================

    t1, t2, t3 = st.tabs(

        [
            "🔥 เลขเด่น TOP-3",

            "🛑 เลขดับ TOP-5",

            "📊 Backtest"
        ]
    )

    # ========================================================
    # HOT
    # ========================================================

    with t1:

        st.markdown(
            "### 🔥 ระบบเลขเด่น AI TOP-3"
        )

        st.caption(
            "ระบบเลขเด่นแยกจากระบบเลขดับ"
        )

        for pos in positions:

            display_hot_card(

                pos,

                final[pos]
            )

    # ========================================================
    # DEAD
    # ========================================================

    with t2:

        st.markdown(
            "### 🛑 ระบบเลขดับ AI TOP-5"
        )

        st.caption(
            "ระบบเลขดับแยกจากระบบเลขเด่น"
        )

        for pos in positions:

            display_dead_card(

                pos,

                final[pos]
            )

    # ========================================================
    # BACKTEST
    # ========================================================

    with t3:

        st.markdown(
            "## 📊 HOT / DEAD BACKTEST"
        )

        # ====================================================
        # HOT
        # ====================================================

        st.markdown(
            "### 🔥 HOT SYSTEM"
        )

        for pos in positions:

            score = (
                hot_backtest[pos]
                .get(
                    "scores",
                    {}
                )
            )

            if not score:

                continue

            st.markdown(
                f"**{POSITION_LABELS[pos]}**"
            )

            st.dataframe(

                pd.DataFrame([

                    {

                        "Metric":
                            "HOT Top-1",

                        "AI":
                            f"{score['top1']*100:.1f}%",

                        "Random":
                            "10%",

                        "Edge":
                            f"{(score['top1']-.10)*100:+.1f}%"
                    },

                    {

                        "Metric":
                            "HOT Top-3",

                        "AI":
                            f"{score['top3']*100:.1f}%",

                        "Random":
                            "30%",

                        "Edge":
                            f"{(score['top3']-.30)*100:+.1f}%"
                    },

                    {

                        "Metric":
                            "HOT Top-5",

                        "AI":
                            f"{score['top5']*100:.1f}%",

                        "Random":
                            "50%",

                        "Edge":
                            f"{(score['top5']-.50)*100:+.1f}%"
                    },

                    {

                        "Metric":
                            "LogLoss",

                        "AI":
                            f"{score['logloss']:.3f}",

                        "Random":
                            "-",

                        "Edge":
                            "-"
                    },

                    {

                        "Metric":
                            "HOT Score",

                        "AI":
                            f"{score['score']*100:.1f}%",

                        "Random":
                            "-",

                        "Edge":
                            "-"
                    }

                ]),

                use_container_width=True,

                hide_index=True
            )

        # ====================================================
        # DEAD
        # ====================================================

        st.markdown(
            "### 🛑 DEAD SYSTEM"
        )

        for pos in positions:

            score = (
                dead_backtest[pos]
                .get(
                    "scores",
                    {}
                )
            )

            if not score:

                continue

            st.markdown(
                f"**{POSITION_LABELS[pos]}**"
            )

            st.dataframe(

                pd.DataFrame([

                    {

                        "Metric":
                            "Dead-5 Hit",

                        "AI":
                            f"{score['dead5']*100:.1f}%",

                        "Random":
                            "50%",

                        "Edge":
                            f"{(score['dead5']-.50)*100:+.1f}%"
                    },

                    {

                        "Metric":
                            "Avoid Dead-5",

                        "AI":
                            f"{score['not_dead5']*100:.1f}%",

                        "Random":
                            "50%",

                        "Edge":
                            f"{(score['not_dead5']-.50)*100:+.1f}%"
                    },

                    {

                        "Metric":
                            "Dead-7",

                        "AI":
                            f"{score['dead7']*100:.1f}%",

                        "Random":
                            "70%",

                        "Edge":
                            f"{(score['dead7']-.70)*100:+.1f}%"
                    },

                    {

                        "Metric":
                            "LogLoss",

                        "AI":
                            f"{score['logloss']:.3f}",

                        "Random":
                            "-",

                        "Edge":
                            "-"
                    },

                    {

                        "Metric":
                            "DEAD Score",

                        "AI":
                            f"{score['score']*100:.1f}%",

                        "Random":
                            "-",

                        "Edge":
                            "-"
                    }

                ]),

                use_container_width=True,

                hide_index=True
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    with st.expander(
        "🎯 สรุปผล V8.3.1"
    ):

        data = []

        for pos in positions:

            hot_score = (
                hot_backtest[pos]
                .get(
                    "scores",
                    {}
                )
            )

            dead_score = (
                dead_backtest[pos]
                .get(
                    "scores",
                    {}
                )
            )

            hot_top = (
                final[pos]["hot"]["hot"][0]
            )

            dead_top = (
                final[pos]["dead"]["dead"][0]
            )

            data.append({

                "ตำแหน่ง":
                    POSITION_LABELS[pos],

                "🔥 HOT Top-3":
                    " - ".join(
                        str(n)
                        for n, _
                        in final[pos][
                            "hot"
                        ]["hot"]
                    ),

                "HOT #1":
                    f"{hot_top[0]} "
                    f"({hot_top[1]*100:.1f}%)",

                "HOT WF Top-3":
                    f"{hot_score.get('top3', 0)*100:.1f}%",

                "🛑 DEAD Top-5":
                    " - ".join(
                        str(n)
                        for n, _
                        in final[pos][
                            "dead"
                        ]["dead"]
                    ),

                "DEAD #1":
                    f"{dead_top[0]} "
                    f"({dead_top[1]*100:.1f}%)",

                "DEAD WF":
                    f"{dead_score.get('dead5', 0)*100:.1f}%"
            })

        st.dataframe(

            pd.DataFrame(data),

            use_container_width=True,

            hide_index=True
        )

    # ========================================================
    # IMPORTANT NOTE
    # ========================================================

    st.caption(
        "⚠️ ระบบนี้เป็นการจัดอันดับเชิงสถิติ/แมชชีนเลิร์นนิง "
        "ไม่สามารถรับประกันผลสลากจริงได้"
    )


# ============================================================
# 26. RUN
# ============================================================

if __name__ == "__main__":

    main()
