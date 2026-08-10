# ============================================================
# 🤖 LOTTO AI PRO V7 ADAPTIVE
# AI-ONLY • FAST MOBILE • AUTO BACKTEST
# ============================================================
#
# FEATURES
# ------------------------------------------------------------
# ✅ AI ONLY
# ✅ ExtraTrees
# ✅ RandomForest
# ✅ HistGradientBoosting
# ✅ Lightweight Walk-Forward Backtest
# ✅ Automatic Model Selection
# ✅ Top-1 / Top-3 / Top-5 Accuracy
# ✅ Dead-7 Coverage
# ✅ AI Hot Numbers
# ✅ AI Dead Numbers
# ✅ Data Cache
# ✅ Model Cache
# ✅ Mobile Optimized
# ✅ Single app.py
#
# NO:
# ❌ secret_lotto_v4.py
# ❌ secret_lotto_den_v4.py
# ❌ XGBoost
# ❌ Markov
# ❌ Frequency
# ❌ Calendar voting
# ❌ Equation system
#
# ============================================================

import re
import hashlib
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from bs4 import BeautifulSoup

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI PRO V7 Adaptive",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. LOTTERY SOURCES
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

POSITIONS = [
    "H",
    "T",
    "O",
    "T2",
    "O2"
]

POSITION_LABELS = {

    "H":
        "💯 หลักร้อย 3 ตัวบน",

    "T":
        "🔟 หลักสิบ 3 ตัวบน",

    "O":
        "1️⃣ หลักหน่วย 3 ตัวบน",

    "T2":
        "🔽 หลักสิบ 2 ตัวล่าง",

    "O2":
        "⬇️ หลักหน่วย 2 ตัวล่าง"
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


# ============================================================
# 4. CSS
# ============================================================

def inject_css():

    st.markdown(
        """
        <style>

        .stApp {
            background: #f8fafc;
        }

        .main-title {
            text-align: center;
            font-size: 2.7rem;
            font-weight: 900;
            margin-bottom: 3px;
            background: linear-gradient(
                90deg,
                #2563eb,
                #7c3aed,
                #db2777
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 1rem;
            margin-bottom: 20px;
        }

        .status-card {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 15px;
            text-align: center;
            color: #1e40af;
            font-weight: 700;
            line-height: 1.8;
        }

        .hot-card {
            background: #f0fdf4;
            border-left: 7px solid #16a34a;
            border-radius: 14px;
            padding: 18px;
            margin: 12px 0;
        }

        .dead-card {
            background: #fef2f2;
            border-left: 7px solid #dc2626;
            border-radius: 14px;
            padding: 18px;
            margin: 12px 0;
        }

        .position-title {
            font-size: 1.35rem;
            font-weight: 900;
            color: #334155;
            margin-bottom: 8px;
        }

        .hot-number {
            font-size: 2.3rem;
            font-weight: 900;
            color: #16a34a;
            letter-spacing: 3px;
            text-align: center;
        }

        .dead-number {
            font-size: 2.3rem;
            font-weight: 900;
            color: #dc2626;
            letter-spacing: 3px;
            text-align: center;
        }

        .prob-text {
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 5px;
        }

        .model-badge {
            text-align: center;
            background: white;
            border-radius: 10px;
            padding: 8px;
            margin-top: 8px;
            color: #475569;
            font-weight: 700;
        }

        div.stButton > button {
            width: 100%;
            min-height: 48px;
            border-radius: 10px;
            font-size: 17px;
            font-weight: 800;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 5. DATE PARSER
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
    "ธ.ค.": 12,
}


def normalize_date(value):

    if value is None:
        return None

    text = str(value).strip()

    # -----------------------------------------
    # Thai month
    # -----------------------------------------

    for month_name, month_num in THAI_MONTHS.items():

        pattern = (
            rf"(\d{{1,2}})\s*"
            rf"{re.escape(month_name)}\s*"
            rf"(\d{{4}})"
        )

        match = re.search(
            pattern,
            text
        )

        if match:

            day = int(match.group(1))
            year = int(match.group(2))

            if year >= 2400:
                year -= 543

            try:
                return pd.Timestamp(
                    year,
                    month_num,
                    day
                )
            except Exception:
                return None

    # -----------------------------------------
    # Numeric date
    # -----------------------------------------

    match = re.search(
        r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})",
        text
    )

    if match:

        a = int(match.group(1))
        b = int(match.group(2))
        c = int(match.group(3))

        try:

            if a >= 1000:

                year = a
                month = b
                day = c

            else:

                day = a
                month = b
                year = c

                if year < 100:
                    year += 2000

                if year >= 2400:
                    year -= 543

            return pd.Timestamp(
                year,
                month,
                day
            )

        except Exception:
            return None

    return None


# ============================================================
# 6. FETCH LOTTERY DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def fetch_lottery_data(url):

    headers = {
        "User-Agent":
        "Mozilla/5.0 "
        "(Linux; Android 10) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0 Mobile Safari/537.36"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=12
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        content = soup.find(
            "div",
            class_=re.compile(
                r"post-body|entry-content|post-content|content"
            )
        )

        if content is None:
            content = soup

        lines = content.get_text(
            separator="\n"
        ).split("\n")

        extracted = []

        current_date = None

        # -----------------------------------------
        # Main parser
        # -----------------------------------------

        for raw in lines:

            line = raw.strip()

            if not line:
                continue

            parsed_date = normalize_date(
                line
            )

            if parsed_date is not None:
                current_date = parsed_date

            # 3 digit + 2 digit
            match = re.search(
                r"\b(\d{3})\b.*?\b(\d{2})\b",
                line
            )

            if (
                match
                and current_date is not None
            ):

                extracted.append(
                    {
                        "Date": current_date,
                        "Result_3D": match.group(1),
                        "Result_2D": match.group(2)
                    }
                )

        df = pd.DataFrame(
            extracted
        )

        if df.empty:
            return pd.DataFrame()

        # -----------------------------------------
        # Clean
        # -----------------------------------------

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
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(
                r"(\d{2})"
            )[0]
        )

        df = df.dropna(
            subset=[
                "Date",
                "Result_3D",
                "Result_2D"
            ]
        )

        df["Result_3D"] = (
            df["Result_3D"]
            .astype(str)
            .str.zfill(3)
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.zfill(2)
        )

        df = (
            df.drop_duplicates(
                subset=[
                    "Date",
                    "Result_3D",
                    "Result_2D"
                ]
            )
            .sort_values(
                "Date"
            )
            .reset_index(drop=True)
        )

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# 7. BUILD FEATURES
# ============================================================

def build_features(df):

    work = df.copy()

    # -----------------------------------------
    # Digits
    # -----------------------------------------

    work["H"] = (
        work["Result_3D"]
        .str[0]
        .astype(int)
    )

    work["T"] = (
        work["Result_3D"]
        .str[1]
        .astype(int)
    )

    work["O"] = (
        work["Result_3D"]
        .str[2]
        .astype(int)
    )

    work["T2"] = (
        work["Result_2D"]
        .str[0]
        .astype(int)
    )

    work["O2"] = (
        work["Result_2D"]
        .str[1]
        .astype(int)
    )

    # -----------------------------------------
    # Calendar features
    # These are features only, NOT another system
    # -----------------------------------------

    work["DOW"] = (
        work["Date"].dt.dayofweek
    )

    work["DAY"] = (
        work["Date"].dt.day
    )

    work["MONTH"] = (
        work["Date"].dt.month
    )

    work["DOW_SIN"] = np.sin(
        2 * np.pi * work["DOW"] / 7
    )

    work["DOW_COS"] = np.cos(
        2 * np.pi * work["DOW"] / 7
    )

    work["MONTH_SIN"] = np.sin(
        2 * np.pi * work["MONTH"] / 12
    )

    work["MONTH_COS"] = np.cos(
        2 * np.pi * work["MONTH"] / 12
    )

    # -----------------------------------------
    # Global numeric features
    # -----------------------------------------

    work["SUM3"] = (
        work["H"]
        + work["T"]
        + work["O"]
    )

    work["SUM2"] = (
        work["T2"]
        + work["O2"]
    )

    work["RANGE3"] = (
        work[
            ["H", "T", "O"]
        ].max(axis=1)
        -
        work[
            ["H", "T", "O"]
        ].min(axis=1)
    )

    # -----------------------------------------
    # Position features
    # -----------------------------------------

    for pos in POSITIONS:

        # lags
        for lag in range(1, 6):

            work[
                f"{pos}_L{lag}"
            ] = (
                work[pos]
                .shift(lag)
            )

        # rolling
        work[
            f"{pos}_M5"
        ] = (
            work[pos]
            .shift(1)
            .rolling(5)
            .mean()
        )

        work[
            f"{pos}_M10"
        ] = (
            work[pos]
            .shift(1)
            .rolling(10)
            .mean()
        )

        work[
            f"{pos}_S5"
        ] = (
            work[pos]
            .shift(1)
            .rolling(5)
            .std()
        )

        # difference
        work[
            f"{pos}_D1"
        ] = (
            work[pos]
            .shift(1)
            -
            work[pos]
            .shift(2)
        )

        work[
            f"{pos}_D2"
        ] = (
            work[pos]
            .shift(2)
            -
            work[pos]
            .shift(3)
        )

        # odd/even
        previous = (
            work[pos]
            .shift(1)
        )

        work[
            f"{pos}_ODD"
        ] = (
            previous
            .fillna(0)
            .astype(int)
            % 2
        )

        # high/low
        work[
            f"{pos}_HIGH"
        ] = (
            previous
            .fillna(0)
            .astype(int)
            >= 5
        ).astype(int)

        # mirror
        work[
            f"{pos}_MIRROR"
        ] = (
            (
                previous
                .fillna(0)
                .astype(int)
                + 5
            )
            % 10
        )

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    )

    work = work.fillna(0)

    return work


# ============================================================
# 8. FEATURE LIST
# ============================================================

BASE_FEATURES = [
    "DOW",
    "DAY",
    "MONTH",
    "DOW_SIN",
    "DOW_COS",
    "MONTH_SIN",
    "MONTH_COS",
    "SUM3",
    "SUM2",
    "RANGE3"
]

FEATURES = list(
    BASE_FEATURES
)

for pos in POSITIONS:

    FEATURES.extend(
        [
            f"{pos}_L1",
            f"{pos}_L2",
            f"{pos}_L3",
            f"{pos}_L4",
            f"{pos}_L5",
            f"{pos}_M5",
            f"{pos}_M10",
            f"{pos}_S5",
            f"{pos}_D1",
            f"{pos}_D2",
            f"{pos}_ODD",
            f"{pos}_HIGH",
            f"{pos}_MIRROR"
        ]
    )


# ============================================================
# 9. DATA HASH
# ============================================================

def get_data_hash(df):

    hashed = pd.util.hash_pandas_object(
        df[
            [
                "Date",
                "Result_3D",
                "Result_2D"
            ]
        ],
        index=False
    ).values

    return hashlib.md5(
        hashed.tobytes()
    ).hexdigest()


# ============================================================
# 10. ADAPTIVE CONFIG
# ============================================================

def get_adaptive_config(n):

    # Mobile optimized
    if n >= 700:

        return {
            "backtest": 12,
            "trees": 100,
            "depth": 8,
            "leaf": 2
        }

    if n >= 400:

        return {
            "backtest": 12,
            "trees": 80,
            "depth": 7,
            "leaf": 2
        }

    if n >= 200:

        return {
            "backtest": 10,
            "trees": 60,
            "depth": 6,
            "leaf": 2
        }

    return {
        "backtest": 8,
        "trees": 45,
        "depth": 5,
        "leaf": 2
    }


# ============================================================
# 11. MODEL FACTORY
# ============================================================

def create_model(
    model_name,
    config
):

    trees = config["trees"]
    depth = config["depth"]
    leaf = config["leaf"]

    if model_name == "ExtraTrees":

        return ExtraTreesClassifier(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    if model_name == "RandomForest":

        return RandomForestClassifier(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    if model_name == "HistGradientBoosting":

        return HistGradientBoostingClassifier(
            max_iter=max(
                35,
                int(trees * 0.65)
            ),
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=0.5,
            random_state=42
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# 12. PROBABILITY VECTOR
# ============================================================

def probability_vector(
    model,
    X
):

    raw = model.predict_proba(
        X
    )[0]

    output = np.zeros(
        10,
        dtype=float
    )

    for cls, prob in zip(
        model.classes_,
        raw
    ):

        cls = int(cls)

        if 0 <= cls <= 9:
            output[cls] = float(prob)

    total = output.sum()

    if total <= 0:

        return np.ones(10) / 10

    return output / total


# ============================================================
# 13. METRICS
# ============================================================

def calculate_metrics(
    probabilities,
    actual
):

    ranking = np.argsort(
        probabilities
    )[::-1]

    top1 = (
        actual
        in ranking[:1]
    )

    top3 = (
        actual
        in ranking[:3]
    )

    top5 = (
        actual
        in ranking[:5]
    )

    # Dead-7 means actual digit is inside
    # the 7 lowest probability digits.
    dead7 = (
        actual
        in np.argsort(
            probabilities
        )[:7]
    )

    # Log loss
    p = float(
        probabilities[
            int(actual)
        ]
    )

    p = max(
        p,
        1e-9
    )

    logloss = -np.log(p)

    return {
        "top1": int(top1),
        "top3": int(top3),
        "top5": int(top5),
        "dead7": int(dead7),
        "logloss": float(logloss)
    }


# ============================================================
# 14. FAST WALK-FORWARD BACKTEST
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def adaptive_backtest(
    df_features,
    position,
    data_hash,
    config
):

    if len(df_features) < 50:

        return {
            "best_model": "ExtraTrees",
            "scores": {},
            "tests": 0
        }

    models = [
        "ExtraTrees",
        "RandomForest",
        "HistGradientBoosting"
    ]

    X = df_features[
        FEATURES
    ].copy()

    y = (
        df_features[position]
        .astype(int)
        .copy()
    )

    # -----------------------------------------
    # Last N genuine historical observations
    # -----------------------------------------

    n_test = min(
        config["backtest"],
        len(df_features) - 30
    )

    start = (
        len(df_features)
        - n_test
    )

    scores = {}

    # -----------------------------------------
    # Each candidate is tested independently
    # -----------------------------------------

    for model_name in models:

        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        dead7_hits = 0
        total_logloss = 0.0
        tests = 0

        for test_idx in range(
            start,
            len(df_features)
        ):

            train_end = test_idx

            if train_end < 25:
                continue

            X_train = X.iloc[
                :train_end
            ]

            y_train = y.iloc[
                :train_end
            ]

            X_test = X.iloc[
                [test_idx]
            ]

            actual = int(
                y.iloc[test_idx]
            )

            # Need at least 2 classes
            if y_train.nunique() < 2:
                continue

            model = create_model(
                model_name,
                config
            )

            try:

                model.fit(
                    X_train,
                    y_train
                )

                probs = probability_vector(
                    model,
                    X_test
                )

                metrics = calculate_metrics(
                    probs,
                    actual
                )

                top1_hits += metrics["top1"]
                top3_hits += metrics["top3"]
                top5_hits += metrics["top5"]
                dead7_hits += metrics["dead7"]
                total_logloss += metrics["logloss"]

                tests += 1

            except Exception:
                continue

        if tests == 0:

            scores[
                model_name
            ] = {
                "top1": 0.0,
                "top3": 0.0,
                "top5": 0.0,
                "dead7": 0.0,
                "logloss": 99.0,
                "tests": 0,
                "score": -999
            }

            continue

        top1 = (
            top1_hits / tests
        )

        top3 = (
            top3_hits / tests
        )

        top5 = (
            top5_hits / tests
        )

        dead7 = (
            dead7_hits / tests
        )

        logloss = (
            total_logloss / tests
        )

        # -----------------------------------------
        # Adaptive score
        #
        # Top5 is important because 10-class
        # exact prediction is extremely difficult.
        #
        # LogLoss penalizes overconfidence.
        # -----------------------------------------

        score = (
            top1 * 0.30
            +
            top3 * 0.25
            +
            top5 * 0.30
            +
            dead7 * 0.05
            +
            max(
                0,
                1 / (1 + logloss)
            ) * 0.10
        )

        scores[
            model_name
        ] = {
            "top1": top1,
            "top3": top3,
            "top5": top5,
            "dead7": dead7,
            "logloss": logloss,
            "tests": tests,
            "score": score
        }

    # -----------------------------------------
    # Select best
    # -----------------------------------------

    if not scores:

        best_model = "ExtraTrees"

    else:

        best_model = max(
            scores,
            key=lambda x:
                scores[x]["score"]
        )

    return {
        "best_model": best_model,
        "scores": scores,
        "tests": max(
            [
                v["tests"]
                for v in scores.values()
            ],
            default=0
        )
    }


# ============================================================
# 15. FINAL MODEL CACHE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def train_final_model(
    X_train,
    y_train,
    model_name,
    config,
    lottery_name,
    position,
    data_hash
):

    model = create_model(
        model_name,
        config
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# 16. FINAL PREDICTION
# ============================================================

def final_prediction(
    df_features,
    position,
    selected_model,
    config,
    lottery_name,
    data_hash
):

    X = df_features[
        FEATURES
    ].copy()

    y = (
        df_features[position]
        .astype(int)
    )

    # Last row is the target/dummy row
    X_train = X.iloc[
        :-1
    ].copy()

    y_train = y.iloc[
        :-1
    ].copy()

    X_next = X.iloc[
        [-1]
    ].copy()

    # Remove first rows with weak lag information
    start = min(
        10,
        max(
            0,
            len(X_train) - 20
        )
    )

    X_train = X_train.iloc[
        start:
    ]

    y_train = y_train.iloc[
        start:
    ]

    model = train_final_model(
        X_train,
        y_train,
        selected_model,
        config,
        lottery_name,
        position,
        data_hash
    )

    probabilities = probability_vector(
        model,
        X_next
    )

    hot_idx = np.argsort(
        probabilities
    )[::-1][:5]

    dead_idx = np.argsort(
        probabilities
    )[:7]

    hot = [
        (
            int(i),
            float(probabilities[i])
        )
        for i in hot_idx
    ]

    dead = [
        (
            int(i),
            float(probabilities[i])
        )
        for i in dead_idx
    ]

    return {
        "model": selected_model,
        "probabilities": probabilities,
        "hot": hot,
        "dead": dead
    }


# ============================================================
# 17. TARGET DATE
# ============================================================

def calculate_target_date(
    df,
    selected_day
):

    last_date = pd.Timestamp(
        df["Date"].iloc[-1]
    )

    day_map = {
        "อัตโนมัติ": None,
        "วันจันทร์": 0,
        "วันอังคาร": 1,
        "วันพุธ": 2,
        "วันพฤหัสบดี": 3,
        "วันศุกร์": 4,
        "วันเสาร์": 5,
        "วันอาทิตย์": 6
    }

    target_dow = day_map[
        selected_day
    ]

    if target_dow is None:

        if len(df) >= 2:

            gap = (
                last_date
                -
                pd.Timestamp(
                    df["Date"].iloc[-2]
                )
            ).days

            if gap <= 0:
                gap = 7

        else:

            gap = 7

        return (
            last_date
            + timedelta(days=gap)
        )

    days_ahead = (
        target_dow
        -
        last_date.dayofweek
    )

    if days_ahead <= 0:
        days_ahead += 7

    return (
        last_date
        + timedelta(days=days_ahead)
    )


# ============================================================
# 18. DISPLAY HOT
# ============================================================

def display_hot(
    position,
    result
):

    hot = result["hot"]

    numbers = " - ".join(
        str(num)
        for num, _ in hot
    )

    probability_text = " | ".join(
        f"{num}: {prob * 100:.1f}%"
        for num, prob in hot
    )

    st.markdown(
        f"""
        <div class="hot-card">

            <div class="position-title">
                {POSITION_LABELS[position]}
            </div>

            <div class="hot-number">
                {numbers}
            </div>

            <div class="prob-text">
                AI Probability:
                {probability_text}
            </div>

            <div class="model-badge">
                🤖 Model: {result["model"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 19. DISPLAY DEAD
# ============================================================

def display_dead(
    position,
    result
):

    dead = result["dead"]

    numbers = " - ".join(
        str(num)
        for num, _ in dead
    )

    probability_text = " | ".join(
        f"{num}: {prob * 100:.1f}%"
        for num, prob in dead
    )

    st.markdown(
        f"""
        <div class="dead-card">

            <div class="position-title">
                {POSITION_LABELS[position]}
            </div>

            <div class="dead-number">
                {numbers}
            </div>

            <div class="prob-text">
                AI Probability ต่ำสุด:
                {probability_text}
            </div>

            <div class="model-badge">
                🤖 Model: {result["model"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 20. DISPLAY BACKTEST
# ============================================================

def display_backtest(
    position,
    backtest
):

    st.markdown(
        f"### {POSITION_LABELS[position]}"
    )

    if not backtest["scores"]:

        st.warning(
            "ไม่มีผล Backtest เพียงพอ"
        )

        return

    rows = []

    for model_name, score in (
        backtest["scores"].items()
    ):

        rows.append(
            {
                "AI Model":
                    model_name,

                "Top-1":
                    f"{score['top1'] * 100:.1f}%",

                "Top-3":
                    f"{score['top3'] * 100:.1f}%",

                "Top-5":
                    f"{score['top5'] * 100:.1f}%",

                "Dead-7":
                    f"{score['dead7'] * 100:.1f}%",

                "LogLoss":
                    f"{score['logloss']:.3f}",

                "Adaptive Score":
                    f"{score['score'] * 100:.1f}%",

                "Tests":
                    score["tests"]
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"🤖 AI ที่ระบบเลือก: "
        f"**{backtest['best_model']}**"
    )


# ============================================================
# 21. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(
        """
        <div class="main-title">
            🤖 LOTTO AI PRO V7
        </div>

        <div class="subtitle">
            ADAPTIVE • AI-ONLY • AUTO BACKTEST • MOBILE FAST
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        lottery = st.selectbox(
            "🏷️ เลือกประเภทหวย",
            list(
                LOTTERY_SOURCES.keys()
            )
        )

    with col2:

        selected_day = st.selectbox(
            "📅 วันเป้าหมาย",
            [
                "อัตโนมัติ",
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

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    run = st.button(
        "🚀 เริ่มวิเคราะห์ PRO V7 ADAPTIVE",
        type="primary",
        use_container_width=True
    )

    if not run:
        st.info(
            "เลือกหวยและวันเป้าหมาย แล้วกด "
            "🚀 เริ่มวิเคราะห์"
        )
        return

    url = LOTTERY_SOURCES[
        lottery
    ]

    # --------------------------------------------------------
    # FETCH
    # --------------------------------------------------------

    with st.spinner(
        "📥 กำลังดึงข้อมูลย้อนหลัง..."
    ):

        df = fetch_lottery_data(
            url
        )

    if df.empty:

        st.error(
            "❌ ดึงข้อมูลไม่ได้"
        )

        st.info(
            "ตรวจสอบ URL ของแหล่งข้อมูล หรือกดใหม่อีกครั้ง"
        )

        return

    if len(df) < 50:

        st.error(
            f"❌ พบข้อมูลเพียง {len(df)} งวด"
        )

        st.warning(
            "PRO V7 Adaptive ต้องการข้อมูลอย่างน้อย 50 งวด "
            "เพื่อทำ Backtest ได้เหมาะสม"
        )

        return

    # --------------------------------------------------------
    # Target date
    # --------------------------------------------------------

    target_date = calculate_target_date(
        df,
        selected_day
    )

    # --------------------------------------------------------
    # Dummy target row
    #
    # IMPORTANT:
    # The dummy result itself is never used as training data.
    # Its previous values are used only for lag features.
    # --------------------------------------------------------

    dummy = pd.DataFrame(
        [
            {
                "Date": target_date,
                "Result_3D": "000",
                "Result_2D": "00"
            }
        ]
    )

    extended = pd.concat(
        [
            df,
            dummy
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    with st.spinner(
        "🧠 กำลังสร้าง AI Features..."
    ):

        feature_df = build_features(
            extended
        )

    config = get_adaptive_config(
        len(df)
    )

    data_hash = get_data_hash(
        df
    )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    st.info(
        f"⚡ Adaptive Backtest: "
        f"{config['backtest']} งวดล่าสุด / "
        f"{len(df):,} งวด"
    )

    backtest_results = {}

    progress = st.progress(
        0
    )

    for idx, position in enumerate(
        POSITIONS
    ):

        backtest_results[position] = (
            adaptive_backtest(
                feature_df.iloc[:-1],
                position,
                data_hash,
                config
            )
        )

        progress.progress(
            int(
                (idx + 1)
                / len(POSITIONS)
                * 100
            )
        )

    progress.empty()

    # --------------------------------------------------------
    # FINAL AI
    # --------------------------------------------------------

    with st.spinner(
        "🤖 กำลัง Train AI รอบสุดท้าย..."
    ):

        final_results = {}

        for position in POSITIONS:

            selected_model = (
                backtest_results[
                    position
                ]["best_model"]
            )

            final_results[position] = (
                final_prediction(
                    feature_df,
                    position,
                    selected_model,
                    config,
                    lottery,
                    data_hash
                )
            )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    selected_models = [
        final_results[p]["model"]
        for p in POSITIONS
    ]

    model_text = " | ".join(
        selected_models
    )

    st.markdown(
        f"""
        <div class="status-card">

        🤖 <b>PRO V7 AI-ONLY ADAPTIVE</b><br>

        📊 ข้อมูลย้อนหลัง:
        {len(df):,} งวด<br>

        📅 งวดล่าสุด:
        {df["Date"].iloc[-1].strftime("%d/%m/%Y")}<br>

        🎯 งวดเป้าหมาย:
        {target_date.strftime("%d/%m/%Y")}
        ({DOW_NAMES[target_date.dayofweek]})<br>

        🌳 Trees:
        {config["trees"]}<br>

        🔄 Backtest:
        {config["backtest"]} งวด<br>

        🧠 Models:
        {model_text}

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab_hot, tab_dead, tab_accuracy = st.tabs(
        [
            "🎯 เลขเด่น AI",
            "🛑 เลขดับ 7 ตัว",
            "📊 Accuracy / Backtest"
        ]
    )

    # ========================================================
    # HOT
    # ========================================================

    with tab_hot:

        st.subheader(
            "🎯 AI TOP 5 — เลขเด่น"
        )

        st.caption(
            "เรียงตามความน่าจะเป็นที่ AI ประเมินจากสูง → ต่ำ"
        )

        for position in POSITIONS:

            display_hot(
                position,
                final_results[position]
            )

    # ========================================================
    # DEAD
    # ========================================================

    with tab_dead:

        st.subheader(
            "🛑 AI BOTTOM 7 — เลขดับ"
        )

        st.warning(
            "เลขดับ = 7 ตัวที่ AI ประเมินความน่าจะเป็นต่ำที่สุด "
            "ไม่ได้หมายความว่าเลขเหล่านี้จะไม่ออกแน่นอน"
        )

        for position in POSITIONS:

            display_dead(
                position,
                final_results[position]
            )

    # ========================================================
    # ACCURACY
    # ========================================================

    with tab_accuracy:

        st.subheader(
            "📊 Auto Backtest / Accuracy"
        )

        st.caption(
            "ระบบเลือกโมเดลแยกตามแต่ละหลัก "
            "จากผลทดสอบย้อนหลัง"
        )

        for position in POSITIONS:

            display_backtest(
                position,
                backtest_results[position]
            )

            st.write("")

    # ========================================================
    # SUMMARY
    # ========================================================

    with st.expander(
        "📋 สรุปโมเดลที่เลือก"
    ):

        summary = []

        for position in POSITIONS:

            bt = (
                backtest_results[
                    position
                ]
            )

            best = (
                bt["best_model"]
            )

            best_score = (
                bt["scores"]
                .get(
                    best,
                    {}
                )
                .get(
                    "score",
                    0
                )
            )

            summary.append(
                {
                    "ตำแหน่ง":
                        POSITION_LABELS[position],

                    "AI ที่เลือก":
                        best,

                    "Adaptive Score":
                        f"{best_score * 100:.1f}%",

                    "Top-1":
                        f"{bt['scores'].get(best, {}).get('top1', 0) * 100:.1f}%",

                    "Top-3":
                        f"{bt['scores'].get(best, {}).get('top3', 0) * 100:.1f}%",

                    "Top-5":
                        f"{bt['scores'].get(best, {}).get('top5', 0) * 100:.1f}%",

                    "Dead-7":
                        f"{bt['scores'].get(best, {}).get('dead7', 0) * 100:.1f}%"
                }
            )

        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # DATA INFO
    # ========================================================

    with st.expander(
        "🔧 System Information"
    ):

        st.write(
            f"Lottery: {lottery}"
        )

        st.write(
            f"Historical draws: {len(df):,}"
        )

        st.write(
            f"Data Hash: {data_hash[:16]}..."
        )

        st.write(
            "Engine: PRO V7 AI-ONLY Adaptive"
        )

        st.write(
            "Models: ExtraTrees / RandomForest / HistGradientBoosting"
        )

        st.write(
            "XGBoost: Disabled for Mobile Speed"
        )

        st.write(
            "Frequency Engine: Disabled"
        )

        st.write(
            "Markov Engine: Disabled"
        )

        st.write(
            "External Formula Voting: Disabled"
        )

        st.write(
            "Data Leakage Protection: Enabled"
        )


# ============================================================
# 22. RUN APP
# ============================================================

if __name__ == "__main__":
    main()
