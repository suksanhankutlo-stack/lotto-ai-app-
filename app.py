# ============================================================
# 🎯 LOTTO AI V2.0
# MAIN ARTICLE SCRAPER + MONGODB UPSERT
# WALK-FORWARD BACKTEST + ADAPTIVE ENSEMBLE
# TOP-3 EVERY POSITION + NEXT DRAW PREDICTION
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import warnings
import os
import math

from collections import Counter

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# XGBoost เป็น optional
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# MongoDB
from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi
import certifi

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "🎯 LOTTO AI V2.0 — Adaptive Walk-Forward"

REQUEST_TIMEOUT = 20

DEFAULT_LAG = 5

# จำนวนงวดสำหรับ Backtest
DEFAULT_BACKTEST = 30

# อย่างน้อยต้องมีข้อมูลเท่านี้
MIN_HISTORY = 40

# TOP-N
TOP_N = 3


# ============================================================
# LOTTERY SOURCES
# ============================================================

LOTTERY_SOURCES = {
    "หวยไทย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",

    "หวยลาว":
        "https://suksan18190.blogspot.com/2026/07/blog-post.html",

    "หวยฮานอย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",

    "หวยธกส":
        "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",

    "หวยออมสิน":
        "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",

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
# PAGE
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MONGODB
# ============================================================

def get_mongo_uri():

    # Streamlit Cloud / secrets
    try:
        uri = st.secrets.get("MONGO_URI", None)

        if uri:
            return uri

    except Exception:
        pass

    # Environment variable
    uri = os.environ.get("MONGO_URI")

    return uri


@st.cache_resource
def init_mongo_connection():

    uri = get_mongo_uri()

    if not uri:
        return None

    try:

        client = MongoClient(
            uri,
            server_api=ServerApi("1"),
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=10000
        )

        # ตรวจ connection
        client.admin.command("ping")

        db = client["lottery_ai_database"]

        return db

    except Exception as e:

        st.error(
            "❌ เชื่อมต่อ MongoDB ไม่สำเร็จ\n\n"
            f"{e}"
        )

        return None


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/130.0 Safari/537.36"
    })

    return session


# ============================================================
# 1. FIND MAIN ARTICLE
# ============================================================

def find_main_article(soup):

    candidates = []

    # Blogger
    selectors = [
        ".post-body",
        ".post-body.entry-content",
        ".entry-content",
        "article",
        ".post",
        ".post-content",
        ".blog-posts"
    ]

    for selector in selectors:

        try:

            elements = soup.select(selector)

            for element in elements:

                text = element.get_text(
                    "\n",
                    strip=True
                )

                if len(text) > 100:

                    candidates.append(
                        (len(text), element)
                    )

        except Exception:
            continue

    if not candidates:

        return None

    # เลือก element ที่มี pattern วันที่มากที่สุด
    scored = []

    for length, element in candidates:

        text = element.get_text(
            "\n",
            strip=True
        )

        date_count = len(
            re.findall(
                r"\d{4}-\d{2}-\d{2}",
                text
            )
        )

        score = date_count * 100000 + length

        scored.append(
            (score, element)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored[0][1]


# ============================================================
# 2. PARSE DRAW DATA
# ============================================================

def parse_draws_from_article(article):

    if article is None:
        return []

    # ใช้ text ของบทความหลักเท่านั้น
    text = article.get_text(
        "\n",
        strip=True
    )

    # normalize
    text = text.replace("\xa0", " ")

    # --------------------------------------------------------
    # รูปแบบหลัก:
    #
    # * 2026-09-20 | 123 | 45
    #
    # --------------------------------------------------------

    patterns = [

        r"\*?\s*"
        r"(\d{4}-\d{2}-\d{2})"
        r"\s*\|\s*"
        r"(\d{3})"
        r"\s*\|\s*"
        r"(\d{2})",

        r"(\d{4}-\d{2}-\d{2})"
        r"\s*\|\s*"
        r"(\d{3})"
        r"\s*\|\s*"
        r"(\d{2})",
    ]

    matches = []

    for pattern in patterns:

        found = re.findall(
            pattern,
            text
        )

        if found:
            matches.extend(found)

    # remove duplicate tuples
    matches = list(
        dict.fromkeys(matches)
    )

    rows = []

    for date_str, three_digit, two_digit in matches:

        try:

            # ตรวจสอบ date
            dt = pd.to_datetime(
                date_str,
                format="%Y-%m-%d",
                errors="coerce"
            )

            if pd.isna(dt):
                continue

            # ตรวจสอบเลข
            if not re.fullmatch(
                r"\d{3}",
                three_digit
            ):
                continue

            if not re.fullmatch(
                r"\d{2}",
                two_digit
            ):
                continue

            rows.append({

                "Date": date_str,

                "ThreeDigit": three_digit,

                "TwoDigit": two_digit,

                "Hundreds": int(three_digit[0]),

                "Tens": int(three_digit[1]),

                "Units": int(three_digit[2]),

                "TwoTens": int(two_digit[0]),

                "TwoUnits": int(two_digit[1]),
            })

        except Exception:
            continue

    return rows


# ============================================================
# 3. DATA VALIDATION
# ============================================================

def validate_dataframe(df):

    if df is None or df.empty:

        return pd.DataFrame(), {
            "raw": 0,
            "valid": 0,
            "duplicates": 0,
            "invalid": 0
        }

    df = df.copy()

    raw_count = len(df)

    required = [
        "Date",
        "ThreeDigit",
        "TwoDigit",
        "Hundreds",
        "Tens",
        "Units",
        "TwoTens",
        "TwoUnits"
    ]

    for col in required:

        if col not in df.columns:

            return pd.DataFrame(), {
                "raw": raw_count,
                "valid": 0,
                "duplicates": 0,
                "invalid": raw_count
            }

    # Date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # ตรวจเลข
    mask = (

        df["Date"].notna()

        & df["ThreeDigit"]
            .astype(str)
            .str.fullmatch(r"\d{3}")

        & df["TwoDigit"]
            .astype(str)
            .str.fullmatch(r"\d{2}")
    )

    invalid_count = int((~mask).sum())

    df = df.loc[mask].copy()

    # บังคับ string ให้คง leading zero
    df["ThreeDigit"] = (
        df["ThreeDigit"]
        .astype(str)
        .str.zfill(3)
    )

    df["TwoDigit"] = (
        df["TwoDigit"]
        .astype(str)
        .str.zfill(2)
    )

    # --------------------------------------------------------
    # Duplicate
    # --------------------------------------------------------

    before_dup = len(df)

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    duplicates = before_dup - len(df)

    # --------------------------------------------------------
    # sort
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # Draw_ID
    df["Draw_ID"] = np.arange(
        1,
        len(df) + 1
    )

    # Date เป็น string
    df["Date"] = df["Date"].dt.strftime(
        "%Y-%m-%d"
    )

    stats = {

        "raw": raw_count,

        "valid": len(df),

        "duplicates": duplicates,

        "invalid": invalid_count
    }

    return df, stats


# ============================================================
# 4. SCRAPE WEBSITE
# ============================================================

def scrape_lottery(lottery_type, url):

    session = get_http_session()

    try:

        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        article = find_main_article(
            soup
        )

        if article is None:

            return None, {
                "error":
                    "ไม่พบ Main Article / post-body"
            }

        rows = parse_draws_from_article(
            article
        )

        if not rows:

            return None, {
                "error":
                    "ไม่พบข้อมูลรูปแบบ วันที่ | 3 ตัว | 2 ตัว"
            }

        raw_df = pd.DataFrame(rows)

        df, stats = validate_dataframe(
            raw_df
        )

        if df.empty:

            return None, {
                "error":
                    "ข้อมูลไม่ผ่านการตรวจสอบ",
                **stats
            }

        stats["url"] = url

        return df, stats

    except Exception as e:

        return None, {
            "error": str(e),
            "url": url
        }


# ============================================================
# 5. SAVE TO MONGODB - UPSERT
# ============================================================

def save_to_mongo(
    lottery_type,
    df,
    db
):

    if db is None:

        return False, "MongoDB ไม่ได้เชื่อมต่อ"

    if df is None or df.empty:

        return False, "ไม่มีข้อมูล"

    try:

        collection = db[lottery_type]

        # Unique index
        collection.create_index(
            [("Date", 1)],
            unique=True
        )

        operations = []

        for _, row in df.iterrows():

            doc = {

                "Date": row["Date"],

                "ThreeDigit":
                    str(row["ThreeDigit"]).zfill(3),

                "TwoDigit":
                    str(row["TwoDigit"]).zfill(2),

                "Hundreds":
                    int(row["Hundreds"]),

                "Tens":
                    int(row["Tens"]),

                "Units":
                    int(row["Units"]),

                "TwoTens":
                    int(row["TwoTens"]),

                "TwoUnits":
                    int(row["TwoUnits"])
            }

            operations.append(

                UpdateOne(

                    {
                        "Date": row["Date"]
                    },

                    {
                        "$set": doc
                    },

                    upsert=True
                )
            )

        if operations:

            result = collection.bulk_write(
                operations,
                ordered=False
            )

        return True, (
            f"อัปเดต {len(operations)} รายการ"
        )

    except Exception as e:

        return False, str(e)


# ============================================================
# 6. LOAD MONGO
# ============================================================

def load_from_mongo(
    lottery_type,
    db
):

    if db is None:
        return None

    try:

        collection = db[lottery_type]

        data = list(
            collection.find(
                {},
                {"_id": 0}
            )
        )

        if not data:
            return None

        df = pd.DataFrame(data)

        df, _ = validate_dataframe(
            df
        )

        return df

    except Exception as e:

        st.error(
            f"MongoDB อ่านข้อมูลไม่ได้: {e}"
        )

        return None


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

def make_feature_row(
    history,
    lag=5
):

    history = np.asarray(
        history,
        dtype=int
    )

    if len(history) < lag:

        return None

    row = {}

    # --------------------------------------------------------
    # LAG
    # --------------------------------------------------------

    for i in range(1, lag + 1):

        row[f"lag_{i}"] = int(
            history[-i]
        )

    # --------------------------------------------------------
    # Recent windows
    # --------------------------------------------------------

    for window in [3, 5, 10, 20]:

        if len(history) >= window:

            recent = history[-window:]

        else:

            recent = history

        row[f"mean_{window}"] = float(
            np.mean(recent)
        )

        row[f"std_{window}"] = float(
            np.std(recent)
        )

        row[f"min_{window}"] = int(
            np.min(recent)
        )

        row[f"max_{window}"] = int(
            np.max(recent)
        )

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    for digit in range(10):

        row[
            f"freq10_{digit}"
        ] = (
            np.sum(
                history[-10:] == digit
            )
            if len(history) >= 10
            else np.sum(
                history == digit
            )
        )

        row[
            f"freq20_{digit}"
        ] = (
            np.sum(
                history[-20:] == digit
            )
            if len(history) >= 20
            else np.sum(
                history == digit
            )
        )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    if len(history) >= 2:

        row["diff1"] = int(
            history[-1] -
            history[-2]
        )

        row["absdiff1"] = abs(
            row["diff1"]
        )

    else:

        row["diff1"] = 0
        row["absdiff1"] = 0

    # --------------------------------------------------------
    # Digit properties
    # --------------------------------------------------------

    last = int(history[-1])

    row["last_even"] = int(
        last % 2 == 0
    )

    row["last_high"] = int(
        last >= 5
    )

    row["last_prime"] = int(
        last in [2, 3, 5, 7]
    )

    row["sum_recent5"] = int(
        np.sum(history[-5:])
    )

    row["repeat_last"] = int(
        np.sum(
            history[-5:] == last
        )
    )

    return pd.DataFrame([row])


# ============================================================
# CREATE TRAINING DATA
# ============================================================

def create_training_data(
    series,
    lag=5
):

    series = np.asarray(
        series,
        dtype=int
    )

    rows = []
    targets = []

    for i in range(lag, len(series)):

        history = series[:i]

        feature = make_feature_row(
            history,
            lag
        )

        if feature is None:
            continue

        rows.append(
            feature.iloc[0].to_dict()
        )

        targets.append(
            int(series[i])
        )

    if not rows:

        return None, None

    X = pd.DataFrame(rows)

    y = pd.Series(
        targets,
        dtype=int
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return X, y


# ============================================================
# 8. PROBABILITY ALIGN
# ============================================================

def align_probability(
    model,
    prob
):

    output = np.zeros(10)

    try:

        for idx, cls in enumerate(
            model.classes_
        ):

            cls = int(cls)

            if 0 <= cls <= 9:

                output[cls] = prob[idx]

    except Exception:
        pass

    total = output.sum()

    if total > 0:

        output /= total

    else:

        output[:] = 0.1

    return output


# ============================================================
# 9. STATISTICAL MODEL
# ============================================================

class StatisticalModel:

    def fit(self, history):

        history = np.asarray(
            history,
            dtype=int
        )

        self.history = history

        # frequency
        counts = np.bincount(
            history,
            minlength=10
        ).astype(float)

        self.freq = (
            counts + 1
        ) / (
            counts.sum() + 10
        )

        # Markov
        matrix = np.ones(
            (10, 10)
        )

        for i in range(
            len(history) - 1
        ):

            a = int(history[i])
            b = int(history[i + 1])

            matrix[a, b] += 1

        matrix /= matrix.sum(
            axis=1,
            keepdims=True
        )

        self.markov = matrix

    def predict_proba(self):

        last = int(
            self.history[-1]
        )

        markov_prob = (
            self.markov[last]
        )

        # recent frequency
        recent = self.history[-20:]

        recent_count = np.bincount(
            recent,
            minlength=10
        ).astype(float)

        recent_freq = (
            recent_count + 1
        ) / (
            recent_count.sum() + 10
        )

        prob = (
            0.35 * self.freq
            +
            0.40 * recent_freq
            +
            0.25 * markov_prob
        )

        prob /= prob.sum()

        return prob


# ============================================================
# 10. EQUATION ENGINE
# ============================================================

class EquationEngine:

    def predict_proba(
        self,
        history
    ):

        history = np.asarray(
            history,
            dtype=int
        )

        scores = np.ones(10) * 0.001

        # ----------------------------------------------------
        # 1. Recent frequency
        # ----------------------------------------------------

        recent10 = history[-10:]

        for d in range(10):

            scores[d] += (
                np.sum(
                    recent10 == d
                ) * 0.10
            )

        # ----------------------------------------------------
        # 2. Gap
        # ----------------------------------------------------

        for d in range(10):

            positions = np.where(
                history == d
            )[0]

            if len(positions) == 0:

                gap = len(history)

            else:

                gap = (
                    len(history)
                    - 1
                    - positions[-1]
                )

            # diminishing gap effect
            scores[d] += min(
                gap,
                20
            ) * 0.015

        # ----------------------------------------------------
        # 3. Last transition
        # ----------------------------------------------------

        if len(history) >= 2:

            last = int(history[-1])

            prev = int(history[-2])

            diff = (
                last - prev
            ) % 10

            for d in range(10):

                if (
                    (d - last) % 10
                    == diff
                ):

                    scores[d] += 0.15

        # ----------------------------------------------------
        # 4. Reversal
        # ----------------------------------------------------

        last = int(history[-1])

        reverse = 9 - last

        scores[reverse] += 0.08

        # ----------------------------------------------------
        # 5. Last 3 pattern
        # ----------------------------------------------------

        if len(history) >= 3:

            a = int(history[-3])
            b = int(history[-2])
            c = int(history[-1])

            d1 = (b - a) % 10
            d2 = (c - b) % 10

            next_digit = (
                c + (d2 - d1)
            ) % 10

            scores[next_digit] += 0.20

        scores = np.maximum(
            scores,
            0.0001
        )

        scores /= scores.sum()

        return scores


# ============================================================
# 11. ML MODELS
# ============================================================

def train_ml_models(
    X,
    y
):

    models = {}

    # --------------------------------------------------------
    # ExtraTrees
    # --------------------------------------------------------

    et = ExtraTreesClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    et.fit(X, y)

    models["et"] = et

    # --------------------------------------------------------
    # HGB
    # --------------------------------------------------------

    hgb = HistGradientBoostingClassifier(
        max_iter=60,
        max_leaf_nodes=15,
        learning_rate=0.06,
        l2_regularization=0.5,
        random_state=42
    )

    hgb.fit(X, y)

    models["hgb"] = hgb

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if XGB_AVAILABLE:

        unique_classes = np.sort(
            y.unique()
        )

        # XGBoost multi-class
        if len(unique_classes) >= 2:

            xgb = XGBClassifier(

                n_estimators=60,

                max_depth=5,

                learning_rate=0.06,

                subsample=0.9,

                colsample_bytree=0.9,

                objective="multi:softprob",

                num_class=10,

                eval_metric="mlogloss",

                random_state=42,

                n_jobs=-1,

                verbosity=0
            )

            try:

                xgb.fit(
                    X,
                    y
                )

                models["xgb"] = xgb

            except Exception:
                pass

    return models


# ============================================================
# ML PREDICTION
# ============================================================

def predict_ml(
    models,
    X_next
):

    probabilities = {}

    for name, model in models.items():

        try:

            raw = model.predict_proba(
                X_next
            )[0]

            probabilities[name] = (
                align_probability(
                    model,
                    raw
                )
            )

        except Exception:

            probabilities[name] = (
                np.ones(10) / 10
            )

    return probabilities


# ============================================================
# 12. SINGLE MODEL PREDICTION
# ============================================================

def predict_all_models(
    history,
    lag=5
):

    history = np.asarray(
        history,
        dtype=int
    )

    if len(history) < lag + 10:

        return None

    X, y = create_training_data(
        history,
        lag
    )

    if X is None or len(y) < 20:

        return None

    # Train all available data
    models = train_ml_models(
        X,
        y
    )

    # Next feature
    X_next = make_feature_row(
        history,
        lag
    )

    if X_next is None:
        return None

    ml_probs = predict_ml(
        models,
        X_next
    )

    # STAT
    stat_model = StatisticalModel()

    stat_model.fit(
        history
    )

    prob_stat = (
        stat_model.predict_proba()
    )

    # EQUATION
    equation = EquationEngine()

    prob_eq = equation.predict_proba(
        history
    )

    result = {

        "stat": prob_stat,

        "equation": prob_eq
    }

    result.update(
        ml_probs
    )

    return result


# ============================================================
# 13. TOP N
# ============================================================

def top_digits(
    probability,
    n=3
):

    order = np.argsort(
        probability
    )[::-1]

    return [
        (
            int(d),
            float(
                probability[d]
            )
        )
        for d in order[:n]
    ]


# ============================================================
# 14. WALK FORWARD BACKTEST
# ============================================================

def walk_forward_backtest(
    series,
    lag=5,
    test_size=30
):

    series = np.asarray(
        series,
        dtype=int
    )

    model_names = [
        "stat",
        "equation",
        "et",
        "hgb"
    ]

    if XGB_AVAILABLE:
        model_names.append(
            "xgb"
        )

    hits_top1 = {
        name: 0
        for name in model_names
    }

    hits_top3 = {
        name: 0
        for name in model_names
    }

    total = 0

    # ต้องมี training พอสมควร
    minimum_train = max(
        35,
        lag + 20
    )

    start = max(
        minimum_train,
        len(series) - test_size
    )

    for target_index in range(
        start,
        len(series)
    ):

        history = series[
            :target_index
        ]

        actual = int(
            series[target_index]
        )

        if len(history) < minimum_train:
            continue

        try:

            predictions = (
                predict_all_models(
                    history,
                    lag
                )
            )

            if predictions is None:
                continue

            total += 1

            for name in model_names:

                if name not in predictions:
                    continue

                probs = predictions[name]

                order = np.argsort(
                    probs
                )[::-1]

                if int(order[0]) == actual:

                    hits_top1[name] += 1

                if actual in order[:3]:

                    hits_top3[name] += 1

        except Exception:
            continue

    if total == 0:
        return None

    results = []

    for name in model_names:

        top1 = (
            hits_top1[name]
            / total
        )

        top3 = (
            hits_top3[name]
            / total
        )

        results.append({

            "Model": name,

            "TOP-1":
                round(
                    top1 * 100,
                    2
                ),

            "TOP-3":
                round(
                    top3 * 100,
                    2
                ),

            "Test": total,

            "_top1":
                top1,

            "_top3":
                top3
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# 15. ADAPTIVE WEIGHT
# ============================================================

def calculate_adaptive_weights(
    backtest_df
):

    if (
        backtest_df is None
        or backtest_df.empty
    ):

        return {}

    weights = {}

    for _, row in backtest_df.iterrows():

        name = row["Model"]

        top3 = float(
            row["_top3"]
        )

        top1 = float(
            row["_top1"]
        )

        # TOP3 สำคัญกว่า TOP1
        score = (
            0.65 * top3
            +
            0.35 * top1
        )

        # กัน weight = 0
        weights[name] = max(
            score,
            0.01
        )

    total = sum(
        weights.values()
    )

    if total <= 0:

        equal = (
            1 / len(weights)
        )

        return {
            k: equal
            for k in weights
        }

    weights = {
        k: v / total
        for k, v in weights.items()
    }

    return weights


# ============================================================
# 16. ENSEMBLE
# ============================================================

def ensemble_probability(
    predictions,
    weights
):

    final = np.zeros(10)

    used_weight = 0

    for name, weight in weights.items():

        if name not in predictions:
            continue

        final += (
            weight
            * predictions[name]
        )

        used_weight += weight

    if used_weight <= 0:

        final[:] = 0.1

    else:

        final /= used_weight

    final = np.maximum(
        final,
        0
    )

    final /= final.sum()

    return final


# ============================================================
# 17. DISPLAY DATA QUALITY
# ============================================================

def show_data_quality(
    stats,
    df
):

    st.subheader(
        "🔎 ตรวจสอบคุณภาพข้อมูล"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "ข้อมูลที่พบ",
        stats.get(
            "raw",
            len(df)
        )
    )

    c2.metric(
        "ผ่านตรวจสอบ",
        stats.get(
            "valid",
            len(df)
        )
    )

    c3.metric(
        "ข้อมูลซ้ำ",
        stats.get(
            "duplicates",
            0
        )
    )

    c4.metric(
        "ผิดรูปแบบ",
        stats.get(
            "invalid",
            0
        )
    )

    if not df.empty:

        date_range = (
            f"{df['Date'].min()} → "
            f"{df['Date'].max()}"
        )

    else:

        date_range = "-"

    c5.metric(
        "ช่วงวันที่",
        date_range
    )


# ============================================================
# 18. DISPLAY PREDICTION
# ============================================================

def show_prediction(
    history,
    position_name,
    predictions,
    weights
):

    final_prob = ensemble_probability(
        predictions,
        weights
    )

    top = top_digits(
        final_prob,
        TOP_N
    )

    rows = []

    for rank, (
        digit,
        probability
    ) in enumerate(
        top,
        start=1
    ):

        rows.append({

            "อันดับ":
                rank,

            "เลข":
                digit,

            "โอกาสเชิงโมเดล (%)":
                round(
                    probability * 100,
                    2
                )
        })

    top_df = pd.DataFrame(
        rows
    )

    st.markdown(
        f"### 🎯 TOP-{TOP_N} {position_name}"
    )

    st.dataframe(
        top_df,
        hide_index=True,
        use_container_width=True
    )

    # Full 0-9
    full_df = pd.DataFrame({

        "เลข":
            range(10),

        "Probability (%)":
            np.round(
                final_prob * 100,
                2
            )
    })

    full_df = full_df.sort_values(
        "Probability (%)",
        ascending=False
    ).reset_index(drop=True)

    st.bar_chart(
        full_df.set_index("เลข")
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    model_rows = []

    for name, prob in predictions.items():

        top1 = int(
            np.argmax(prob)
        )

        top3 = [
            int(x)
            for x in np.argsort(
                prob
            )[::-1][:3]
        ]

        model_rows.append({

            "Model":
                name,

            "TOP-1":
                top1,

            "TOP-3":
                ", ".join(
                    map(
                        str,
                        top3
                    )
                ),

            "Weight (%)":
                round(
                    weights.get(
                        name,
                        0
                    ) * 100,
                    2
                )
        })

    st.dataframe(
        pd.DataFrame(model_rows),
        hide_index=True,
        use_container_width=True
    )

    return final_prob


# ============================================================
# 19. MAIN
# ============================================================

def main():

    st.title(
        "🎯 LOTTO AI V2.0"
    )

    st.markdown(
        """
        **Main Article Scraper + MongoDB Upsert + 
        Walk-Forward Backtest + Adaptive Ensemble + TOP-3**
        """
    )

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    st.sidebar.header(
        "⚙️ ตั้งค่าระบบ"
    )

    lottery_type = st.sidebar.selectbox(
        "เลือกประเภทหวย",
        list(
            LOTTERY_SOURCES.keys()
        )
    )

    source_url = (
        LOTTERY_SOURCES[
            lottery_type
        ]
    )

    lag = st.sidebar.slider(
        "จำนวน Lag",
        min_value=3,
        max_value=10,
        value=5
    )

    backtest_size = st.sidebar.slider(
        "จำนวนงวด Backtest",
        min_value=10,
        max_value=50,
        value=DEFAULT_BACKTEST
    )

    use_mongo = st.sidebar.checkbox(
        "ใช้ MongoDB",
        value=True
    )

    # --------------------------------------------------------
    # Mongo
    # --------------------------------------------------------

    db = None

    if use_mongo:

        db = init_mongo_connection()

        if db is not None:

            st.sidebar.success(
                "🟢 MongoDB Connected"
            )

        else:

            st.sidebar.warning(
                "🟡 MongoDB ไม่พร้อมใช้งาน"
            )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    col_a, col_b = st.columns(2)

    with col_a:

        update_web = st.button(
            "🌐 ดึงข้อมูลใหม่จากเว็บ",
            use_container_width=True
        )

    with col_b:

        load_db = st.button(
            "🗄️ โหลดจาก MongoDB",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    df = None
    stats = {}

    # --------------------------------------------------------
    # WEB
    # --------------------------------------------------------

    if update_web:

        with st.spinner(
            "กำลังอ่านเฉพาะ Main Article..."
        ):

            df, stats = scrape_lottery(
                lottery_type,
                source_url
            )

        if df is None:

            st.error(
                "❌ ดึงข้อมูลไม่สำเร็จ"
            )

            if "error" in stats:

                st.code(
                    stats["error"]
                )

            st.stop()

        show_data_quality(
            stats,
            df
        )

        # Save Mongo
        if db is not None:

            ok, message = (
                save_to_mongo(
                    lottery_type,
                    df,
                    db
                )
            )

            if ok:

                st.success(
                    f"🗄️ MongoDB: {message}"
                )

            else:

                st.warning(
                    f"MongoDB: {message}"
                )

    # --------------------------------------------------------
    # MONGO
    # --------------------------------------------------------

    elif load_db:

        if db is None:

            st.error(
                "❌ MongoDB ไม่พร้อมใช้งาน"
            )

            st.stop()

        df = load_from_mongo(
            lottery_type,
            db
        )

        if df is None:

            st.warning(
                "ไม่พบข้อมูลใน MongoDB"
            )

            st.info(
                "กรุณากด "
                "🌐 ดึงข้อมูลใหม่จากเว็บ"
            )

            st.stop()

        stats = {

            "raw": len(df),

            "valid": len(df),

            "duplicates": 0,

            "invalid": 0
        }

        show_data_quality(
            stats,
            df
        )

    # --------------------------------------------------------
    # AUTO LOAD
    # --------------------------------------------------------

    else:

        if db is not None:

            df = load_from_mongo(
                lottery_type,
                db
            )

        if df is None:

            with st.spinner(
                "ยังไม่มีข้อมูล กำลังดึงจากเว็บ..."
            ):

                df, stats = scrape_lottery(
                    lottery_type,
                    source_url
                )

            if df is None:

                st.error(
                    "❌ ไม่สามารถโหลดข้อมูลจริงได้"
                )

                st.stop()

            if db is not None:

                save_to_mongo(
                    lottery_type,
                    df,
                    db
                )

        else:

            stats = {

                "raw": len(df),

                "valid": len(df),

                "duplicates": 0,

                "invalid": 0
            }

        show_data_quality(
            stats,
            df
        )

    # --------------------------------------------------------
    # Data Check
    # --------------------------------------------------------

    if df is None or df.empty:

        st.error(
            "❌ ไม่มีข้อมูลจริงสำหรับวิเคราะห์"
        )

        st.stop()

    if len(df) < MIN_HISTORY:

        st.error(
            f"❌ ข้อมูลมีเพียง {len(df)} งวด "
            f"ต้องการอย่างน้อย {MIN_HISTORY} งวด"
        )

        st.stop()

    # --------------------------------------------------------
    # Display latest
    # --------------------------------------------------------

    st.subheader(
        f"📊 ข้อมูลย้อนหลัง — {lottery_type}"
    )

    display_cols = [
        "Date",
        "ThreeDigit",
        "TwoDigit"
    ]

    st.dataframe(
        df[
            display_cols
        ].tail(20),
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Main Prediction Button
    # --------------------------------------------------------

    if st.button(
        "🚀 RUN AI NEXT DRAW",
        type="primary",
        use_container_width=True
    ):

        st.markdown(
            "---"
        )

        st.header(
            "🔮 วิเคราะห์งวดถัดไป"
        )

        st.caption(
            f"ข้อมูลล่าสุด: "
            f"{df['Date'].max()}"
        )

        # ====================================================
        # POSITION
        # ====================================================

        positions = [

            (
                "Hundreds",
                "หลักร้อย",
                "ThreeDigit"
            ),

            (
                "Tens",
                "หลักสิบ",
                "ThreeDigit"
            ),

            (
                "Units",
                "หลักหน่วย",
                "ThreeDigit"
            ),

            (
                "TwoTens",
                "2 ตัวบน/หลักสิบ",
                "TwoDigit"
            ),

            (
                "TwoUnits",
                "2 ตัวบน/หลักหน่วย",
                "TwoDigit"
            )
        ]

        tabs = st.tabs(
            [
                p[1]
                for p in positions
            ]
        )

        for tab, (
            col_name,
            display_name,
            source
        ) in zip(
            tabs,
            positions
        ):

            with tab:

                series = (
                    df[col_name]
                    .astype(int)
                    .values
                )

                st.markdown(
                    f"## 🎯 {display_name}"
                )

                # ------------------------------------------------
                # BACKTEST
                # ------------------------------------------------

                with st.spinner(
                    "กำลังทำ Walk-Forward Backtest..."
                ):

                    bt = (
                        walk_forward_backtest(
                            series,
                            lag=lag,
                            test_size=backtest_size
                        )
                    )

                if bt is None:

                    st.error(
                        "Backtest ไม่สามารถทำได้"
                    )

                    continue

                # ------------------------------------------------
                # Backtest table
                # ------------------------------------------------

                st.markdown(
                    "### 📈 Walk-Forward Backtest"
                )

                bt_display = bt[
                    [
                        "Model",
                        "TOP-1",
                        "TOP-3",
                        "Test"
                    ]
                ].copy()

                bt_display.columns = [
                    "โมเดล",
                    "TOP-1 (%)",
                    "TOP-3 (%)",
                    "จำนวนทดสอบ"
                ]

                st.dataframe(
                    bt_display,
                    hide_index=True,
                    use_container_width=True
                )

                # ------------------------------------------------
                # Adaptive weights
                # ------------------------------------------------

                weights = (
                    calculate_adaptive_weights(
                        bt
                    )
                )

                weight_df = pd.DataFrame({

                    "Model":
                        list(weights.keys()),

                    "Weight (%)":
                        [
                            round(
                                v * 100,
                                2
                            )
                            for v in weights.values()
                        ]
                })

                st.markdown(
                    "### ⚖️ Adaptive Model Weight"
                )

                st.dataframe(
                    weight_df,
                    hide_index=True,
                    use_container_width=True
                )

                # ------------------------------------------------
                # FINAL TRAIN + NEXT
                # ------------------------------------------------

                with st.spinner(
                    "กำลัง Train ด้วยข้อมูลทั้งหมด "
                    "และสร้าง X_NEXT..."
                ):

                    predictions = (
                        predict_all_models(
                            series,
                            lag
                        )
                    )

                if predictions is None:

                    st.error(
                        "ไม่สามารถสร้าง Prediction ได้"
                    )

                    continue

                final_prob = show_prediction(
                    series,
                    display_name,
                    predictions,
                    weights
                )

                # ------------------------------------------------
                # TOP 3
                # ------------------------------------------------

                top3 = top_digits(
                    final_prob,
                    3
                )

                st.success(
                    "🔒 TOP-3 FINAL: "
                    +
                    " | ".join(
                        [
                            f"{d} "
                            f"({p*100:.2f}%)"
                            for d, p in top3
                        ]
                    )
                )

                # ------------------------------------------------
                # Recent history
                # ------------------------------------------------

                recent = series[-10:]

                history_df = pd.DataFrame({

                    "งวดล่าสุดย้อนกลับ":
                        range(
                            len(recent),
                            0,
                            -1
                        ),

                    "เลข":
                        recent
                })

                st.markdown(
                    "### 📜 ประวัติ 10 งวดล่าสุด"
                )

                st.dataframe(
                    history_df,
                    hide_index=True,
                    use_container_width=True
                )

        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown(
            "---"
        )

        st.header(
            "📌 สรุป TOP-3 ทุกหลัก"
        )

        summary_rows = []

        for col_name, display_name, source in positions:

            series = (
                df[col_name]
                .astype(int)
                .values
            )

            bt = (
                walk_forward_backtest(
                    series,
                    lag=lag,
                    test_size=backtest_size
                )
            )

            if bt is None:
                continue

            weights = (
                calculate_adaptive_weights(
                    bt
                )
            )

            predictions = (
                predict_all_models(
                    series,
                    lag
                )
            )

            if predictions is None:
                continue

            final_prob = (
                ensemble_probability(
                    predictions,
                    weights
                )
            )

            top3 = top_digits(
                final_prob,
                3
            )

            summary_rows.append({

                "หลัก":
                    display_name,

                "อันดับ 1":
                    f"{top3[0][0]} "
                    f"({top3[0][1]*100:.2f}%)",

                "อันดับ 2":
                    f"{top3[1][0]} "
                    f"({top3[1][1]*100:.2f}%)",

                "อันดับ 3":
                    f"{top3[2][0]} "
                    f"({top3[2][1]*100:.2f}%)"
            })

        if summary_rows:

            st.dataframe(
                pd.DataFrame(
                    summary_rows
                ),
                hide_index=True,
                use_container_width=True
            )

        st.info(
            "หมายเหตุ: ค่า Probability เป็นคะแนนจากโมเดล "
            "ไม่ใช่ความน่าจะเป็นทางคณิตศาสตร์ที่รับประกันผลรางวัล"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
