# ================================================================
# 🎯 LOTTO AI V2.2
# ROBUST BLOGGER SCRAPER + WALK-FORWARD + ADAPTIVE ENSEMBLE
# TOP-3 EVERY POSITION
# ================================================================

import re
import time
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

from bs4 import BeautifulSoup

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")


# ================================================================
# OPTIONAL XGBOOST
# ================================================================

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="LOTTO AI V2.2",
    page_icon="🎯",
    layout="wide"
)


# ================================================================
# LOTTERY SOURCES
# ================================================================

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


# ================================================================
# CONFIG
# ================================================================

MIN_HISTORY = 40

BACKTEST_MAX = 25

TOP_K = 3

REQUEST_TIMEOUT = 20

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


# ================================================================
# HEADER
# ================================================================

st.title("🎯 LOTTO AI V2.2")
st.caption(
    "Robust Blogger Scraper • Walk-Forward Backtest • "
    "Adaptive Ensemble • TOP-3 ทุกหลัก"
)

st.info(
    "ระบบนี้เป็นการวิเคราะห์ข้อมูลย้อนหลังเชิงสถิติ/แมชชีนเลิร์นนิง "
    "ไม่สามารถรับประกันผลรางวัลจริงได้"
)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.header("⚙️ ตั้งค่าระบบ")

lottery_type = st.sidebar.selectbox(
    "เลือกประเภทหวย",
    list(LOTTERY_SOURCES.keys())
)

target_date = st.sidebar.date_input(
    "วันที่ต้องการวิเคราะห์",
    value=datetime.now().date()
)

min_history = st.sidebar.slider(
    "จำนวนข้อมูลขั้นต่ำ",
    20,
    200,
    MIN_HISTORY,
    5
)

backtest_n = st.sidebar.slider(
    "จำนวนงวด Backtest",
    5,
    50,
    BACKTEST_MAX,
    5
)

use_xgb = st.sidebar.checkbox(
    "ใช้ XGBoost ถ้าติดตั้งได้",
    value=True
)

show_debug = st.sidebar.checkbox(
    "แสดง Debug Scraper",
    value=False
)


# ================================================================
# SESSION STATE
# ================================================================

if "data" not in st.session_state:
    st.session_state.data = None

if "scraper_debug" not in st.session_state:
    st.session_state.scraper_debug = {}

if "backtest" not in st.session_state:
    st.session_state.backtest = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ================================================================
# TEXT NORMALIZATION
# ================================================================

def normalize_text(text):

    if text is None:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # normalize Thai digits
    thai_digits = "๐๑๒๓๔๕๖๗๘๙"
    arabic_digits = "0123456789"

    trans = str.maketrans(
        thai_digits,
        arabic_digits
    )

    text = text.translate(trans)

    # normalize separators
    text = text.replace("｜", "|")
    text = text.replace("¦", "|")

    # normalize newlines
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text


# ================================================================
# DATE VALIDATOR
# ================================================================

def valid_date(s):

    try:

        d = pd.to_datetime(
            s,
            format="%Y-%m-%d",
            errors="coerce"
        )

        return not pd.isna(d)

    except Exception:
        return False


# ================================================================
# PARSE ONE LINE
# ================================================================

def parse_draw_line(line):

    line = normalize_text(line).strip()

    if not line:
        return None

    # ------------------------------------------------------------
    # Format:
    # 2026-09-21 | 225 | 92
    # ------------------------------------------------------------

    m = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{3})\s*\|\s*(\d{2})",
        line
    )

    if not m:
        return None

    date_str = m.group(1)
    three = m.group(2)
    two = m.group(3)

    if not valid_date(date_str):
        return None

    if len(three) != 3:
        return None

    if len(two) != 2:
        return None

    return {
        "date": date_str,
        "three": three,
        "two": two
    }


# ================================================================
# FIND BLOGGER POST BODY
# ================================================================

def find_post_body(soup):

    candidates = []

    selectors = [

        "div.post-body",

        "div.post-body.entry-content",

        "article div.post-body",

        "article .post-body",

        ".post-body",

        ".entry-content",

    ]

    for selector in selectors:

        try:

            nodes = soup.select(selector)

            for node in nodes:

                text = node.get_text(
                    "\n",
                    strip=True
                )

                if text:

                    date_count = len(
                        re.findall(
                            r"\d{4}-\d{2}-\d{2}",
                            text
                        )
                    )

                    pipe_count = text.count("|")

                    score = (
                        date_count * 1000
                        + pipe_count * 10
                        + min(len(text), 5000) / 10000
                    )

                    candidates.append(
                        (score, node, selector)
                    )

        except Exception:
            pass

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    _, node, selector = candidates[0]

    return node, selector


# ================================================================
# EXTRACT MAIN ARTICLE
# ================================================================

def extract_main_post_text(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # ------------------------------------------------------------
    # 1. Try post body
    # ------------------------------------------------------------

    node, selector = find_post_body(soup)

    if node is not None:

        text = node.get_text(
            "\n",
            strip=True
        )

        return text, f"POST_BODY:{selector}"


    # ------------------------------------------------------------
    # 2. Try article
    # ------------------------------------------------------------

    articles = soup.find_all("article")

    best_text = ""
    best_score = -1

    for article in articles:

        text = article.get_text(
            "\n",
            strip=True
        )

        dates = len(
            re.findall(
                r"\d{4}-\d{2}-\d{2}",
                text
            )
        )

        if dates > best_score:

            best_score = dates
            best_text = text

    if best_text:

        return best_text, "ARTICLE_FALLBACK"


    # ------------------------------------------------------------
    # 3. Body fallback
    # ------------------------------------------------------------

    body = soup.find("body")

    if body:

        text = body.get_text(
            "\n",
            strip=True
        )

        # Important:
        # stop before popular posts / sidebar
        stop_words = [
            "โพสต์ยอดนิยมจากบล็อกนี้",
            "Popular Posts",
            "Popular post",
            "บทความยอดนิยม"
        ]

        for word in stop_words:

            pos = text.find(word)

            if pos > 0:

                text = text[:pos]

        return text, "BODY_FALLBACK"


    return "", "NO_CONTENT"


# ================================================================
# HTTP FETCH
# ================================================================

def fetch_url(url):

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,"
            "image/webp,*/*;q=0.8"
        ),
        "Accept-Language":
            "th-TH,th;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    urls = [
        url,
        url + ("&" if "?" in url else "?") + "m=1"
    ]

    last_error = None

    for test_url in urls:

        try:

            response = requests.get(
                test_url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True
            )

            html = response.text

            if response.status_code == 200 and len(html) > 1000:

                return {
                    "success": True,
                    "url": test_url,
                    "final_url": response.url,
                    "status": response.status_code,
                    "html": html,
                    "length": len(html),
                    "error": None
                }

            last_error = (
                f"HTTP {response.status_code}, "
                f"HTML={len(html)}"
            )

        except Exception as e:

            last_error = str(e)

    return {
        "success": False,
        "url": url,
        "final_url": "",
        "status": None,
        "html": "",
        "length": 0,
        "error": last_error
    }


# ================================================================
# PARSE ALL DRAW DATA
# ================================================================

def parse_draws(text):

    rows = []

    # ------------------------------------------------------------
    # First: line parser
    # ------------------------------------------------------------

    for line in text.splitlines():

        item = parse_draw_line(line)

        if item:

            rows.append(item)


    # ------------------------------------------------------------
    # Second: global fallback
    # ------------------------------------------------------------

    if len(rows) < 3:

        matches = re.findall(
            r"(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{3})\s*\|\s*(\d{2})",
            text
        )

        for d, three, two in matches:

            if valid_date(d):

                rows.append({
                    "date": d,
                    "three": three,
                    "two": two
                })


    # ------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------

    if not rows:

        return pd.DataFrame(
            columns=[
                "date",
                "three",
                "two"
            ]
        )

    df = pd.DataFrame(rows)

    # normalize
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["three"] = (
        df["three"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(3)
    )

    df["two"] = (
        df["two"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(2)
    )

    df = df.dropna(
        subset=["date"]
    )

    # valid
    df = df[
        df["three"].str.len().eq(3)
        &
        df["two"].str.len().eq(2)
    ]

    # remove duplicates
    df = (
        df.sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df


# ================================================================
# SCRAPE LOTTERY
# ================================================================

def scrape_lottery(lottery):

    url = LOTTERY_SOURCES[lottery]

    result = fetch_url(url)

    debug = {
        "source": url,
        "request_url": result.get("url"),
        "final_url": result.get("final_url"),
        "status": result.get("status"),
        "html_length": result.get("length"),
        "success": result.get("success"),
        "error": result.get("error"),
        "parser": "",
        "dates_found": 0,
        "rows": 0
    }

    if not result["success"]:

        return None, debug

    html = result["html"]

    text, parser_name = extract_main_post_text(
        html
    )

    debug["parser"] = parser_name

    # ------------------------------------------------------------
    # parse
    # ------------------------------------------------------------

    df = parse_draws(text)

    debug["dates_found"] = len(
        re.findall(
            r"\d{4}-\d{2}-\d{2}",
            text
        )
    )

    debug["rows"] = len(df)

    # ------------------------------------------------------------
    # emergency full HTML fallback
    # ------------------------------------------------------------

    if len(df) < 3:

        raw_text = BeautifulSoup(
            html,
            "html.parser"
        ).get_text(
            "\n",
            strip=True
        )

        df2 = parse_draws(raw_text)

        if len(df2) > len(df):

            df = df2

            debug["parser"] += " + FULL_HTML_FALLBACK"

            debug["rows"] = len(df)

    if len(df) == 0:

        debug["sample"] = text[:1500]

        return None, debug

    return df, debug


# ================================================================
# DATA VALIDATION
# ================================================================

def validate_data(df):

    report = {}

    if df is None or df.empty:

        return False, {
            "error": "ไม่มีข้อมูล"
        }

    report["rows"] = len(df)

    report["date_min"] = (
        df["date"].min()
    )

    report["date_max"] = (
        df["date"].max()
    )

    report["duplicate_dates"] = int(
        df["date"].duplicated().sum()
    )

    report["nulls"] = int(
        df[["date", "three", "two"]]
        .isna()
        .sum()
        .sum()
    )

    report["valid_3digit"] = int(
        df["three"]
        .astype(str)
        .str.fullmatch(r"\d{3}")
        .sum()
    )

    report["valid_2digit"] = int(
        df["two"]
        .astype(str)
        .str.fullmatch(r"\d{2}")
        .sum()
    )

    ok = (
        report["rows"] >= 3
        and report["valid_3digit"] == report["rows"]
        and report["valid_2digit"] == report["rows"]
    )

    return ok, report


# ================================================================
# MONGODB
# ================================================================

def get_mongo_uri():

    try:

        if "MONGO_URI" in st.secrets:

            return st.secrets["MONGO_URI"]

    except Exception:
        pass

    return None


@st.cache_resource
def get_mongo_collection():

    uri = get_mongo_uri()

    if not uri:
        return None

    try:

        from pymongo import MongoClient

        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000
        )

        client.admin.command(
            "ping"
        )

        db = client["lotto_ai"]

        collection = db["results"]

        collection.create_index(
            [
                ("lottery", 1),
                ("date", 1)
            ],
            unique=True
        )

        return collection

    except Exception:

        return None


# ================================================================
# SAVE MONGO
# ================================================================

def save_to_mongo(
    df,
    lottery
):

    collection = get_mongo_collection()

    if collection is None:

        return False, "MongoDB ไม่พร้อมใช้งาน"

    from pymongo import UpdateOne

    operations = []

    for _, row in df.iterrows():

        operations.append(
            UpdateOne(
                {
                    "lottery": lottery,
                    "date": row["date"].strftime(
                        "%Y-%m-%d"
                    )
                },
                {
                    "$set": {
                        "lottery": lottery,
                        "date": row["date"].strftime(
                            "%Y-%m-%d"
                        ),
                        "three": row["three"],
                        "two": row["two"],
                        "updated_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
        )

    if operations:

        collection.bulk_write(
            operations,
            ordered=False
        )

    return True, f"บันทึก {len(operations)} รายการ"


# ================================================================
# LOAD MONGO
# ================================================================

def load_from_mongo(lottery):

    collection = get_mongo_collection()

    if collection is None:

        return None

    docs = list(
        collection.find(
            {
                "lottery": lottery
            },
            {
                "_id": 0
            }
        )
        .sort("date", 1)
    )

    if not docs:

        return None

    df = pd.DataFrame(docs)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["three"] = (
        df["three"]
        .astype(str)
        .str.zfill(3)
    )

    df["two"] = (
        df["two"]
        .astype(str)
        .str.zfill(2)
    )

    return (
        df[
            ["date", "three", "two"]
        ]
        .dropna()
        .drop_duplicates(
            "date"
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


# ================================================================
# FEATURE ENGINE
# ================================================================

def digit_features(
    series,
    idx,
    windows=(5, 10, 20)
):

    # IMPORTANT:
    # features use ONLY rows before idx

    hist = np.asarray(
        series[:idx],
        dtype=int
    )

    if len(hist) == 0:

        return None

    f = {}

    # ------------------------------------------------------------
    # Lag
    # ------------------------------------------------------------

    for lag in [1, 2, 3, 5, 7]:

        if len(hist) >= lag:

            f[f"lag_{lag}"] = int(
                hist[-lag]
            )

        else:

            f[f"lag_{lag}"] = -1


    # ------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------

    for w in windows:

        h = hist[-w:]

        counts = np.bincount(
            h,
            minlength=10
        )

        for d in range(10):

            f[f"freq_{w}_{d}"] = (
                counts[d] / max(len(h), 1)
            )


    # ------------------------------------------------------------
    # Last occurrence / GAP
    # ------------------------------------------------------------

    for d in range(10):

        pos = np.where(
            hist == d
        )[0]

        if len(pos):

            gap = (
                len(hist) - 1 - pos[-1]
            )

        else:

            gap = len(hist) + 5

        f[f"gap_{d}"] = gap


    # ------------------------------------------------------------
    # Recent transitions
    # ------------------------------------------------------------

    if len(hist) >= 2:

        f["delta_1"] = (
            int(hist[-1])
            -
            int(hist[-2])
        )

    else:

        f["delta_1"] = 0


    if len(hist) >= 3:

        f["delta_2"] = (
            int(hist[-1])
            -
            int(hist[-3])
        )

    else:

        f["delta_2"] = 0


    # ------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------

    for w in windows:

        h = hist[-w:]

        f[f"mean_{w}"] = float(
            np.mean(h)
        )

        f[f"std_{w}"] = float(
            np.std(h)
        )

        f[f"min_{w}"] = float(
            np.min(h)
        )

        f[f"max_{w}"] = float(
            np.max(h)
        )

        f[f"sum_{w}"] = float(
            np.sum(h)
        )


    # ------------------------------------------------------------
    # Digit parity
    # ------------------------------------------------------------

    f["last_even"] = (
        int(hist[-1] % 2 == 0)
    )

    f["last_high"] = (
        int(hist[-1] >= 5)
    )

    # ------------------------------------------------------------
    # Date features
    # ------------------------------------------------------------

    # idx based proxy
    f["index_mod_7"] = idx % 7
    f["index_mod_10"] = idx % 10

    return f


# ================================================================
# BUILD DATASET
# ================================================================

def build_position_dataset(
    series
):

    rows = []
    y = []

    for i in range(
        len(series)
    ):

        features = digit_features(
            series,
            i
        )

        if features is None:
            continue

        # Need sufficient history
        if i < 10:
            continue

        rows.append(features)

        y.append(
            int(series[i])
        )

    if not rows:

        return None, None

    X = pd.DataFrame(
        rows
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    y = np.asarray(
        y,
        dtype=int
    )

    return X, y


# ================================================================
# NEXT DRAW FEATURE
# ================================================================

def build_next_features(
    series
):

    features = digit_features(
        series,
        len(series)
    )

    X = pd.DataFrame(
        [features]
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return X


# ================================================================
# MODEL FACTORY
# ================================================================

def make_models():

    models = {

        "ExtraTrees":
            ExtraTreesClassifier(
                n_estimators=160,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            ),

        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.06,
                max_leaf_nodes=15,
                l2_regularization=0.5,
                random_state=42
            ),

        "GaussianNB":
            GaussianNB()
    }


    if use_xgb and XGB_AVAILABLE:

        models["XGBoost"] = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=2
        )

    return models


# ================================================================
# ALIGN PROBABILITY
# ================================================================

def aligned_probability(
    model,
    X
):

    p = model.predict_proba(X)

    result = np.zeros(
        (len(X), 10),
        dtype=float
    )

    classes = getattr(
        model,
        "classes_",
        []
    )

    for j, cls in enumerate(classes):

        try:

            d = int(cls)

            if 0 <= d <= 9:

                result[:, d] = p[:, j]

        except Exception:
            pass

    # normalize
    s = result.sum(
        axis=1,
        keepdims=True
    )

    result = np.divide(
        result,
        s,
        out=np.ones_like(result) / 10,
        where=s > 0
    )

    return result


# ================================================================
# TOP K
# ================================================================

def top_digits(
    probabilities,
    k=3
):

    order = np.argsort(
        probabilities
    )[::-1]

    return [
        int(x)
        for x in order[:k]
    ]


# ================================================================
# WALK FORWARD BACKTEST
# ================================================================

def walk_forward_backtest(
    series,
    max_tests=25,
    min_train=40
):

    series = np.asarray(
        series,
        dtype=int
    )

    n = len(series)

    if n < min_train + 5:

        return pd.DataFrame()

    start = max(
        min_train,
        n - max_tests
    )

    records = []

    models_names = list(
        make_models().keys()
    )

    for test_i in range(
        start,
        n
    ):

        # --------------------------------------------------------
        # TRAIN DATA
        # --------------------------------------------------------

        train_series = series[
            :test_i
        ]

        X_train, y_train = build_position_dataset(
            train_series
        )

        if X_train is None:
            continue

        actual = int(
            series[test_i]
        )

        # --------------------------------------------------------
        # Must have multiple classes
        # --------------------------------------------------------

        if len(
            np.unique(y_train)
        ) < 2:

            continue

        for model_name in models_names:

            models = make_models()

            model = models[
                model_name
            ]

            try:

                model.fit(
                    X_train,
                    y_train
                )

                X_test = build_next_features(
                    train_series
                )

                prob = aligned_probability(
                    model,
                    X_test
                )[0]

                top1 = top_digits(
                    prob,
                    1
                )

                top3 = top_digits(
                    prob,
                    3
                )

                records.append({
                    "test_index": test_i,
                    "model": model_name,
                    "actual": actual,
                    "top1": top1[0],
                    "top3": top3,
                    "hit_top1": int(
                        actual == top1[0]
                    ),
                    "hit_top3": int(
                        actual in top3
                    )
                })

            except Exception as e:

                records.append({
                    "test_index": test_i,
                    "model": model_name,
                    "actual": actual,
                    "top1": -1,
                    "top3": [],
                    "hit_top1": 0,
                    "hit_top3": 0
                })

    return pd.DataFrame(
        records
    )


# ================================================================
# ADAPTIVE WEIGHTS
# ================================================================

def calculate_adaptive_weights(
    bt
):

    if bt is None or bt.empty:

        return {
            name: 1.0
            for name in make_models()
        }


    scores = {}

    for model_name in bt["model"].unique():

        sub = bt[
            bt["model"] == model_name
        ]

        if len(sub) == 0:
            continue

        top3_acc = (
            sub["hit_top3"].mean()
        )

        top1_acc = (
            sub["hit_top1"].mean()
        )

        # top3 is primary
        score = (
            0.70 * top3_acc
            +
            0.30 * top1_acc
        )

        # small floor prevents zero weight
        scores[model_name] = max(
            float(score),
            0.01
        )


    if not scores:

        return {
            name: 1.0
            for name in make_models()
        }


    total = sum(
        scores.values()
    )

    weights = {
        k: v / total
        for k, v in scores.items()
    }

    return weights


# ================================================================
# ENSEMBLE PREDICTION
# ================================================================

def ensemble_predict(
    series,
    weights
):

    X_train, y_train = build_position_dataset(
        series
    )

    X_next = build_next_features(
        series
    )

    if X_train is None:

        return None

    if len(
        np.unique(y_train)
    ) < 2:

        return None

    models = make_models()

    final_prob = np.zeros(
        10,
        dtype=float
    )

    details = {}

    for name, model in models.items():

        if name not in weights:
            continue

        try:

            model.fit(
                X_train,
                y_train
            )

            prob = aligned_probability(
                model,
                X_next
            )[0]

            w = weights.get(
                name,
                0
            )

            final_prob += (
                w * prob
            )

            details[name] = {
                "weight": w,
                "prob": prob
            }

        except Exception as e:

            details[name] = {
                "weight": 0,
                "error": str(e)
            }


    # normalize
    total = final_prob.sum()

    if total > 0:

        final_prob /= total

    return {
        "prob": final_prob,
        "top3": top_digits(
            final_prob,
            3
        ),
        "top5": top_digits(
            final_prob,
            5
        ),
        "details": details
    }


# ================================================================
# POSITION ANALYSIS
# ================================================================

def analyze_position(
    series,
    position_name
):

    series = np.asarray(
        series,
        dtype=int
    )

    # ------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------

    bt = walk_forward_backtest(
        series,
        max_tests=backtest_n,
        min_train=min_history
    )

    weights = calculate_adaptive_weights(
        bt
    )

    prediction = ensemble_predict(
        series,
        weights
    )

    # ------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------

    counts = np.bincount(
        series,
        minlength=10
    )

    freq = (
        counts / len(series)
    )

    # ------------------------------------------------------------
    # Recent frequency
    # ------------------------------------------------------------

    recent = series[-20:]

    recent_counts = np.bincount(
        recent,
        minlength=10
    )

    recent_freq = (
        recent_counts
        /
        max(len(recent), 1)
    )

    # ------------------------------------------------------------
    # Gap
    # ------------------------------------------------------------

    gaps = {}

    for d in range(10):

        pos = np.where(
            series == d
        )[0]

        if len(pos):

            gaps[d] = (
                len(series)
                -
                1
                -
                pos[-1]
            )

        else:

            gaps[d] = len(series)


    # ------------------------------------------------------------
    # Probability table
    # ------------------------------------------------------------

    table = pd.DataFrame({
        "เลข": range(10),
        "ความถี่ทั้งหมด": np.round(
            freq * 100,
            2
        ),
        "ความถี่ 20 งวด": np.round(
            recent_freq * 100,
            2
        ),
        "Gap": [
            gaps[d]
            for d in range(10)
        ]
    })


    if prediction:

        table["AI Probability"] = np.round(
            prediction["prob"] * 100,
            2
        )

    else:

        table["AI Probability"] = 10.0


    table = table.sort_values(
        "AI Probability",
        ascending=False
    ).reset_index(
        drop=True
    )


    return {
        "position": position_name,
        "backtest": bt,
        "weights": weights,
        "prediction": prediction,
        "table": table
    }


# ================================================================
# FULL ANALYSIS
# ================================================================

def run_analysis(df):

    results = {}

    # ------------------------------------------------------------
    # 3 DIGIT
    # ------------------------------------------------------------

    three = (
        df["three"]
        .astype(str)
        .str.zfill(3)
    )

    for i in range(3):

        pos = f"3D-{i+1}"

        series = np.array([
            int(x[i])
            for x in three
        ])

        results[pos] = analyze_position(
            series,
            pos
        )


    # ------------------------------------------------------------
    # 2 DIGIT
    # ------------------------------------------------------------

    two = (
        df["two"]
        .astype(str)
        .str.zfill(2)
    )

    for i in range(2):

        pos = f"2D-{i+1}"

        series = np.array([
            int(x[i])
            for x in two
        ])

        results[pos] = analyze_position(
            series,
            pos
        )


    return results


# ================================================================
# DISPLAY BACKTEST
# ================================================================

def show_backtest(
    result
):

    bt = result["backtest"]

    if bt is None or bt.empty:

        st.warning(
            "ข้อมูลยังไม่พอสำหรับ Backtest"
        )

        return


    summary = (
        bt.groupby("model")
        .agg(
            งวด=("actual", "count"),
            TOP1=(
                "hit_top1",
                "mean"
            ),
            TOP3=(
                "hit_top3",
                "mean"
            )
        )
        .reset_index()
    )

    summary["TOP1"] = (
        summary["TOP1"] * 100
    ).round(2)

    summary["TOP3"] = (
        summary["TOP3"] * 100
    ).round(2)

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# DISPLAY PREDICTION
# ================================================================

def show_prediction(
    result
):

    pred = result["prediction"]

    if not pred:

        st.warning(
            "ไม่สามารถสร้าง Prediction ได้"
        )

        return

    top3 = pred["top3"]

    st.subheader(
        f"🎯 {result['position']}"
    )

    cols = st.columns(3)

    for i, digit in enumerate(top3):

        probability = (
            pred["prob"][digit]
            * 100
        )

        cols[i].metric(
            f"อันดับ {i+1}",
            str(digit),
            f"{probability:.2f}%"
        )

    st.write(
        "TOP-3:",
        " • ".join(
            str(x)
            for x in top3
        )
    )

    # model weights
    weights = result["weights"]

    weight_df = pd.DataFrame({
        "Model": list(
            weights.keys()
        ),
        "Adaptive Weight": [
            round(
                x * 100,
                2
            )
            for x in weights.values()
        ]
    })

    with st.expander(
        "⚖️ Adaptive Model Weights"
    ):

        st.dataframe(
            weight_df,
            use_container_width=True,
            hide_index=True
        )


# ================================================================
# DATA QUALITY
# ================================================================

def show_data_quality(
    df,
    debug
):

    st.subheader(
        "🔎 Data Quality"
    )

    ok, report = validate_data(
        df
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "จำนวนข้อมูล",
        report.get(
            "rows",
            0
        )
    )

    c2.metric(
        "วันที่เริ่ม",
        str(
            report.get(
                "date_min",
                "-"
            )
        )[:10]
    )

    c3.metric(
        "วันที่ล่าสุด",
        str(
            report.get(
                "date_max",
                "-"
            )
        )[:10]
    )

    c4.metric(
        "Duplicate",
        report.get(
            "duplicate_dates",
            0
        )
    )

    if ok:

        st.success(
            "✅ ข้อมูลผ่านการตรวจสอบ"
        )

    else:

        st.warning(
            "⚠️ ข้อมูลมีปัญหาบางส่วน"
        )

    if show_debug:

        with st.expander(
            "🛠️ Scraper Debug"
        ):

            st.json(
                {
                    k: str(v)
                    for k, v in debug.items()
                    if k != "sample"
                }
            )

            if "sample" in debug:

                st.text(
                    debug["sample"]
                )


# ================================================================
# LOAD DATA
# ================================================================

st.divider()

col1, col2, col3 = st.columns(3)


# ================================================================
# BUTTON: SCRAPE
# ================================================================

with col1:

    scrape_btn = st.button(
        "🌐 ดึงข้อมูลใหม่จากเว็บ",
        use_container_width=True
    )


# ================================================================
# BUTTON: MONGO
# ================================================================

with col2:

    mongo_btn = st.button(
        "🗄️ โหลดจาก MongoDB",
        use_container_width=True
    )


# ================================================================
# BUTTON: ANALYZE
# ================================================================

with col3:

    analyze_btn = st.button(
        "🚀 วิเคราะห์ AI",
        use_container_width=True
    )


# ================================================================
# SCRAPE
# ================================================================

if scrape_btn:

    with st.spinner(
        "กำลังดึงข้อมูลจาก Blogger..."
    ):

        df, debug = scrape_lottery(
            lottery_type
        )

        st.session_state.scraper_debug = debug

        if df is None:

            st.error(
                "❌ ดึงข้อมูลไม่สำเร็จ"
            )

            st.error(
                "ไม่พบข้อมูลรูปแบบ "
                "วันที่ | 3 ตัว | 2 ตัว"
            )

            if show_debug:

                st.json(
                    {
                        k: str(v)
                        for k, v
                        in debug.items()
                    }
                )

        else:

            st.session_state.data = df

            # save Mongo
            success, msg = save_to_mongo(
                df,
                lottery_type
            )

            if success:

                st.success(
                    f"✅ ดึงข้อมูลสำเร็จ "
                    f"{len(df):,} งวด | {msg}"
                )

            else:

                st.success(
                    f"✅ ดึงข้อมูลสำเร็จ "
                    f"{len(df):,} งวด"
                )

            show_data_quality(
                df,
                debug
            )


# ================================================================
# LOAD MONGO
# ================================================================

if mongo_btn:

    with st.spinner(
        "กำลังโหลด MongoDB..."
    ):

        df = load_from_mongo(
            lottery_type
        )

        if df is None:

            st.error(
                "❌ ไม่พบข้อมูลใน MongoDB"
            )

            st.info(
                "ให้กด 'ดึงข้อมูลใหม่จากเว็บ' ก่อน"
            )

        else:

            st.session_state.data = df

            st.success(
                f"✅ โหลด MongoDB สำเร็จ "
                f"{len(df):,} งวด"
            )

            show_data_quality(
                df,
                st.session_state.scraper_debug
            )


# ================================================================
# SHOW CURRENT DATA
# ================================================================

df = st.session_state.data

if df is not None:

    st.divider()

    st.subheader(
        f"📋 ข้อมูล {lottery_type}"
    )

    st.dataframe(
        df.tail(20)
        .sort_values(
            "date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# ANALYZE
# ================================================================

if analyze_btn:

    if df is None:

        st.error(
            "❌ ยังไม่มีข้อมูล "
            "กรุณาดึงข้อมูลจากเว็บหรือ MongoDB ก่อน"
        )

    elif len(df) < min_history:

        st.error(
            f"❌ ข้อมูลมี {len(df)} งวด "
            f"แต่กำหนดขั้นต่ำ {min_history} งวด"
        )

    else:

        with st.spinner(
            "กำลังทำ Walk-Forward + Adaptive AI..."
        ):

            results = run_analysis(
                df
            )

            st.session_state.prediction = results

        st.success(
            "✅ วิเคราะห์เสร็จแล้ว"
        )


# ================================================================
# DISPLAY RESULTS
# ================================================================

results = st.session_state.prediction

if results:

    st.divider()

    st.header(
        "🔮 ผลวิเคราะห์ TOP-3 ทุกหลัก"
    )

    # ------------------------------------------------------------
    # Prediction summary
    # ------------------------------------------------------------

    summary_rows = []

    for pos, result in results.items():

        pred = result["prediction"]

        if pred:

            summary_rows.append({

                "ตำแหน่ง":
                    pos,

                "TOP-1":
                    pred["top3"][0],

                "TOP-2":
                    pred["top3"][1],

                "TOP-3":
                    pred["top3"][2],

                "TOP-1 %":
                    round(
                        pred["prob"][
                            pred["top3"][0]
                        ] * 100,
                        2
                    ),

                "TOP-3 รวม %":
                    round(
                        sum(
                            pred["prob"][d]
                            for d in pred["top3"]
                        ) * 100,
                        2
                    )
            })


    summary_df = pd.DataFrame(
        summary_rows
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------------
    # Individual positions
    # ------------------------------------------------------------

    for pos, result in results.items():

        with st.expander(
            f"📌 {pos}",
            expanded=True
        ):

            show_prediction(
                result
            )

            st.markdown(
                "### 📊 สถิติเลข"
            )

            st.dataframe(
                result["table"],
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "### 🧪 Walk-Forward Backtest"
            )

            show_backtest(
                result
            )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "LOTTO AI V2.2 | Robust Scraper + "
    "Walk-Forward + Adaptive Ensemble + TOP-3"
    )
