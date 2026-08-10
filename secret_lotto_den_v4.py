# ============================================================
# PRO V7 AI-ONLY – Mobile Accuracy Edition
# AI Ensemble + Walk-Forward Backtest + Dynamic Weight
# Google Colab / GitHub Ready
# ============================================================

import os
import re
import glob
import hashlib
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests

from bs4 import BeautifulSoup
from joblib import Memory

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================
# 0. CONFIG / CACHE
# ============================================================

CACHE_DIR = "model_cache_v7"
MEMORY_DIR = "/tmp/lotto_memory_cache_v7"

os.makedirs(CACHE_DIR, exist_ok=True)

memory = Memory(
    location=MEMORY_DIR,
    verbose=0
)


# ============================================================
# 1. LOTTERY SOURCES
# ============================================================

LOTTERY_SOURCES = {
    "1. หวยไทย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",

    "2. หวยธกส.":
        "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",

    "3. หวยออมสิน":
        "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",

    "4. หวยลาว":
        "https://suksan18190.blogspot.com/2026/07/blog-post.html",

    "5. หวยฮานอย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",

    "6. หวยมาเลย์":
        "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",

    "7. หวยหุ้นไทยเย็น":
        "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",

    "8. หวยหุ้นนิเคอิบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",

    "9. หวยหุ้นฮั่งเส็งบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",

    "10. หวยหุ้นจีนบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}


# ============================================================
# 2. LOAD + CLEAN REAL DATA
# ============================================================

@memory.cache
def fetch_and_clean_data(url):
    """
    โหลดข้อมูลจริงจากเว็บไซต์
    ไม่มีการสร้างข้อมูลสุ่ม
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 10; K) "
            "AppleWebKit/537.36 "
            "Chrome/138.0 Mobile Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        main_content = soup.find(
            "div",
            class_=re.compile(
                r"post-body|entry-content|post-content|content"
            )
        )

        if main_content is None:
            main_content = soup

        lines = main_content.get_text(
            separator="\n"
        ).split("\n")

        date_pattern = re.compile(
            r"("
            r"\d{4}-\d{1,2}-\d{1,2}"
            r"|"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
            r")"
        )

        num_pattern = re.compile(
            r"\b(\d{3})\b.*?\b(\d{2})\b"
            r"|"
            r"\b(\d{5,6})\b.*?\b(\d{2})\b"
        )

        extracted = []
        current_date = None

        for line in lines:

            line = line.strip()

            if not line:
                continue

            date_match = date_pattern.search(line)

            if date_match:
                current_date = date_match.group(1)

            num_match = num_pattern.search(line)

            if (
                num_match is None
                or current_date is None
            ):
                continue

            if (
                num_match.group(1)
                and num_match.group(2)
            ):
                result_3d = num_match.group(1)
                result_2d = num_match.group(2)

            elif (
                num_match.group(3)
                and num_match.group(4)
            ):
                result_3d = num_match.group(3)[-3:]
                result_2d = num_match.group(4)

            else:
                continue

            extracted.append({
                "Date": current_date,
                "Result_3D": str(result_3d).zfill(3),
                "Result_2D": str(result_2d).zfill(2)
            })

        if len(extracted) < 30:
            raise ValueError(
                f"ข้อมูลจริงน้อยเกินไป: {len(extracted)} งวด"
            )

        df = pd.DataFrame(extracted)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Result_3D"] = (
            df["Result_3D"]
            .astype(str)
            .str.extract(r"(\d{3})")[0]
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(r"(\d{2})")[0]
        )

        df = df.dropna(
            subset=[
                "Date",
                "Result_3D",
                "Result_2D"
            ]
        )

        df = df[
            df["Result_3D"].str.fullmatch(r"\d{3}")
            & df["Result_2D"].str.fullmatch(r"\d{2}")
        ]

        df = (
            df
            .drop_duplicates(
                subset=[
                    "Date",
                    "Result_3D",
                    "Result_2D"
                ]
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if len(df) < 30:
            raise ValueError(
                f"หลังทำความสะอาดเหลือเพียง {len(df)} งวด"
            )

        return df

    except Exception as e:

        raise RuntimeError(
            f"โหลดข้อมูลจริงไม่ได้: {e}"
        )


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

@memory.cache
def build_features(df, lags, rolls):

    df_feat = df.copy()

    # --------------------------------------------------------
    # Digits
    # --------------------------------------------------------

    digit_map = [
        ("H", "Result_3D", 0),
        ("T", "Result_3D", 1),
        ("O", "Result_3D", 2),
        ("T2", "Result_2D", 0),
        ("O2", "Result_2D", 1)
    ]

    for col, src, idx in digit_map:

        df_feat[col] = (
            df_feat[src]
            .astype(str)
            .str[idx]
            .astype(int)
        )

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    df_feat["DayOfWeek"] = (
        df_feat["Date"].dt.dayofweek
    )

    df_feat["Month"] = (
        df_feat["Date"].dt.month
    )

    df_feat["Day"] = (
        df_feat["Date"].dt.day
    )

    df_feat["WeekOfYear"] = (
        df_feat["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df_feat["DayOfYear"] = (
        df_feat["Date"].dt.dayofyear
    )

    df_feat["DrawIndex"] = (
        np.arange(len(df_feat))
    )

    df_feat["Gap"] = (
        df_feat["Date"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(0, 60)
    )

    # --------------------------------------------------------
    # Cyclical Calendar
    # --------------------------------------------------------

    df_feat["sin_wd"] = np.sin(
        2 * np.pi *
        df_feat["DayOfWeek"] / 7
    )

    df_feat["cos_wd"] = np.cos(
        2 * np.pi *
        df_feat["DayOfWeek"] / 7
    )

    df_feat["sin_month"] = np.sin(
        2 * np.pi *
        df_feat["Month"] / 12
    )

    df_feat["cos_month"] = np.cos(
        2 * np.pi *
        df_feat["Month"] / 12
    )

    # --------------------------------------------------------
    # Per-position Features
    # --------------------------------------------------------

    positions = [
        "H",
        "T",
        "O",
        "T2",
        "O2"
    ]

    for pos in positions:

        prev = df_feat[pos].shift(1)

        # Basic
        df_feat[f"Odd_{pos}"] = (
            prev % 2
        ).fillna(0)

        df_feat[f"High_{pos}"] = (
            prev >= 5
        ).fillna(0).astype(int)

        df_feat[f"Mirror_{pos}"] = (
            (prev + 5) % 10
        ).fillna(0)

        # Lags
        for lag in lags:

            df_feat[
                f"Lag_{lag}_{pos}"
            ] = df_feat[pos].shift(lag)

        # Differences
        df_feat[
            f"Diff12_{pos}"
        ] = (
            df_feat[pos].shift(1)
            -
            df_feat[pos].shift(2)
        )

        df_feat[
            f"Diff23_{pos}"
        ] = (
            df_feat[pos].shift(2)
            -
            df_feat[pos].shift(3)
        )

        # Rolling
        shifted = df_feat[pos].shift(1)

        for w in rolls:

            df_feat[
                f"Mean_{w}_{pos}"
            ] = shifted.rolling(w).mean()

            df_feat[
                f"Std_{w}_{pos}"
            ] = shifted.rolling(w).std()

            df_feat[
                f"Min_{w}_{pos}"
            ] = shifted.rolling(w).min()

            df_feat[
                f"Max_{w}_{pos}"
            ] = shifted.rolling(w).max()

        # Hot counts
        for d in range(10):

            df_feat[
                f"Hot20_{pos}_{d}"
            ] = (
                shifted.eq(d)
                .rolling(20)
                .sum()
            )

        # Skip
        values = (
            df_feat[pos]
            .astype(int)
            .values
        )

        last_seen = np.full(
            10,
            -1,
            dtype=int
        )

        skips = np.zeros(
            len(values),
            dtype=float
        )

        for i in range(len(values)):

            if i == 0:
                skips[i] = 10

            else:

                previous = int(
                    values[i - 1]
                )

                if last_seen[previous] >= 0:

                    skips[i] = (
                        i - 1
                        -
                        last_seen[previous]
                    )

                else:

                    skips[i] = i

            current = int(values[i])

            last_seen[current] = i

        df_feat[
            f"Skip_{pos}"
        ] = np.clip(
            skips,
            0,
            35
        )

        # Repeat
        if (
            f"Lag_1_{pos}" in df_feat.columns
            and
            f"Lag_2_{pos}" in df_feat.columns
        ):

            df_feat[
                f"Repeat_{pos}"
            ] = (
                df_feat[f"Lag_1_{pos}"]
                ==
                df_feat[f"Lag_2_{pos}"]
            ).astype(int)

    # --------------------------------------------------------
    # Cross-position Features
    # --------------------------------------------------------

    h = df_feat["H"].shift(1)
    t = df_feat["T"].shift(1)
    o = df_feat["O"].shift(1)

    df_feat["DigitSum"] = (
        h + t + o
    ) % 10

    df_feat["DigitRange"] = (
        pd.concat(
            [h, t, o],
            axis=1
        ).max(axis=1)
        -
        pd.concat(
            [h, t, o],
            axis=1
        ).min(axis=1)
    )

    df_feat["HT_Diff"] = h - t
    df_feat["TO_Diff"] = t - o
    df_feat["HO_Diff"] = h - o

    df_feat["OddCount"] = (
        (h % 2)
        +
        (t % 2)
        +
        (o % 2)
    )

    df_feat["HighCount"] = (
        (h >= 5).astype(int)
        +
        (t >= 5).astype(int)
        +
        (o >= 5).astype(int)
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    df_feat = (
        df_feat
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(-1)
    )

    return df_feat


# ============================================================
# 4. AI ENSEMBLE
# ============================================================

class AIEnsemble:

    def __init__(
        self,
        lottery_id,
        use_xgb=True
    ):

        self.lottery_id = lottery_id
        self.use_xgb = (
            use_xgb and XGB_AVAILABLE
        )

    def create_models(self, trees):

        models = {

            "RF":
                RandomForestClassifier(
                    n_estimators=max(
                        40,
                        trees
                    ),
                    max_depth=6,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=42
                ),

            "ET":
                ExtraTreesClassifier(
                    n_estimators=max(
                        50,
                        trees
                    ),
                    max_depth=6,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=43
                ),

            "HGB":
                HistGradientBoostingClassifier(
                    max_iter=70,
                    learning_rate=0.035,
                    max_leaf_nodes=15,
                    min_samples_leaf=5,
                    l2_regularization=0.7,
                    random_state=44
                )
        }

        if self.use_xgb:

            models["XGB"] = XGBClassifier(
                n_estimators=max(
                    50,
                    trees
                ),
                max_depth=3,
                learning_rate=0.035,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.15,
                reg_lambda=1.5,
                tree_method="hist",
                eval_metric="mlogloss",
                verbosity=0,
                n_jobs=1,
                random_state=45
            )

        return models

    @staticmethod
    def align_probs(
        model,
        probs
    ):

        result = np.zeros(10)

        for c, p in zip(
            model.classes_,
            probs
        ):

            result[int(c)] = p

        return result

    @staticmethod
    def temperature_scale(
        probs,
        temperature=1.20
    ):

        probs = np.asarray(
            probs,
            dtype=float
        )

        probs = np.maximum(
            probs,
            1e-12
        )

        logits = (
            np.log(probs)
            /
            temperature
        )

        logits -= logits.max()

        scaled = np.exp(logits)

        scaled += 0.001

        return (
            scaled /
            scaled.sum()
        )


# ============================================================
# 5. PRO V7 AI-ONLY ENGINE
# ============================================================

class PROV7AIOnly:

    def __init__(
        self,
        df_raw,
        lottery_name,
        target_dow=None
    ):

        self.df_raw = (
            df_raw.copy()
        )

        self.lottery_name = (
            lottery_name
        )

        self.lottery_id = (
            lottery_name
            .split(".")[0]
            .strip()
        )

        self.target_dow = target_dow

        n = len(
            self.df_raw
        )

        # ----------------------------------------------------
        # Dynamic Mode
        # ----------------------------------------------------

        if n >= 700:

            self.mode_name = (
                "Mode 4 | 700+ "
                "| PRO V7 AI-ONLY"
            )

            self.trees = 120
            self.test_size = 45

            self.lags = [
                1, 2, 3, 4,
                5, 7, 8, 13
            ]

            self.rolls = [
                10, 20, 50
            ]

            self.use_xgb = True

        elif n >= 400:

            self.mode_name = (
                "Mode 3 | 400-699 "
                "| PRO V7 AI-ONLY"
            )

            self.trees = 100
            self.test_size = 35

            self.lags = [
                1, 2, 3,
                4, 5, 7, 8
            ]

            self.rolls = [
                10, 20, 50
            ]

            self.use_xgb = True

        elif n >= 200:

            self.mode_name = (
                "Mode 2 | 200-399 "
                "| PRO V7 AI-ONLY"
            )

            self.trees = 80
            self.test_size = 25

            self.lags = [
                1, 2, 3,
                4, 5, 8
            ]

            self.rolls = [
                10, 20
            ]

            self.use_xgb = False

        else:

            self.mode_name = (
                "Mode 1 | 100-199 "
                "| PRO V7 AI-ONLY"
            )

            self.trees = 60
            self.test_size = 20

            self.lags = [
                1, 2, 3, 5
            ]

            self.rolls = [
                10, 20
            ]

            self.use_xgb = False

        # ----------------------------------------------------
        # Disable XGB automatically if unavailable
        # ----------------------------------------------------

        if self.use_xgb and not XGB_AVAILABLE:

            self.use_xgb = False

            self.mode_name += (
                " | XGB OFF"
            )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        self.features = [

            "DayOfWeek",
            "Month",
            "Day",
            "WeekOfYear",
            "DayOfYear",
            "DrawIndex",
            "Gap",

            "sin_wd",
            "cos_wd",
            "sin_month",
            "cos_month",

            "DigitSum",
            "DigitRange",
            "HT_Diff",
            "TO_Diff",
            "HO_Diff",
            "OddCount",
            "HighCount"
        ]

        positions = [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]

        for pos in positions:

            self.features.extend([
                f"Odd_{pos}",
                f"High_{pos}",
                f"Mirror_{pos}",
                f"Diff12_{pos}",
                f"Diff23_{pos}",
                f"Skip_{pos}",
                f"Repeat_{pos}"
            ])

            for lag in self.lags:

                self.features.append(
                    f"Lag_{lag}_{pos}"
                )

            for w in self.rolls:

                self.features.extend([
                    f"Mean_{w}_{pos}",
                    f"Std_{w}_{pos}",
                    f"Min_{w}_{pos}",
                    f"Max_{w}_{pos}"
                ])

            for d in range(10):

                self.features.append(
                    f"Hot20_{pos}_{d}"
                )

        # ----------------------------------------------------
        # Data Hash
        # ----------------------------------------------------

        hash_array = (
            pd.util
            .hash_pandas_object(
                self.df_raw[
                    [
                        "Date",
                        "Result_3D",
                        "Result_2D"
                    ]
                ],
                index=False
            )
            .values
        )

        base_hash = hashlib.md5(
            hash_array.tobytes()
        ).hexdigest()

        self.data_hash = (
            f"{base_hash}_"
            f"{self.trees}_"
            f"{self.test_size}_"
            f"{len(self.features)}_"
            f"{int(self.use_xgb)}"
        )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        self.ai = AIEnsemble(
            self.lottery_id,
            self.use_xgb
        )

        # ----------------------------------------------------
        # Default Weights
        # ----------------------------------------------------

        self.default_weights = {
            "RF": 0.25,
            "ET": 0.35,
            "HGB": 0.25,
            "XGB": 0.15
        }

        if not self.use_xgb:

            self.default_weights = {
                "RF": 0.30,
                "ET": 0.40,
                "HGB": 0.30
            }

    # ========================================================
    # 6. WALK-FORWARD BACKTEST
    # ========================================================

    def backtest_position(
        self,
        pos,
        X_all,
        df_hist
    ):

        bt_size = min(
            self.test_size,
            len(df_hist) - 40
        )

        if bt_size < 8:

            return (
                self.default_weights.copy(),
                "Backtest: ข้อมูลไม่เพียงพอ"
            )

        start = (
            len(df_hist)
            -
            bt_size
        )

        models = list(
            self.default_weights.keys()
        )

        scores = {
            m: {
                "top1": 0.0,
                "top3": 0.0,
                "top5": 0.0,
                "loss": 0.0
            }
            for m in models
        }

        total_weight = 0.0

        for step, i in enumerate(
            range(
                start,
                len(df_hist)
            )
        ):

            X_train = X_all.iloc[:i]

            y_train = (
                df_hist[pos]
                .iloc[:i]
            )

            X_test = (
                X_all.iloc[[i]]
            )

            actual = int(
                df_hist[pos]
                .iloc[i]
            )

            recency = (
                0.70
                +
                0.30 *
                (
                    (step + 1)
                    /
                    bt_size
                )
            )

            total_weight += recency

            current_models = (
                self.ai.create_models(
                    self.trees
                )
            )

            for name in models:

                model = (
                    current_models[name]
                )

                try:

                    model.fit(
                        X_train,
                        y_train
                    )

                    raw = (
                        model
                        .predict_proba(
                            X_test
                        )[0]
                    )

                    prob = (
                        self.ai
                        .align_probs(
                            model,
                            raw
                        )
                    )

                    prob = (
                        self.ai
                        .temperature_scale(
                            prob,
                            1.20
                        )
                    )

                    ranking = (
                        np.argsort(
                            prob
                        )[::-1]
                    )

                    rank = np.where(
                        ranking == actual
                    )[0][0]

                    if rank == 0:

                        scores[name][
                            "top1"
                        ] += recency

                    if rank < 3:

                        scores[name][
                            "top3"
                        ] += recency

                    if rank < 5:

                        scores[name][
                            "top5"
                        ] += recency

                    scores[name][
                        "loss"
                    ] += (
                        -np.log(
                            max(
                                prob[actual],
                                1e-9
                            )
                        )
                        *
                        recency
                    )

                except Exception:

                    continue

        if total_weight <= 0:

            return (
                self.default_weights.copy(),
                "Backtest ล้มเหลว"
            )

        raw_weights = {}
        report = []

        for name in models:

            top1 = (
                scores[name]["top1"]
                /
                total_weight
            )

            top3 = (
                scores[name]["top3"]
                /
                total_weight
            )

            top5 = (
                scores[name]["top5"]
                /
                total_weight
            )

            loss = (
                scores[name]["loss"]
                /
                total_weight
            )

            # Top-5 สำคัญที่สุด
            accuracy_score = (
                top1 * 0.20
                +
                top3 * 0.30
                +
                top5 * 0.50
            )

            loss_score = np.exp(
                -0.25 * loss
            )

            score = (
                accuracy_score
                *
                loss_score
            )

            raw_weights[name] = max(
                score,
                0.01
            )

            report.append(
                f"{name}: "
                f"T1={top1:.0%} "
                f"T3={top3:.0%} "
                f"T5={top5:.0%} "
                f"LL={loss:.2f}"
            )

        total = sum(
            raw_weights.values()
        )

        weights = {
            k: v / total
            for k, v in raw_weights.items()
        }

        msg = (
            f"BT {bt_size} งวด | "
            +
            " | ".join(report)
        )

        return (
            weights,
            msg
        )

    # ========================================================
    # 7. TRAIN FINAL AI
    # ========================================================

    def predict_position(
        self,
        pos,
        df_hist,
        X_all,
        next_x
    ):

        weight_path = (
            f"{CACHE_DIR}/"
            f"weight_"
            f"{self.lottery_id}_"
            f"{pos}_"
            f"{self.data_hash}.joblib"
        )

        if os.path.exists(
            weight_path
        ):

            weights, bt_msg = (
                joblib.load(
                    weight_path
                )
            )

        else:

            (
                weights,
                bt_msg
            ) = self.backtest_position(
                pos,
                X_all,
                df_hist
            )

            joblib.dump(
                (
                    weights,
                    bt_msg
                ),
                weight_path
            )

        model_path = (
            f"{CACHE_DIR}/"
            f"models_"
            f"{self.lottery_id}_"
            f"{pos}_"
            f"{self.data_hash}.joblib"
        )

        if os.path.exists(
            model_path
        ):

            trained_models = (
                joblib.load(
                    model_path
                )
            )

        else:

            trained_models = {}

            model_set = (
                self.ai.create_models(
                    self.trees
                )
            )

            for name, model in (
                model_set.items()
            ):

                model.fit(
                    X_all,
                    df_hist[pos]
                )

                trained_models[name] = (
                    model
                )

            joblib.dump(
                trained_models,
                model_path
            )

        final_prob = np.zeros(10)

        individual = {}

        for name, model in (
            trained_models.items()
        ):

            raw = (
                model
                .predict_proba(
                    next_x
                )[0]
            )

            prob = (
                self.ai
                .align_probs(
                    model,
                    raw
                )
            )

            prob = (
                self.ai
                .temperature_scale(
                    prob,
                    1.20
                )
            )

            individual[name] = prob

            final_prob += (
                weights[name]
                *
                prob
            )

        final_prob += 1e-9

        final_prob /= (
            final_prob.sum()
        )

        ranking = (
            np.argsort(
                final_prob
            )[::-1]
        )

        top5 = [
            (
                int(d),
                float(
                    final_prob[d]
                )
            )
            for d in ranking[:5]
        ]

        top3 = [
            int(d)
            for d in ranking[:3]
        ]

        return {

            "AI_Ensemble": top5,

            "Top3": top3,

            "Probability": final_prob,

            "Individual": {

                name: sorted(
                    [
                        (
                            i,
                            float(
                                prob[i]
                            )
                        )
                        for i in range(10)
                    ],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                for name, prob
                in individual.items()
            },

            "Weights": weights,

            "BT_Msg": bt_msg
        }

    # ========================================================
    # 8. PREDICT ALL POSITIONS
    # ========================================================

    def predict_all(self):

        if len(self.df_raw) < 40:

            raise ValueError(
                "ข้อมูลต้องมีอย่างน้อย 40 งวด"
            )

        last_date = (
            self.df_raw[
                "Date"
            ].iloc[-1]
        )

        if self.target_dow is not None:

            days_ahead = (
                self.target_dow
                -
                last_date.dayofweek
            )

            if days_ahead <= 0:
                days_ahead += 7

            next_date = (
                last_date
                +
                timedelta(
                    days=days_ahead
                )
            )

        else:

            if len(
                self.df_raw
            ) <= 1:

                gap = 7

            else:

                gap = (
                    self.df_raw[
                        "Date"
                    ].iloc[-1]
                    -
                    self.df_raw[
                        "Date"
                    ].iloc[-2]
                ).days

                gap = max(
                    1,
                    min(
                        gap,
                        30
                    )
                )

            next_date = (
                last_date
                +
                timedelta(
                    days=gap
                )
            )

        # ----------------------------------------------------
        # Dummy row for future feature generation only
        # ----------------------------------------------------

        dummy = pd.DataFrame([
            {
                "Date": next_date,
                "Result_3D": "000",
                "Result_2D": "00"
            }
        ])

        df_ext = pd.concat(
            [
                self.df_raw,
                dummy
            ],
            ignore_index=True
        )

        df_ext = build_features(
            df_ext,
            self.lags,
            self.rolls
        )

        df_hist = (
            df_ext
            .iloc[:-1]
            .copy()
        )

        next_x = (
            df_ext
            .iloc[[-1]]
            [self.features]
        )

        X_all = (
            df_hist[
                self.features
            ]
        )

        predictions = {}

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            predictions[pos] = (
                self.predict_position(
                    pos,
                    df_hist,
                    X_all,
                    next_x
                )
            )

        return (
            predictions,
            next_date
        )


# ============================================================
# 9. RESULT FORMATTER
# ============================================================

def format_v7_result(
    predictions,
    next_date
):

    result = {

        "NextDate": next_date,

        "Positions": {}
    }

    for pos, data in (
        predictions.items()
    ):

        result["Positions"][pos] = {

            "Top5":
                data["AI_Ensemble"],

            "Top3":
                data["Top3"],

            "Weights":
                data["Weights"],

            "Backtest":
                data["BT_Msg"]
        }

    return result


# ============================================================
# 10. SIMPLE RUNNER
# ============================================================

def run_pro_v7(
    lottery_name
):

    if lottery_name not in LOTTERY_SOURCES:

        raise ValueError(
            "ไม่พบชื่อหวยใน LOTTERY_SOURCES"
        )

    url = (
        LOTTERY_SOURCES[
            lottery_name
        ]
    )

    print("=" * 60)
    print("PRO V7 AI-ONLY")
    print("Mobile Accuracy Edition")
    print("=" * 60)

    print(
        f"กำลังโหลด: {lottery_name}"
    )

    df = fetch_and_clean_data(
        url
    )

    print(
        f"ข้อมูลจริง: {len(df)} งวด"
    )

    engine = PROV7AIOnly(
        df_raw=df,
        lottery_name=lottery_name
    )

    print(
        f"Mode: {engine.mode_name}"
    )

    print(
        f"Trees: {engine.trees}"
    )

    print(
        f"Backtest: {engine.test_size}"
    )

    print(
        f"XGB: {'ON' if engine.use_xgb else 'OFF'}"
    )

    predictions, next_date = (
        engine.predict_all()
    )

    result = format_v7_result(
        predictions,
        next_date
    )

    print("=" * 60)
    print(
        f"งวดถัดไป: "
        f"{next_date.strftime('%Y-%m-%d')}"
    )
    print("=" * 60)

    for pos, data in (
        predictions.items()
    ):

        print(
            f"\n[{pos}]"
        )

        print(
            "Top 5:"
        )

        for digit, prob in (
            data["AI_Ensemble"]
        ):

            print(
                f"  {digit} "
                f"= {prob:.2%}"
            )

        print(
            "Top 3:",
            data["Top3"]
        )

        print(
            "Weights:",
            {
                k: round(v, 3)
                for k, v
                in data["Weights"].items()
            }
        )

        print(
            data["BT_Msg"]
        )

    return result
