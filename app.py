# ============================================================
# 🧠 LOTTO AI
# AUTO SYMBOLIC EQUATION V2
# POSITION SEARCH ENGINE
# ============================================================
#
# ค้นหาสูตรอัตโนมัติแบบ "แยกทีละหลัก"
#
# หลักร้อย  -> สูตรที่ดีที่สุด
# หลักสิบ   -> สูตรที่ดีที่สุด
# หลักหน่วย -> สูตรที่ดีที่สุด
#
# INPUT
#   3D = เลข 3 ตัว
#   2D = เลข 2 ตัว
#
# FEATURES
#   L1 / L2 / L3 / L5
#   H / T / O
#   T2 / O2
#   SUM3 / SUM2
#   ABS
#   + - * /
#   MOD 10
#   MOD 9
#   DIGIT SUM
#   CROSS POSITION
#
# VALIDATION
#   Walk Forward
#   Recent 10
#   Stability
#   Overfit Penalty
#
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import itertools
import re
import math
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Lotto AI Symbolic Equation V2",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 LOTTO AI — AUTO SYMBOLIC EQUATION V2")

st.markdown("""
### Position Search Engine

ระบบจะค้นหาสมการแบบอัตโนมัติจากข้อมูลย้อนหลัง

**เลข 3 หลัก**
- H = หลักร้อย
- T = หลักสิบ
- O = หลักหน่วย

**เลข 2 หลัก**
- T2 = หลักสิบ
- O2 = หลักหน่วย

แล้วค้นหาสูตรแยกเป็น

`หลักร้อย → สูตรของหลักร้อย`

`หลักสิบ → สูตรของหลักสิบ`

`หลักหน่วย → สูตรของหลักหน่วย`
""")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ SETTINGS")

min_history = st.sidebar.slider(
    "จำนวนงวดขั้นต่ำก่อนเริ่มค้นหา",
    20,
    200,
    40
)

top_n = st.sidebar.slider(
    "จำนวนสูตรที่เก็บต่อหลัก",
    3,
    30,
    10
)

recent_window = st.sidebar.slider(
    "Recent Window",
    5,
    30,
    10
)

max_formulas = st.sidebar.slider(
    "จำนวนสูตรสูงสุดที่จะทดสอบ",
    500,
    20000,
    5000,
    step=500
)

st.sidebar.markdown("---")

st.sidebar.info(
    "ระบบนี้ใช้การค้นหาสมการจากข้อมูลย้อนหลัง "
    "และ Walk-forward validation "
    "เพื่อช่วยลดการจำข้อมูลย้อนหลังมากเกินไป"
)


# ============================================================
# HELPERS
# ============================================================

def clean_number(x, width):

    if pd.isna(x):
        return None

    s = str(x).strip()

    s = re.sub(r"\.0$", "", s)

    digits = re.sub(r"\D", "", s)

    if digits == "":
        return None

    return digits.zfill(width)[-width:]


def digit_sum(x):

    s = str(int(x)).zfill(3)

    return sum(int(c) for c in s)


def reverse_number(x, width):

    return int(
        str(int(x)).zfill(width)[::-1]
    )


def safe_div(a, b):

    if abs(b) < 1e-12:
        return None

    return a / b


def mod10(x):

    if x is None:
        return None

    return int(round(x)) % 10


def mod9(x):

    if x is None:
        return None

    return int(round(x)) % 9


# ============================================================
# LOAD DATA
# ============================================================

st.header("📥 1. โหลดข้อมูล")

uploaded = st.file_uploader(
    "อัปโหลด CSV",
    type=["csv"]
)

df = None


if uploaded is not None:

    try:

        df = pd.read_csv(
            uploaded,
            dtype=str
        )

        st.success(
            f"โหลดข้อมูลสำเร็จ {len(df):,} งวด"
        )

        st.write(
            "Columns:",
            list(df.columns)
        )

    except Exception as e:

        st.error(
            f"อ่านไฟล์ไม่ได้: {e}"
        )


# ============================================================
# MANUAL DATA
# ============================================================

st.subheader("หรือกรอกข้อมูลเอง")

manual_text = st.text_area(
    "รูปแบบ: 3D,2D ต่อหนึ่งงวด",
    value="",
    height=120,
    placeholder="""615,53
222,04
381,21
742,72"""
)


if manual_text.strip():

    rows = []

    for line in manual_text.strip().splitlines():

        parts = re.split(
            r"[,;\\s]+",
            line.strip()
        )

        if len(parts) >= 2:

            rows.append({
                "3D": parts[0],
                "2D": parts[1]
            })

    if rows:

        df = pd.DataFrame(rows)

        st.success(
            f"รับข้อมูล {len(df):,} งวด"
        )


# ============================================================
# AUTO DETECT COLUMNS
# ============================================================

def find_column(columns, candidates):

    lower_map = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    # fuzzy

    for c in columns:

        cl = str(c).lower()

        for candidate in candidates:

            if candidate.lower() in cl:

                return c

    return None


if df is not None:

    col3 = find_column(
        df.columns,
        [
            "3d",
            "3D",
            "result3",
            "three",
            "three_digit",
            "เลข3ตัว",
            "สามตัว"
        ]
    )

    col2 = find_column(
        df.columns,
        [
            "2d",
            "2D",
            "result2",
            "two",
            "two_digit",
            "เลข2ตัว",
            "สองตัว"
        ]
    )

    if col3 is None or col2 is None:

        st.warning(
            "ไม่สามารถหา column 3D / 2D ได้อัตโนมัติ"
        )

        c1, c2 = st.columns(2)

        with c1:

            col3 = st.selectbox(
                "เลือก Column เลข 3 ตัว",
                df.columns
            )

        with c2:

            col2 = st.selectbox(
                "เลือก Column เลข 2 ตัว",
                df.columns
            )

    data = pd.DataFrame()

    data["3D"] = df[col3].apply(
        lambda x: clean_number(x, 3)
    )

    data["2D"] = df[col2].apply(
        lambda x: clean_number(x, 2)
    )

    data = data.dropna().reset_index(
        drop=True
    )

else:

    data = None


# ============================================================
# SHOW DATA
# ============================================================

if data is not None:

    st.subheader("ข้อมูลที่ใช้")

    st.dataframe(
        data.tail(20),
        use_container_width=True
    )


# ============================================================
# CREATE RAW VARIABLES
# ============================================================

def make_raw_variables(row):

    n3 = str(row["3D"]).zfill(3)
    n2 = str(row["2D"]).zfill(2)

    H = int(n3[0])
    T = int(n3[1])
    O = int(n3[2])

    T2 = int(n2[0])
    O2 = int(n2[1])

    S3 = H + T + O
    S2 = T2 + O2

    return {

        "H": H,
        "T": T,
        "O": O,

        "T2": T2,
        "O2": O2,

        "S3": S3,
        "S2": S2,

        "HT": abs(H - T),
        "TO": abs(T - O),
        "HO": abs(H - O),

        "HT2": abs(H - T2),
        "HO2": abs(H - O2),

        "TT2": abs(T - T2),
        "TO2": abs(T - O2),

        "R3": reverse_number(
            n3,
            3
        ),

        "R2": reverse_number(
            n2,
            2
        ),

        "DS3": digit_sum(n3),

        "D2": T2 + O2
    }


# ============================================================
# BUILD LAG FEATURES
# ============================================================

def build_features(data):

    records = []

    for i in range(len(data)):

        record = {}

        for lag in [1, 2, 3, 5]:

            idx = i - lag

            if idx >= 0:

                values = make_raw_variables(
                    data.iloc[idx]
                )

                for key, value in values.items():

                    record[
                        f"{key}_L{lag}"
                    ] = value

        records.append(record)

    return pd.DataFrame(
        records
    ).fillna(0)


# ============================================================
# TARGET DIGITS
# ============================================================

def target_digit(data, i, position):

    s = str(
        data.iloc[i]["3D"]
    ).zfill(3)

    if position == "H":
        return int(s[0])

    if position == "T":
        return int(s[1])

    return int(s[2])


# ============================================================
# FORMULA OBJECT
# ============================================================

class Formula:

    def __init__(
        self,
        name,
        func
    ):

        self.name = name
        self.func = func

    def calculate(self, row):

        try:

            value = self.func(row)

            if value is None:
                return None

            if not np.isfinite(value):
                return None

            return int(round(value)) % 10

        except:

            return None


# ============================================================
# FORMULA GENERATOR
# ============================================================

def generate_formulas():

    formulas = []

    # --------------------------------------------------------
    # BASE VARIABLES
    # --------------------------------------------------------

    base = []

    for lag in [1, 2, 3, 5]:

        for name in [

            "H",
            "T",
            "O",

            "T2",
            "O2",

            "S3",
            "S2",

            "HT",
            "TO",
            "HO",

            "HT2",
            "HO2",

            "TT2",
            "TO2",

            "R3",
            "R2",

            "DS3",
            "D2"

        ]:

            base.append(
                f"{name}_L{lag}"
            )

    # --------------------------------------------------------
    # SINGLE
    # --------------------------------------------------------

    for a in base:

        formulas.append(
            Formula(
                a,
                lambda row, a=a:
                    row.get(a, 0)
            )
        )

    # --------------------------------------------------------
    # BINARY
    # --------------------------------------------------------

    for a, b in itertools.combinations(
        base,
        2
    ):

        formulas.append(
            Formula(
                f"({a}+{b})",
                lambda row, a=a, b=b:
                    row.get(a, 0)
                    +
                    row.get(b, 0)
            )
        )

        formulas.append(
            Formula(
                f"({a}-{b})",
                lambda row, a=a, b=b:
                    row.get(a, 0)
                    -
                    row.get(b, 0)
            )
        )

        formulas.append(
            Formula(
                f"ABS({a}-{b})",
                lambda row, a=a, b=b:
                    abs(
                        row.get(a, 0)
                        -
                        row.get(b, 0)
                    )
            )
        )

        formulas.append(
            Formula(
                f"({a}*{b})",
                lambda row, a=a, b=b:
                    row.get(a, 0)
                    *
                    row.get(b, 0)
            )
        )

        formulas.append(
            Formula(
                f"MOD9({a}+{b})",
                lambda row, a=a, b=b:
                    mod9(
                        row.get(a, 0)
                        +
                        row.get(b, 0)
                    )
            )
        )

        formulas.append(
            Formula(
                f"MOD10({a}+{b})",
                lambda row, a=a, b=b:
                    mod10(
                        row.get(a, 0)
                        +
                        row.get(b, 0)
                    )
            )
        )

        formulas.append(
            Formula(
                f"MOD10({a}*{b})",
                lambda row, a=a, b=b:
                    mod10(
                        row.get(a, 0)
                        *
                        row.get(b, 0)
                    )
            )
        )

        formulas.append(
            Formula(
                f"({a}/{b})",
                lambda row, a=a, b=b:
                    safe_div(
                        row.get(a, 0),
                        row.get(b, 0)
                    )
            )
        )

    # --------------------------------------------------------
    # TRIPLE
    # --------------------------------------------------------

    important = [
        x for x in base
        if (
            x.startswith("H_")
            or x.startswith("T_")
            or x.startswith("O_")
            or x.startswith("T2_")
            or x.startswith("O2_")
        )
    ]

    # จำกัดจำนวนเพื่อให้เร็ว
    important = important[:24]

    for a, b, c in itertools.combinations(
        important,
        3
    ):

        formulas.append(
            Formula(
                f"(({a}+{b})+{c})",
                lambda row, a=a, b=b, c=c:
                    row.get(a, 0)
                    +
                    row.get(b, 0)
                    +
                    row.get(c, 0)
            )
        )

        formulas.append(
            Formula(
                f"(({a}+{b})-{c})",
                lambda row, a=a, b=b, c=c:
                    row.get(a, 0)
                    +
                    row.get(b, 0)
                    -
                    row.get(c, 0)
            )
        )

        formulas.append(
            Formula(
                f"(({a}-{b})+{c})",
                lambda row, a=a, b=b, c=c:
                    row.get(a, 0)
                    -
                    row.get(b, 0)
                    +
                    row.get(c, 0)
            )
        )

        formulas.append(
            Formula(
                f"(({a}*{b})+{c})",
                lambda row, a=a, b=b, c=c:
                    row.get(a, 0)
                    *
                    row.get(b, 0)
                    +
                    row.get(c, 0)
            )
        )

        formulas.append(
            Formula(
                f"MOD10(({a}+{b})*{c})",
                lambda row, a=a, b=b, c=c:
                    mod10(
                        (
                            row.get(a, 0)
                            +
                            row.get(b, 0)
                        )
                        *
                        row.get(c, 0)
                    )
            )
        )

        formulas.append(
            Formula(
                f"MOD10({a}+{b}+{c})",
                lambda row, a=a, b=b, c=c:
                    mod10(
                        row.get(a, 0)
                        +
                        row.get(b, 0)
                        +
                        row.get(c, 0)
                    )
            )
        )

    return formulas


# ============================================================
# TARGET HIT
# ============================================================

def evaluate_formula_history(
    formula,
    features,
    data,
    start
):

    predictions = []
    actuals = []

    for i in range(start, len(data)):

        row = features.iloc[i].to_dict()

        pred = formula.calculate(
            row
        )

        actual = target_digit(
            data,
            i,
            "H"
        )

        predictions.append(pred)
        actuals.append(actual)

    return predictions, actuals


# ============================================================
# SCORE
# ============================================================

def score_formula(
    formula,
    features,
    data,
    position,
    start,
    recent_window
):

    predictions = []
    actuals = []

    for i in range(
        start,
        len(data)
    ):

        row = features.iloc[i].to_dict()

        pred = formula.calculate(
            row
        )

        actual = target_digit(
            data,
            i,
            position
        )

        predictions.append(pred)
        actuals.append(actual)

    if len(actuals) < 5:

        return None

    hits = [
        int(
            p is not None
            and p == a
        )
        for p, a in zip(
            predictions,
            actuals
        )
    ]

    total_hit = np.mean(hits)

    recent_hits = hits[
        -recent_window:
    ]

    recent_hit = (
        np.mean(recent_hits)
        if recent_hits
        else 0
    )

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    if len(hits) >= 20:

        chunks = np.array_split(
            np.array(hits),
            4
        )

        rates = [
            np.mean(c)
            for c in chunks
            if len(c)
        ]

        stability = 1 - np.std(
            rates
        )

    else:

        stability = 0.5

    stability = float(
        np.clip(
            stability,
            0,
            1
        )
    )

    # --------------------------------------------------------
    # OVERFIT
    # --------------------------------------------------------

    if len(hits) > recent_window:

        old_hits = hits[
            :-recent_window
        ]

        old_rate = np.mean(
            old_hits
        )

        gap = (
            recent_hit
            -
            old_rate
        )

    else:

        gap = 0

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = (

        total_hit * 0.40

        +

        recent_hit * 0.30

        +

        stability * 0.20

        +

        max(
            0,
            1 - abs(gap)
        ) * 0.10

    )

    # Penalty
    if gap > 0.50:

        score *= 0.65

    return {

        "formula": formula.name,

        "hit_rate": total_hit,

        "recent": recent_hit,

        "stability": stability,

        "gap": gap,

        "score": score

    }


# ============================================================
# DISCOVERY
# ============================================================

def discover_position(
    data,
    features,
    formulas,
    position,
    start,
    recent_window,
    top_n
):

    results = []

    for formula in formulas:

        result = score_formula(
            formula,
            features,
            data,
            position,
            start,
            recent_window
        )

        if result is not None:

            results.append(
                result
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_n]


# ============================================================
# PREDICT NEXT
# ============================================================

def predict_from_results(
    results,
    features
):

    if not results:

        return None

    row = features.iloc[
        -1
    ].to_dict()

    predictions = []

    for r in results:

        # สร้าง formula lookup
        pass

    return predictions


# ============================================================
# FORMULA LOOKUP
# ============================================================

def formula_map(formulas):

    return {
        f.name: f
        for f in formulas
    }


# ============================================================
# POSITION PREDICTION
# ============================================================

def position_predictions(
    results,
    formulas,
    features
):

    fmap = formula_map(
        formulas
    )

    row = features.iloc[
        -1
    ].to_dict()

    output = []

    for r in results:

        f = fmap.get(
            r["formula"]
        )

        if f is None:
            continue

        pred = f.calculate(
            row
        )

        if pred is None:
            continue

        output.append({

            "Prediction": int(pred),

            "Formula": f.name,

            "Score": r["score"],

            "Hit %": r["hit_rate"] * 100,

            "Recent %": r["recent"] * 100,

            "Stability %":
                r["stability"] * 100

        })

    return output


# ============================================================
# ENSEMBLE
# ============================================================

def build_digit_ensemble(
    prediction_lists
):

    counter = {
        0: {},
        1: {},
        2: {}
    }

    for pos, predictions in enumerate(
        prediction_lists
    ):

        for item in predictions:

            digit = item[
                "Prediction"
            ]

            weight = (

                item["Score"]
                * 0.55

                +

                (
                    item["Recent %"]
                    / 100
                )
                * 0.30

                +

                (
                    item["Stability %"]
                    / 100
                )
                * 0.15

            )

            counter[pos][
                digit
            ] = (
                counter[pos].get(
                    digit,
                    0
                )
                +
                weight
            )

    result = []

    for pos in range(3):

        ranked = sorted(
            counter[pos].items(),
            key=lambda x: x[1],
            reverse=True
        )

        result.append(
            ranked
        )

    return result


# ============================================================
# GENERATE NUMBER COMBINATIONS
# ============================================================

def generate_number_candidates(
    ensemble,
    top_digits=3
):

    digit_sets = []

    for ranked in ensemble:

        digits = [
            int(x[0])
            for x in ranked[
                :top_digits
            ]
        ]

        digit_sets.append(
            digits
        )

    numbers = []

    for a, b, c in itertools.product(
        *digit_sets
    ):

        number = (
            f"{a}{b}{c}"
        )

        weight = 0

        for pos, digit in enumerate(
            [a, b, c]
        ):

            for d, w in ensemble[pos]:

                if d == digit:

                    weight += w
                    break

        numbers.append({

            "Number": number,

            "Weight": weight

        })

    numbers.sort(
        key=lambda x: x["Weight"],
        reverse=True
    )

    return pd.DataFrame(
        numbers
    )


# ============================================================
# MAIN
# ============================================================

if data is not None and len(data) >= min_history:

    st.markdown("---")

    st.header(
        "🧠 2. AUTO SYMBOLIC DISCOVERY"
    )

    st.write(
        f"จำนวนข้อมูล: **{len(data):,} งวด**"
    )

    if st.button(
        "🚀 เริ่มค้นหาสมการอัตโนมัติ",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "AI กำลังค้นหาสมการ..."
        ):

            # --------------------------------------------
            # BUILD FEATURES
            # --------------------------------------------

            features = build_features(
                data
            )

            # --------------------------------------------
            # GENERATE
            # --------------------------------------------

            formulas = generate_formulas()

            # จำกัดจำนวน
            if len(formulas) > max_formulas:

                rng = np.random.default_rng(
                    42
                )

                idx = rng.choice(
                    len(formulas),
                    size=max_formulas,
                    replace=False
                )

                formulas = [
                    formulas[i]
                    for i in idx
                ]

            st.session_state[
                "features"
            ] = features

            st.session_state[
                "formulas"
            ] = formulas

            # --------------------------------------------
            # SEARCH
            # --------------------------------------------

            positions = [
                "H",
                "T",
                "O"
            ]

            all_results = {}

            progress = st.progress(
                0
            )

            for p_idx, position in enumerate(
                positions
            ):

                result = discover_position(
                    data,
                    features,
                    formulas,
                    position,
                    min_history,
                    recent_window,
                    top_n
                )

                all_results[
                    position
                ] = result

                progress.progress(
                    (p_idx + 1)
                    / 3
                )

            st.session_state[
                "results"
            ] = all_results

            st.success(
                "ค้นหาสมการเสร็จแล้ว"
            )


# ============================================================
# RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state[
        "results"
    ]

    features = st.session_state[
        "features"
    ]

    formulas = st.session_state[
        "formulas"
    ]

    st.markdown("---")

    st.header(
        "🏆 3. TOP DISCOVERED EQUATIONS"
    )

    tabs = st.tabs([
        "🔴 หลักร้อย H",
        "🟢 หลักสิบ T",
        "🔵 หลักหน่วย O"
    ])

    prediction_lists = []

    for tab, position in zip(
        tabs,
        ["H", "T", "O"]
    ):

        with tab:

            result = results[
                position
            ]

            table = pd.DataFrame([

                {
                    "อันดับ": i + 1,

                    "สูตร":
                        r["formula"],

                    "Hit %":
                        round(
                            r["hit_rate"] * 100,
                            2
                        ),

                    "Recent %":
                        round(
                            r["recent"] * 100,
                            2
                        ),

                    "Stability %":
                        round(
                            r["stability"] * 100,
                            2
                        ),

                    "Overfit Gap %":
                        round(
                            r["gap"] * 100,
                            2
                        ),

                    "SCORE":
                        round(
                            r["score"] * 100,
                            3
                        )
                }

                for i, r in enumerate(
                    result
                )

            ])

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------
            # PREDICTION
            # ----------------------------------------

            preds = position_predictions(
                result,
                formulas,
                features
            )

            prediction_lists.append(
                preds
            )

            st.subheader(
                f"🔮 สูตรที่ใช้คำนวณงวดถัดไป — {position}"
            )

            pred_table = pd.DataFrame(
                preds
            )

            if not pred_table.empty:

                pred_table[
                    "Score"
                ] = pred_table[
                    "Score"
                ].round(4)

                st.dataframe(
                    pred_table,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# ENSEMBLE
# ============================================================

if "results" in st.session_state:

    st.markdown("---")

    st.header(
        "🎯 4. SYMBOLIC EQUATION ENSEMBLE"
    )

    results = st.session_state[
        "results"
    ]

    features = st.session_state[
        "features"
    ]

    formulas = st.session_state[
        "formulas"
    ]

    prediction_lists = []

    for position in [
        "H",
        "T",
        "O"
    ]:

        prediction_lists.append(
            position_predictions(
                results[position],
                formulas,
                features
            )
        )

    ensemble = build_digit_ensemble(
        prediction_lists
    )

    c1, c2, c3 = st.columns(3)

    for col, title, ranked in zip(
        [c1, c2, c3],
        ["🔴 H", "🟢 T", "🔵 O"],
        ensemble
    ):

        with col:

            st.subheader(
                title
            )

            if ranked:

                dtable = pd.DataFrame(
                    ranked[:10],
                    columns=[
                        "Digit",
                        "Weight"
                    ]
                )

                dtable[
                    "Weight"
                ] = dtable[
                    "Weight"
                ].round(4)

                st.dataframe(
                    dtable,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# NUMBER CANDIDATES
# ============================================================

if "results" in st.session_state:

    st.markdown("---")

    st.header(
        "🔢 5. TOP 3-DIGIT CANDIDATES"
    )

    results = st.session_state[
        "results"
    ]

    features = st.session_state[
        "features"
    ]

    formulas = st.session_state[
        "formulas"
    ]

    prediction_lists = []

    for position in [
        "H",
        "T",
        "O"
    ]:

        prediction_lists.append(
            position_predictions(
                results[position],
                formulas,
                features
            )
        )

    ensemble = build_digit_ensemble(
        prediction_lists
    )

    candidates = generate_number_candidates(
        ensemble,
        top_digits=4
    )

    if not candidates.empty:

        candidates[
            "Weight"
        ] = candidates[
            "Weight"
        ].round(5)

        st.dataframe(
            candidates.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🔥 Top 10"
        )

        top10 = candidates.head(
            10
        )["Number"].tolist()

        st.success(
            "   ".join(
                top10
            )
        )


# ============================================================
# RECENT 10 TEST
# ============================================================

if "results" in st.session_state:

    st.markdown("---")

    st.header(
        "🧪 6. ตรวจสอบสูตรกับ 10 งวดล่าสุด"
    )

    data = data.reset_index(
        drop=True
    )

    features = st.session_state[
        "features"
    ]

    formulas = st.session_state[
        "formulas"
    ]

    results = st.session_state[
        "results"
    ]

    fmap = formula_map(
        formulas
    )

    recent_n = min(
        10,
        len(data)
    )

    recent_rows = []

    for i in range(
        len(data) - recent_n,
        len(data)
    ):

        row = features.iloc[
            i
        ].to_dict()

        actual = data.iloc[
            i
        ]["3D"]

        pred_digits = []

        for position in [
            "H",
            "T",
            "O"
        ]:

            best = results[
                position
            ][0]

            formula = fmap.get(
                best["formula"]
            )

            if formula:

                pred = formula.calculate(
                    row
                )

            else:

                pred = None

            pred_digits.append(
                "?"
                if pred is None
                else str(pred)
            )

        prediction = "".join(
            pred_digits
        )

        recent_rows.append({

            "งวด": i + 1,

            "AI Formula":
                prediction,

            "Actual":
                actual,

            "ตรง":
                prediction == actual

        })

    recent_df = pd.DataFrame(
        recent_rows
    )

    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True
    )

    recent_accuracy = (
        recent_df["ตรง"].mean()
        * 100
    )

    st.metric(
        "Exact Match 10 งวดล่าสุด",
        f"{recent_accuracy:.1f}%"
    )


# ============================================================
# EXPLANATION
# ============================================================

st.markdown("---")

with st.expander(
    "ℹ️ วิธีอ่านผล"
):

    st.markdown("""
### ตัวอย่าง

ถ้า AI ค้นพบ

`MOD10(H_L1 + O2_L1)`

หมายถึง

**เอาหลักร้อยของงวดก่อน + หลักหน่วยของเลข 2 ตัวงวดก่อน แล้ว Mod 10**

ถ้าได้

`6 + 5 = 11`

ดังนั้น

`11 Mod 10 = 1`

AI จะเสนอ **1** สำหรับตำแหน่งนั้น

---

### Score

คะแนนรวมมาจาก

- Historical Hit Rate
- Recent Window
- Stability
- Overfit Control

ดังนั้นสูตรที่ได้ Hit สูงอย่างเดียวไม่ได้หมายความว่าจะเป็นสูตรอันดับ 1 เสมอไป

---

### จุดสำคัญ

ระบบนี้ไม่ได้บอกว่า "พบสูตรลับของหวย"

แต่เป็นการค้นหา **รูปแบบทางคณิตศาสตร์ที่เคยสัมพันธ์กับข้อมูลย้อนหลัง** แล้วตรวจสอบว่าความสัมพันธ์นั้นยังเสถียรหรือไม่
""")


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "LOTTO AI AUTO SYMBOLIC EQUATION V2 • "
    "Research / Experimental Pattern Discovery"
)
