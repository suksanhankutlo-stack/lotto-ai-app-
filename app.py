# ============================================================
# LOTTO AI - AUTO SYMBOLIC EQUATION V3
# BLOGSPOT AUTO SCRAPER + POSITION EQUATION DISCOVERY
# ============================================================
# Install:
#   pip install streamlit pandas numpy requests beautifulsoup4
#
# Run:
#   streamlit run lotto_symbolic_blogspot_v3.py
#
# IMPORTANT:
# This is a pattern-discovery / backtest tool, not a guarantee of
# future lottery results.
# ============================================================

import re
import warnings
import itertools
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Lotto AI Symbolic Blogspot V3",
    page_icon="🧠",
    layout="wide",
)

BLOG_URLS = {
    "หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "หวยธกส": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 10) "
        "AppleWebKit/537.36 Chrome/125 Safari/537.36"
    )
}


# ============================================================
# HTTP / BLOGSPOT
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_html(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    root = (
        soup.select_one(".post-body")
        or soup.select_one(".entry-content")
        or soup.select_one("article")
        or soup.body
    )

    if root is None:
        return "", soup

    return root.get_text("\n", strip=True), soup


def normalize_url(base, href):
    try:
        u = urljoin(base, href)
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            return None
        return u.split("#")[0]
    except Exception:
        return None


def same_blog(url_a, url_b):
    return urlparse(url_a).netloc.lower() == urlparse(url_b).netloc.lower()


def blog_links(url, soup):
    out = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(url, a["href"])
        if u and same_blog(url, u):
            out.append(u)
    return list(dict.fromkeys(out))


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def norm3(x):
    s = re.sub(r"\D", "", str(x))
    return s.zfill(3)[-3:] if s else None


def norm2(x):
    s = re.sub(r"\D", "", str(x))
    return s.zfill(2)[-2:] if s else None


def extract_labeled_numbers(text):
    """
    Try common Thai labels first.
    Returns candidate 3D/2D values.
    """
    t = text.replace("\u200b", " ")

    p3 = [
        r"(?:เลขสามตัว|3\s*ตัว|สามตัว|3d|3\s*digit)"
        r"\s*[:=\-]?\s*([0-9]{3})",
        r"(?:สามตัวบน|3\s*ตัวบน)"
        r"\s*[:=\-]?\s*([0-9]{3})",
        r"(?:สามตัวโต๊ด|3\s*ตัวโต๊ด)"
        r"\s*[:=\-]?\s*([0-9]{3})",
    ]

    p2 = [
        r"(?:เลขสองตัว|2\s*ตัว|สองตัว|2d|2\s*digit)"
        r"\s*[:=\-]?\s*([0-9]{2})",
        r"(?:สองตัวล่าง|2\s*ตัวล่าง)"
        r"\s*[:=\-]?\s*([0-9]{2})",
        r"(?:สองตัวบน|2\s*ตัวบน)"
        r"\s*[:=\-]?\s*([0-9]{2})",
    ]

    three = []
    two = []

    for p in p3:
        three += re.findall(p, t, flags=re.I)

    for p in p2:
        two += re.findall(p, t, flags=re.I)

    three = [norm3(x) for x in three if norm3(x)]
    two = [norm2(x) for x in two if norm2(x)]

    return list(dict.fromkeys(three)), list(dict.fromkeys(two))


def extract_generic_pairs(text):
    """
    Fallback parser for pages without obvious labels.

    It searches each line for 3-digit and 2-digit numbers that occur
    near each other. It deliberately ignores most date-like tokens.
    """
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        nums = re.findall(r"(?<!\d)\d{1,3}(?!\d)", line)

        if not nums:
            continue

        # Prefer explicit 3-digit + 2-digit on same line.
        threes = [x for x in nums if len(x) == 3]
        twos = [x for x in nums if len(x) <= 2]

        if threes and twos:
            for a in threes[:3]:
                # Prefer the last short number on the same line.
                b = twos[-1]
                rows.append((norm3(a), norm2(b)))

    return rows


def parse_page(url):
    html = fetch_html(url)
    text, soup = clean_text(html)

    threes, twos = extract_labeled_numbers(text)

    rows = []

    # If labels found, pair by order.
    if threes and twos:
        n = min(len(threes), len(twos))
        for i in range(n):
            rows.append((threes[i], twos[i]))

    # Generic fallback.
    if not rows:
        rows = extract_generic_pairs(text)

    # Deduplicate.
    rows = list(dict.fromkeys(
        (a, b) for a, b in rows
        if a and b
    ))

    return rows, text, soup


# ============================================================
# HISTORICAL PAGE CRAWLER
# ============================================================

def score_link_for_category(url, category):
    """
    Keep same-blog links. The score helps prioritize links likely
    to belong to the same category / result pages.
    """
    s = url.lower()
    score = 0

    keys = {
        "หวยไทย": ["ไทย", "รัฐบาล", "lotto"],
        "หวยลาว": ["ลาว", "lao"],
        "หวยฮานอย": ["ฮานอย", "hanoi"],
        "หวยธกส": ["ธกส", "ธ.ก.ส"],
        "หวยออมสิน": ["ออมสิน"],
        "หวยมาเลย์": ["มาเลย์", "malay"],
        "หวยหุ้นไทยเย็น": ["หุ้นไทย", "ไทยเย็น"],
        "หวยหุ้นนิเคอิบ่าย": ["นิเคอิ", "nikkei"],
        "หวยหุ้นฮั่งเส็งบ่าย": ["ฮั่งเส็ง", "hangseng"],
        "หวยหุ้นจีนบ่าย": ["หุ้นจีน", "จีนบ่าย"],
    }

    for k in keys.get(category, []):
        if k.lower() in s:
            score += 3

    return score


def crawl_blogspot(start_url, category, max_pages=80):
    """
    Crawl same-domain pages from the supplied seed URL.
    It prioritizes links that look like the selected category.

    Important:
    The seed page is always included.
    """
    visited = set()
    queue = [(start_url, 100)]
    collected = []

    while queue and len(visited) < max_pages:
        queue.sort(key=lambda x: x[1], reverse=True)
        url, _ = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        try:
            rows, text, soup = parse_page(url)
        except Exception:
            continue

        if rows:
            for a, b in rows:
                collected.append({
                    "source_url": url,
                    "3D": a,
                    "2D": b,
                })

        for link in blog_links(url, soup):
            if link in visited:
                continue

            # Avoid obvious non-post resources.
            low = link.lower()
            if any(x in low for x in [
                "/p/", "/search", "/feeds/", ".xml",
                "javascript:", "mailto:"
            ]):
                continue

            score = score_link_for_category(
                link, category
            )

            # Blogspot post URLs get a small priority boost.
            if ".html" in low:
                score += 2

            # Do not completely exclude ordinary same-domain
            # links because Blogspot templates vary.
            queue.append((link, score))

    return pd.DataFrame(collected)


# ============================================================
# CLEAN / SORT HISTORY
# ============================================================

def clean_history(df):
    if df.empty:
        return df

    out = df.copy()

    out["3D"] = out["3D"].apply(norm3)
    out["2D"] = out["2D"].apply(norm2)

    out = out.dropna(subset=["3D", "2D"])
    out = out.drop_duplicates(
        subset=["source_url", "3D", "2D"]
    ).reset_index(drop=True)

    # Keep crawler order. The pages are usually chronological around
    # the seed, but source dates vary by blog template.
    out["row_id"] = np.arange(len(out))

    return out


# ============================================================
# SYMBOLIC FEATURES
# ============================================================

def make_raw(row):
    a = str(row["3D"]).zfill(3)
    b = str(row["2D"]).zfill(2)

    H, T, O = map(int, a)
    T2, O2 = map(int, b)

    return {
        "H": H,
        "T": T,
        "O": O,
        "T2": T2,
        "O2": O2,

        "S3": H + T + O,
        "S2": T2 + O2,

        "HT": abs(H - T),
        "TO": abs(T - O),
        "HO": abs(H - O),

        "HT2": abs(H - T2),
        "HO2": abs(H - O2),
        "TT2": abs(T - T2),
        "TO2": abs(T - O2),

        "R3": int(a[::-1]),
        "R2": int(b[::-1]),

        "DS3": H + T + O,
    }


def build_features(data):
    rows = []

    for i in range(len(data)):
        r = {}

        # Only previous draws: no look-ahead.
        for lag in [1, 2, 3, 5]:
            j = i - lag
            if j >= 0:
                vals = make_raw(data.iloc[j])
                for k, v in vals.items():
                    r[f"{k}_L{lag}"] = v

        rows.append(r)

    return pd.DataFrame(rows).fillna(0)


# ============================================================
# SYMBOLIC FORMULA
# ============================================================

class Formula:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def calc(self, row):
        try:
            x = self.fn(row)
            if x is None or not np.isfinite(x):
                return None
            return int(round(x)) % 10
        except Exception:
            return None


def generate_formulas(max_formulas=8000):
    formulas = []

    base = [
        f"{x}_L{lag}"
        for lag in [1, 2, 3, 5]
        for x in [
            "H", "T", "O", "T2", "O2",
            "S3", "S2", "HT", "TO", "HO",
            "HT2", "HO2", "TT2", "TO2",
            "R3", "R2", "DS3"
        ]
    ]

    # Singles
    for a in base:
        formulas.append(
            Formula(
                a,
                lambda r, a=a: r.get(a, 0)
            )
        )

    # Binary operations
    for a, b in itertools.combinations(base, 2):
        formulas.extend([
            Formula(
                f"({a}+{b})",
                lambda r, a=a, b=b:
                    r.get(a, 0) + r.get(b, 0)
            ),
            Formula(
                f"({a}-{b})",
                lambda r, a=a, b=b:
                    r.get(a, 0) - r.get(b, 0)
            ),
            Formula(
                f"ABS({a}-{b})",
                lambda r, a=a, b=b:
                    abs(r.get(a, 0) - r.get(b, 0))
            ),
            Formula(
                f"({a}*{b})",
                lambda r, a=a, b=b:
                    r.get(a, 0) * r.get(b, 0)
            ),
            Formula(
                f"MOD10({a}+{b})",
                lambda r, a=a, b=b:
                    (r.get(a, 0) + r.get(b, 0)) % 10
            ),
            Formula(
                f"MOD9({a}+{b})",
                lambda r, a=a, b=b:
                    (r.get(a, 0) + r.get(b, 0)) % 9
            ),
            Formula(
                f"MOD10({a}*{b})",
                lambda r, a=a, b=b:
                    (r.get(a, 0) * r.get(b, 0)) % 10
            ),
            Formula(
                f"({a}/{b})",
                lambda r, a=a, b=b:
                    None if r.get(b, 0) == 0
                    else r.get(a, 0) / r.get(b, 0)
            ),
        ])

        if len(formulas) >= max_formulas:
            return formulas[:max_formulas]

    # Triple formulas from a compact set to keep the app fast.
    important = [
        x for x in base
        if x.split("_")[0] in [
            "H", "T", "O", "T2", "O2",
            "S3", "S2", "HT", "TO", "HO"
        ]
    ]

    for a, b, c in itertools.combinations(
        important, 3
    ):
        formulas.extend([
            Formula(
                f"(({a}+{b})+{c})",
                lambda r, a=a, b=b, c=c:
                    r.get(a, 0) + r.get(b, 0) + r.get(c, 0)
            ),
            Formula(
                f"(({a}+{b})-{c})",
                lambda r, a=a, b=b, c=c:
                    r.get(a, 0) + r.get(b, 0) - r.get(c, 0)
            ),
            Formula(
                f"(({a}-{b})+{c})",
                lambda r, a=a, b=b, c=c:
                    r.get(a, 0) - r.get(b, 0) + r.get(c, 0)
            ),
            Formula(
                f"MOD10(({a}*{b})+{c})",
                lambda r, a=a, b=b, c=c:
                    (r.get(a, 0) * r.get(b, 0) + r.get(c, 0)) % 10
            ),
            Formula(
                f"MOD10(({a}+{b})*{c})",
                lambda r, a=a, b=b, c=c:
                    ((r.get(a, 0) + r.get(b, 0)) * r.get(c, 0)) % 10
            ),
        ])

        if len(formulas) >= max_formulas:
            break

    return formulas[:max_formulas]


# ============================================================
# DISCOVERY / WALK-FORWARD
# ============================================================

def target_digit(data, i, position):
    s = str(data.iloc[i]["3D"]).zfill(3)
    return int({
        "H": s[0],
        "T": s[1],
        "O": s[2],
    }[position])


def score_formula(
    formula,
    features,
    data,
    position,
    start,
    recent_window
):
    preds = []
    acts = []

    for i in range(start, len(data)):
        pred = formula.calc(
            features.iloc[i].to_dict()
        )
        actual = target_digit(
            data, i, position
        )

        preds.append(pred)
        acts.append(actual)

    if len(acts) < 5:
        return None

    hits = np.array([
        int(p is not None and p == a)
        for p, a in zip(preds, acts)
    ])

    hit = float(np.mean(hits))
    recent = float(
        np.mean(hits[-recent_window:])
    )

    if len(hits) >= 20:
        chunks = np.array_split(hits, 4)
        rates = [
            float(np.mean(c))
            for c in chunks if len(c)
        ]
        stability = float(
            np.clip(1 - np.std(rates), 0, 1)
        )
    else:
        stability = 0.5

    if len(hits) > recent_window:
        old = float(
            np.mean(hits[:-recent_window])
        )
        gap = recent - old
    else:
        gap = 0.0

    score = (
        hit * 0.40
        + recent * 0.30
        + stability * 0.20
        + max(0, 1 - abs(gap)) * 0.10
    )

    # Penalize formulas that only suddenly work in the latest window.
    if gap > 0.50:
        score *= 0.65

    return {
        "formula": formula.name,
        "hit": hit,
        "recent": recent,
        "stability": stability,
        "gap": gap,
        "score": score,
    }


def discover(
    data,
    features,
    formulas,
    position,
    start,
    recent_window,
    top_n
):
    out = []

    for f in formulas:
        r = score_formula(
            f,
            features,
            data,
            position,
            start,
            recent_window
        )
        if r is not None:
            out.append(r)

    out.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return out[:top_n]


def predict_top(
    results,
    formulas,
    features
):
    fmap = {f.name: f for f in formulas}
    row = features.iloc[-1].to_dict()

    out = []

    for r in results:
        f = fmap.get(r["formula"])
        if f is None:
            continue

        p = f.calc(row)
        if p is None:
            continue

        out.append({
            "Digit": int(p),
            "Formula": f.name,
            "Score %": round(r["score"] * 100, 3),
            "Hit %": round(r["hit"] * 100, 2),
            "Recent %": round(r["recent"] * 100, 2),
            "Stability %": round(
                r["stability"] * 100, 2
            ),
        })

    return out


def ensemble_digits(pred_lists):
    """
    Weighted vote for H/T/O.
    """
    ranked = []

    for preds in pred_lists:
        counter = {}

        for p in preds:
            d = p["Digit"]
            w = (
                p["Score %"] * 0.55
                + p["Recent %"] * 0.30
                + p["Stability %"] * 0.15
            )
            counter[d] = counter.get(d, 0) + w

        ranked.append(
            sorted(
                counter.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

    return ranked


def candidate_numbers(ranked, top_digits=4):
    if any(not x for x in ranked):
        return pd.DataFrame()

    sets = [
        [int(x[0]) for x in r[:top_digits]]
        for r in ranked
    ]

    rows = []

    for h in sets[0]:
        for t in sets[1]:
            for o in sets[2]:
                weight = 0.0

                for pos, d in enumerate([h, t, o]):
                    for dd, ww in ranked[pos]:
                        if int(dd) == d:
                            weight += ww
                            break

                rows.append({
                    "Number": f"{h}{t}{o}",
                    "Weight": weight,
                })

    return pd.DataFrame(rows).sort_values(
        "Weight",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ SETTINGS")

category = st.sidebar.selectbox(
    "เลือกหวย",
    list(BLOG_URLS.keys())
)

max_pages = st.sidebar.slider(
    "จำนวนหน้า Blogspot สูงสุด",
    1, 150, 50
)

min_history = st.sidebar.slider(
    "จำนวนงวดขั้นต่ำ",
    20, 200, 40
)

top_n = st.sidebar.slider(
    "Top สูตรต่อหลัก",
    3, 30, 10
)

recent_window = st.sidebar.slider(
    "Recent Window",
    5, 30, 10
)

max_formulas = st.sidebar.slider(
    "จำนวนสูตรสูงสุด",
    1000, 12000, 5000,
    step=500
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Seed URL จะเป็น URL ที่คุณให้มา "
    "จากนั้นระบบจะพยายามไล่ลิงก์ใน Blogspot "
    "เพื่อหาโพสต์ย้อนหลังในโดเมนเดียวกัน"
)


# ============================================================
# URL DISPLAY
# ============================================================

st.title("🧠 LOTTO AI — AUTO SYMBOLIC EQUATION V3")

st.info(
    f"แหล่งข้อมูลที่เลือก: **{category}**\n\n"
    f"{BLOG_URLS[category]}"
)

if st.button(
    "🌐 ดึงข้อมูลจาก Blogspot",
    type="primary",
    use_container_width=True
):
    with st.spinner(
        "กำลังอ่าน Blogspot และค้นหาโพสต์ย้อนหลัง..."
    ):
        raw = crawl_blogspot(
            BLOG_URLS[category],
            category,
            max_pages=max_pages
        )

        data = clean_history(raw)

        st.session_state["blog_data"] = data
        st.session_state["blog_category"] = category


# ============================================================
# MANUAL URL OPTION
# ============================================================

with st.expander("🔗 เพิ่ม URL Blogspot เอง"):
    custom_url = st.text_input(
        "URL ของหน้า Blogspot"
    )

    if st.button(
        "ดึง URL นี้"
    ):
        if custom_url.strip():
            try:
                raw_rows, _, _ = parse_page(
                    custom_url.strip()
                )

                custom_df = pd.DataFrame(
                    [
                        {
                            "source_url": custom_url.strip(),
                            "3D": a,
                            "2D": b,
                        }
                        for a, b in raw_rows
                    ]
                )

                st.session_state[
                    "custom_data"
                ] = clean_history(
                    custom_df
                )

                st.success(
                    f"พบ {len(custom_df)} รายการ"
                )
            except Exception as e:
                st.error(
                    f"อ่าน URL ไม่สำเร็จ: {e}"
                )


# ============================================================
# DATA VIEW
# ============================================================

data = st.session_state.get(
    "blog_data",
    pd.DataFrame()
)

if not st.session_state.get(
    "blog_category"
) == category:
    data = pd.DataFrame()

custom_data = st.session_state.get(
    "custom_data",
    pd.DataFrame()
)

if not custom_data.empty:
    if data.empty:
        data = custom_data
    else:
        data = pd.concat(
            [data, custom_data],
            ignore_index=True
        ).drop_duplicates(
            subset=["source_url", "3D", "2D"]
        )


if not data.empty:

    st.subheader(
        f"📊 ข้อมูลที่ดึงได้: {len(data):,} รายการ"
    )

    st.dataframe(
        data.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "ระบบเก็บ source_url ไว้เพื่อให้ตรวจสอบย้อนกลับได้"
    )


# ============================================================
# DISCOVERY
# ============================================================

if not data.empty and len(data) >= min_history:

    st.markdown("---")
    st.header(
        "🧠 AUTO SYMBOLIC DISCOVERY"
    )

    if st.button(
        "🚀 ค้นหาสมการอัตโนมัติ",
        use_container_width=True
    ):

        with st.spinner(
            "กำลังสร้างสูตรและทดสอบแบบ Walk-forward..."
        ):

            features = build_features(
                data
            )

            formulas = generate_formulas(
                max_formulas=max_formulas
            )

            st.write(
                f"สร้างสูตรทั้งหมด: "
                f"**{len(formulas):,} สูตร**"
            )

            results = {}

            progress = st.progress(0)

            for idx, pos in enumerate(
                ["H", "T", "O"]
            ):
                results[pos] = discover(
                    data,
                    features,
                    formulas,
                    pos,
                    min_history,
                    recent_window,
                    top_n
                )

                progress.progress(
                    (idx + 1) / 3
                )

            st.session_state[
                "features"
            ] = features

            st.session_state[
                "formulas"
            ] = formulas

            st.session_state[
                "results"
            ] = results

        st.success(
            "ค้นหาสมการเสร็จแล้ว"
        )


# ============================================================
# SHOW RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state["results"]
    features = st.session_state["features"]
    formulas = st.session_state["formulas"]

    st.markdown("---")
    st.header("🏆 TOP EQUATIONS")

    tabs = st.tabs([
        "🔴 H หลักร้อย",
        "🟢 T หลักสิบ",
        "🔵 O หลักหน่วย",
    ])

    prediction_lists = []

    for tab, pos in zip(
        tabs,
        ["H", "T", "O"]
    ):
        with tab:

            rows = results[pos]

            table = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Formula": r["formula"],
                    "Hit %": round(
                        r["hit"] * 100, 2
                    ),
                    "Recent %": round(
                        r["recent"] * 100, 2
                    ),
                    "Stability %": round(
                        r["stability"] * 100, 2
                    ),
                    "Overfit Gap %": round(
                        r["gap"] * 100, 2
                    ),
                    "SCORE %": round(
                        r["score"] * 100, 3
                    ),
                }
                for i, r in enumerate(rows)
            ])

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )

            preds = predict_top(
                rows,
                formulas,
                features
            )

            prediction_lists.append(preds)

            st.subheader(
                "🔮 ผลสูตรสำหรับงวดถัดไป"
            )

            if preds:
                st.dataframe(
                    pd.DataFrame(preds),
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# DIGIT ENSEMBLE
# ============================================================

if "results" in st.session_state:

    st.markdown("---")
    st.header(
        "🎯 SYMBOLIC EQUATION ENSEMBLE"
    )

    results = st.session_state["results"]
    features = st.session_state["features"]
    formulas = st.session_state["formulas"]

    prediction_lists = [
        predict_top(
            results[p],
            formulas,
            features
        )
        for p in ["H", "T", "O"]
    ]

    ranked = ensemble_digits(
        prediction_lists
    )

    c1, c2, c3 = st.columns(3)

    for col, title, values in zip(
        [c1, c2, c3],
        ["🔴 H", "🟢 T", "🔵 O"],
        ranked
    ):
        with col:
            st.subheader(title)

            if values:
                dt = pd.DataFrame(
                    values[:10],
                    columns=[
                        "Digit",
                        "Weight"
                    ]
                )

                dt["Weight"] = dt[
                    "Weight"
                ].round(3)

                st.dataframe(
                    dt,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# NUMBER CANDIDATES
# ============================================================

if "results" in st.session_state:

    st.markdown("---")
    st.header(
        "🔢 TOP 3-DIGIT CANDIDATES"
    )

    candidates = candidate_numbers(
        ranked,
        top_digits=4
    )

    if not candidates.empty:

        candidates["Weight"] = candidates[
            "Weight"
        ].round(3)

        st.dataframe(
            candidates.head(30),
            use_container_width=True,
            hide_index=True
        )

        top10 = candidates.head(
            10
        )["Number"].tolist()

        st.success(
            "   ".join(top10)
        )


# ============================================================
# RECENT 10 VALIDATION
# ============================================================

if "results" in st.session_state:

    st.markdown("---")
    st.header(
        "🧪 CHECK — 10 งวดล่าสุด"
    )

    results = st.session_state["results"]
    features = st.session_state["features"]
    formulas = st.session_state["formulas"]

    fmap = {
        f.name: f
        for f in formulas
    }

    n = min(
        recent_window,
        len(data)
    )

    rows = []

    for i in range(
        len(data) - n,
        len(data)
    ):

        row = features.iloc[i].to_dict()

        pred = []

        for pos in ["H", "T", "O"]:

            if not results[pos]:
                pred.append("?")
                continue

            best = results[pos][0]

            f = fmap.get(
                best["formula"]
            )

            if f is None:
                pred.append("?")
            else:
                x = f.calc(row)
                pred.append(
                    "?"
                    if x is None
                    else str(x)
                )

        p = "".join(pred)

        rows.append({
            "Index": i + 1,
            "AI": p,
            "Actual": data.iloc[i]["3D"],
            "Exact": p == data.iloc[i]["3D"],
        })

    check = pd.DataFrame(rows)

    st.dataframe(
        check,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        f"Exact Match {n} งวด",
        f"{check['Exact'].mean() * 100:.1f}%"
    )


# ============================================================
# WARNINGS / HELP
# ============================================================

st.markdown("---")

with st.expander(
    "⚠️ ข้อควรระวังในการใช้ตัวดึงข้อมูล"
):
    st.markdown("""
1. Blogspot แต่ละโพสต์อาจมีรูปแบบข้อความไม่เหมือนกัน
2. ระบบจะพยายามอ่าน `.post-body`, `.entry-content` และ `article`
3. ถ้าหน้าเว็บใช้รูปภาพแทนตัวเลขทั้งหมด ระบบจะอ่านเลขจากรูปไม่ได้
4. ถ้าคอลัมน์ 3D/2D ในโพสต์ไม่มีป้ายกำกับชัดเจน ระบบจะใช้ fallback parser
5. หลังดึงข้อมูลควรตรวจตาราง "ข้อมูลที่ดึงได้" ก่อนกดค้นหาสมการ
6. ถ้าข้อมูลผิด การค้นหาสมการก็จะผิดตามข้อมูล
7. คะแนน Hit/Recent/Stability เป็นสถิติย้อนหลัง ไม่ใช่การรับประกันผลอนาคต
""")

with st.expander(
    "🧩 ความหมายของสูตร"
):
    st.markdown("""
ตัวอย่าง:

`MOD10(H_L1 + O2_L1)`

หมายถึงนำหลักร้อยของงวดก่อนหน้า
บวกกับหลักหน่วยของเลข 2 ตัวงวดก่อนหน้า
แล้วเอาเศษจากการหาร 10

`ABS(T_L1 - T2_L1)`

หมายถึงหาค่าสัมบูรณ์ของ
หลักสิบ 3 ตัวงวดก่อนหน้า
ลบหลักสิบ 2 ตัวงวดก่อนหน้า

`(({H_L1}+{T_L1})*{O_L1})`

เป็นตัวอย่างโครงสร้างการรวม 3 ตัวแปร
""")

st.caption(
    "LOTTO AI AUTO SYMBOLIC EQUATION V3 • "
    "Blogspot + Walk-forward + Position Search"
)
