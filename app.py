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

# ============================================================
# STREAMLIT CONFIG & SIDEBAR SETTINGS
# ============================================================

st.set_page_config(
    page_title="Lotto AI Symbolic V4",
    page_icon="🧠",
    layout="wide",
)

st.sidebar.header("⚙️ SETTINGS")

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

category = st.sidebar.selectbox("เลือกหวย", list(BLOG_URLS.keys()))
max_pages = st.sidebar.slider("จำนวนหน้า Blogspot สูงสุด", 1, 150, 50)
min_history = st.sidebar.slider("จำนวนงวดขั้นต่ำ", 20, 200, 40)
max_formulas = st.sidebar.slider("จำนวนสูตรสูงสุด", 1000, 12000, 5000, step=500)
LOCK_WINDOW = st.sidebar.slider("หน้าต่างประเมินสูตร (Lock Window)", 10, 50, 20)

st.sidebar.markdown("---")
st.sidebar.info(
    f"🔒 ระบบ Lock:\n\n"
    f"• ใช้ {LOCK_WINDOW} งวดล่าสุดคัดสูตร\n"
    f"• ล็อกแยก H / T / O\n"
    f"• ผิด 1 งวด = เตือน\n"
    f"• ผิด 2 งวดติด = เปลี่ยนสูตร\n"
    f"• เปลี่ยนเฉพาะหลักที่หลุด"
)

# ============================================================
# CONSTANTS
# ============================================================

POSITIONS = ["H", "T", "O"]
FAIL_LIMIT = 2
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
}

# ============================================================
# HTTP & TEXT CLEANING
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
    root = soup.select_one(".post-body") or soup.select_one(".entry-content") or soup.select_one("article") or soup.body
    if root is None: return "", soup
    return root.get_text("\n", strip=True), soup

# ============================================================
# URL HELPERS
# ============================================================

def normalize_url(base, href):
    try:
        u = urljoin(base, href)
        p = urlparse(u)
        if p.scheme not in ("http", "https"): return None
        return u.split("#")[0]
    except Exception:
        return None

def same_blog(url_a, url_b):
    return urlparse(url_a).netloc.lower() == urlparse(url_b).netloc.lower()

def blog_links(url, soup):
    out = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(url, a["href"])
        if u and same_blog(url, u): out.append(u)
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
    t = text.replace("\u200b", " ")
    p3 = [
        r"(?:เลขสามตัว|3\s*ตัว|สามตัว|3d|3\s*digit)\s*[:=\-]?\s*([0-9]{3})",
        r"(?:สามตัวบน|3\s*ตัวบน)\s*[:=\-]?\s*([0-9]{3})",
        r"(?:สามตัวโต๊ด|3\s*ตัวโต๊ด)\s*[:=\-]?\s*([0-9]{3})",
    ]
    p2 = [
        r"(?:เลขสองตัว|2\s*ตัว|สองตัว|2d|2\s*digit)\s*[:=\-]?\s*([0-9]{2})",
        r"(?:สองตัวล่าง|2\s*ตัวล่าง)\s*[:=\-]?\s*([0-9]{2})",
        r"(?:สองตัวบน|2\s*ตัวบน)\s*[:=\-]?\s*([0-9]{2})",
    ]
    three, two = [], []
    for p in p3: three += re.findall(p, t, flags=re.I)
    for p in p2: two += re.findall(p, t, flags=re.I)

    three = [norm3(x) for x in three if norm3(x)]
    two = [norm2(x) for x in two if norm2(x)]
    return list(dict.fromkeys(three)), list(dict.fromkeys(two))

# ============================================================
# DATE & PAGE PARSING (🌟 ลอจิกใหม่: บังคับหาตารางสถิติอันดับ 1)
# ============================================================

def get_page_date(soup, url):
    meta = soup.find("meta", itemprop="datePublished")
    if meta and meta.get("content"): return meta["content"]
    
    time_tag = soup.find(["time", "abbr"], class_="published")
    if time_tag: return time_tag.get("datetime") or time_tag.get("title") or time_tag.get_text()
        
    dh = soup.find(class_=re.compile("date-header", re.I))
    if dh: return dh.get_text(strip=True)
        
    m = re.search(r"/(\d{4})/(\d{2})/", url)
    if m: return f"{m.group(1)}-{m.group(2)}-01"
        
    return None

def parse_page(url):
    html = fetch_html(url)
    text, soup = clean_text(html)
    page_date = get_page_date(soup, url)
    rows = []
    
    # 🌟 1. ค้นหารูปแบบประวัติโดยตรงก่อนเสมอ (เช่น "* 2026-09-02 | 351 | 53")
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        
        # Regex: วันที่ (YYYY-MM-DD หรือ DD/MM/YY) คั่นด้วยอะไรก็ได้ ตามด้วย 3 หลัก และ 2 หลัก
        match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{2,4})\D+?(\d{3})\D+?(\d{2})(?!\d)", line)
        if match:
            d_str, d3, d2 = match.groups()
            rows.append({"3D": d3, "2D": d2, "date": d_str})
    
    # 2. ถ้าไม่เจอตารางสถิติแบบ List ถึงจะค่อยไปงมหาตัวเลขเดี่ยวๆ ในหน้า (Fallback)
    if not rows:
        threes, twos = extract_labeled_numbers(text)
        if threes and twos:
            n = min(len(threes), len(twos))
            for i in range(n):
                rows.append({"3D": threes[i], "2D": twos[i], "date": page_date})
                
        # 3. Fallback ลำดับสุดท้าย (Generic Pairs)
        if not rows:
            for line in text.splitlines():
                line = line.strip()
                if not line: continue
                nums = re.findall(r"(?<!\d)\d{1,3}(?!\d)", line)
                threes_list = [x for x in nums if len(x) == 3]
                twos_list = [x for x in nums if len(x) <= 2]
                if threes_list and twos_list:
                    rows.append({
                        "3D": norm3(threes_list[0]),
                        "2D": norm2(twos_list[-1]),
                        "date": page_date
                    })

    # กรองข้อมูลซ้ำ (ใช้ 3D, 2D และ Date เป็นเงื่อนไข เพื่อไม่ให้ประวัติหาย)
    unique_rows = []
    seen = set()
    for r in rows:
        if r["3D"] and r["2D"]:
            k = (r["3D"], r["2D"], r["date"])
            if k not in seen:
                seen.add(k)
                unique_rows.append(r)
                
    return unique_rows, text, soup, page_date

# ============================================================
# CRAWLER
# ============================================================

def score_link_for_category(url, category):
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
        if k.lower() in s: score += 3
    return score

def crawl_blogspot(start_url, category, max_pages=80):
    visited = set()
    queue = [(start_url, 100)]
    collected = []

    while queue and len(visited) < max_pages:
        queue.sort(key=lambda x: x[1], reverse=True)
        url, _ = queue.pop(0)

        if url in visited: continue
        visited.add(url)

        try:
            rows, text, soup, page_date = parse_page(url)
        except Exception:
            continue

        if rows:
            for r in rows:
                collected.append({
                    "source_url": url,
                    "published_date": r["date"] or page_date,
                    "3D": r["3D"],
                    "2D": r["2D"]
                })

        for link in blog_links(url, soup):
            if link in visited: continue
            low = link.lower()
            if any(x in low for x in ["/p/", "/search", "/feeds/", ".xml", "javascript:", "mailto:"]): continue
            
            score = score_link_for_category(link, category)
            if ".html" in low: score += 2
            queue.append((link, score))

    return pd.DataFrame(collected)

# ============================================================
# DATA CLEANING & FEATURES
# ============================================================

def clean_history(df):
    if df.empty: return df
    out = df.copy()

    # จัดการรูปแบบตัวเลขก่อน เพื่อให้ Drop Duplicate ถูกต้อง
    out["3D"] = out["3D"].apply(norm3)
    out["2D"] = out["2D"].apply(norm2)
    out = out.dropna(subset=["3D", "2D"])

    # จัดการ Date Format ให้เสถียร
    out["published_date"] = pd.to_datetime(out["published_date"], errors="coerce", utc=True).dt.tz_localize(None)
    out["published_date"] = out["published_date"].fillna(pd.Timestamp('1970-01-01')) 
    
    # เอา source_url ออกจาก subset เพื่อให้ประวัติใน URL หน้าเดียวกันไม่โดนตัดทิ้ง
    out = out.drop_duplicates(subset=["published_date", "3D", "2D"], keep='last')
    
    # 🌟 เรียงลำดับวันที่จากเก่าไปใหม่ เพื่อให้ Backtest ตรงความจริง
    out = out.sort_values(by="published_date", ascending=True).reset_index(drop=True)
    out["row_id"] = np.arange(len(out))
    
    return out

def make_raw(row):
    a = str(row["3D"]).zfill(3)
    b = str(row["2D"]).zfill(2)
    H, T, O = map(int, a)
    T2, O2 = map(int, b)
    return {
        "H": H, "T": T, "O": O, "T2": T2, "O2": O2,
        "S3": H + T + O, "S2": T2 + O2,
        "HT": abs(H - T), "TO": abs(T - O), "HO": abs(H - O),
        "HT2": abs(H - T2), "HO2": abs(H - O2),
        "TT2": abs(T - T2), "TO2": abs(T - O2),
        "R3": int(a[::-1]), "R2": int(b[::-1]),
        "DS3": H + T + O,
    }

def build_features(data):
    rows = []
    for i in range(len(data)):
        r = {}
        for lag in [1, 2, 3, 5]:
            j = i - lag
            if j >= 0:
                vals = make_raw(data.iloc[j])
                for k, v in vals.items(): r[f"{k}_L{lag}"] = v
        rows.append(r)
    return pd.DataFrame(rows).fillna(0)

def build_next_features(data):
    r = {}
    i = len(data)
    for lag in [1, 2, 3, 5]:
        j = i - lag
        if j >= 0:
            vals = make_raw(data.iloc[j])
            for k, v in vals.items(): r[f"{k}_L{lag}"] = v
    return pd.DataFrame([r]).fillna(0)

# ============================================================
# SYMBOLIC ENGINE (CACHE RESOURCE)
# ============================================================

class Formula:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def calc(self, row):
        try:
            x = self.fn(row)
            if x is None or not np.isfinite(x): return None
            return int(round(x)) % 10
        except Exception:
            return None

@st.cache_resource(show_spinner=False)
def generate_formulas(max_formulas=5000):
    formulas = []
    base = [f"{x}_L{lag}" for lag in [1, 2, 3, 5] for x in ["H", "T", "O", "T2", "O2", "S3", "S2", "HT", "TO", "HO", "HT2", "HO2", "TT2", "TO2", "R3", "R2", "DS3"]]

    for a in base:
        formulas.append(Formula(a, lambda r, a=a: r.get(a, 0)))

    for a, b in itertools.combinations(base, 2):
        formulas.extend([
            Formula(f"({a}+{b})", lambda r, a=a, b=b: r.get(a, 0) + r.get(b, 0)),
            Formula(f"({a}-{b})", lambda r, a=a, b=b: r.get(a, 0) - r.get(b, 0)),
            Formula(f"ABS({a}-{b})", lambda r, a=a, b=b: abs(r.get(a, 0) - r.get(b, 0))),
            Formula(f"({a}*{b})", lambda r, a=a, b=b: r.get(a, 0) * r.get(b, 0)),
            Formula(f"MOD10({a}+{b})", lambda r, a=a, b=b: (r.get(a, 0) + r.get(b, 0)) % 10),
            Formula(f"MOD9({a}+{b})", lambda r, a=a, b=b: (r.get(a, 0) + r.get(b, 0)) % 9),
            Formula(f"MOD10({a}*{b})", lambda r, a=a, b=b: (r.get(a, 0) * r.get(b, 0)) % 10),
            Formula(f"({a}/{b})", lambda r, a=a, b=b: None if r.get(b, 0) == 0 else r.get(a, 0) / r.get(b, 0)),
        ])
        if len(formulas) >= max_formulas: return formulas[:max_formulas]

    important = [x for x in base if x.split("_")[0] in ["H", "T", "O", "T2", "O2", "S3", "S2", "HT", "TO", "HO"]]
    for a, b, c in itertools.combinations(important, 3):
        formulas.extend([
            Formula(f"(({a}+{b})+{c})", lambda r, a=a, b=b, c=c: r.get(a, 0) + r.get(b, 0) + r.get(c, 0)),
            Formula(f"(({a}+{b})-{c})", lambda r, a=a, b=b, c=c: r.get(a, 0) + r.get(b, 0) - r.get(c, 0)),
            Formula(f"(({a}-{b})+{c})", lambda r, a=a, b=b, c=c: r.get(a, 0) - r.get(b, 0) + r.get(c, 0)),
            Formula(f"MOD10(({a}*{b})+{c})", lambda r, a=a, b=b, c=c: (r.get(a, 0) * r.get(b, 0) + r.get(c, 0)) % 10),
            Formula(f"MOD10(({a}+{b})*{c})", lambda r, a=a, b=b, c=c: ((r.get(a, 0) + r.get(b, 0)) * r.get(c, 0)) % 10),
        ])
        if len(formulas) >= max_formulas: break

    return formulas[:max_formulas]

def target_digit(data, i, position):
    s = str(data.iloc[i]["3D"]).zfill(3)
    return int({"H": s[0], "T": s[1], "O": s[2]}[position])

def evaluate_formula_10(formula, features, data, position, end_index=None):
    if end_index is None: end_index = len(data)
    start_index = max(0, end_index - LOCK_WINDOW)
    
    details = []
    for i in range(start_index, end_index):
        if i >= len(features): continue
        row = features.iloc[i].to_dict()
        pred = formula.calc(row)
        actual = target_digit(data, i, position)
        hit = (pred is not None and int(pred) == int(actual))
        details.append({"index": i, "prediction": pred, "actual": actual, "hit": bool(hit)})

    if not details: return {"hit_rate": 0.0, "hits": 0, "total": 0, "details": []}
    hits = sum(x["hit"] for x in details)
    return {"hit_rate": hits / len(details), "hits": hits, "total": len(details), "details": details}

def discover_best_10(data, features, formulas, position, end_index=None, top_candidates=30):
    if end_index is None: end_index = len(data)
    scored = []
    for formula in formulas:
        result = evaluate_formula_10(formula, features, data, position, end_index)
        if result["total"] < LOCK_WINDOW: continue
        scored.append({"formula": formula.name, "hit": result["hit_rate"], "hits": result["hits"], "total": result["total"]})

    if not scored: return []
    scored.sort(key=lambda x: (x["hit"], x["hits"]), reverse=True)
    return scored[:top_candidates]

def create_formula_lock(data, features, formulas):
    lock = {}
    for pos in POSITIONS:
        candidates = discover_best_10(data, features, formulas, pos, end_index=len(data), top_candidates=30)
        if not candidates: continue
        best = candidates[0]
        lock[pos] = {
            "formula": best["formula"], "hits": best["hits"], "total": best["total"],
            "hit_rate": best["hit"], "fail_streak": 0, "history": [], "version": 1
        }
    return lock

def predict_locked(lock, formulas, next_features):
    fmap = {f.name: f for f in formulas}
    row = next_features.iloc[0].to_dict()
    output = {}
    for pos in POSITIONS:
        if pos not in lock:
            output[pos] = None
            continue
        formula_name = lock[pos]["formula"]
        f = fmap.get(formula_name)
        if f is None:
            output[pos] = None
            continue
        output[pos] = f.calc(row)
    return output

def check_locked_draw(lock, formulas, features, data, actual_index):
    fmap = {f.name: f for f in formulas}
    result = {}
    if actual_index >= len(data): return result
    row = features.iloc[actual_index].to_dict()
    
    for pos in POSITIONS:
        if pos not in lock: continue
        formula_name = lock[pos]["formula"]
        f = fmap.get(formula_name)
        if f is None: continue
        
        pred = f.calc(row)
        actual = target_digit(data, actual_index, pos)
        hit = (pred is not None and int(pred) == int(actual))
        result[pos] = {"prediction": pred, "actual": actual, "hit": bool(hit)}
    return result

def update_lock_after_draw(lock, formulas, features, data, actual_index):
    result = check_locked_draw(lock, formulas, features, data, actual_index)
    for pos, info in result.items():
        if info["hit"]: lock[pos]["fail_streak"] = 0
        else: lock[pos]["fail_streak"] += 1
            
        lock[pos]["history"].append({
            "index": actual_index, "prediction": info["prediction"], 
            "actual": info["actual"], "hit": info["hit"]
        })
        lock[pos]["history"] = lock[pos]["history"][-20:]
    return lock, result

def refresh_failed_formulas(lock, data, features, formulas):
    replaced = []
    for pos in POSITIONS:
        if pos not in lock: continue
        if lock[pos]["fail_streak"] < FAIL_LIMIT: continue

        candidates = discover_best_10(data, features, formulas, pos, end_index=len(data), top_candidates=30)
        if not candidates: continue

        old_formula = lock[pos]["formula"]
        selected = next((c for c in candidates if c["formula"] != old_formula), candidates[0])

        lock[pos] = {
            "formula": selected["formula"], "hits": selected["hits"], "total": selected["total"],
            "hit_rate": selected["hit"], "fail_streak": 0, "history": [],
            "version": lock[pos].get("version", 1) + 1, "replaced_from": old_formula
        }
        replaced.append({
            "position": pos, "old": old_formula, "new": selected["formula"],
            "new_hits": selected["hits"], "new_hit_rate": selected["hit"]
        })
    return lock, replaced

def run_lock_backtest(data, features, formulas, start_index):
    if len(data) < (start_index + LOCK_WINDOW): return pd.DataFrame(), {}
    
    working_lock = {}
    history_rows = []
    replacement_log = []
    
    candidates_by_pos = {pos: discover_best_10(data, features, formulas, pos, end_index=start_index, top_candidates=30) for pos in POSITIONS}

    for pos in POSITIONS:
        candidates = candidates_by_pos[pos]
        if not candidates: continue
        best = candidates[0]
        working_lock[pos] = {
            "formula": best["formula"], "hits": best["hits"], "total": best["total"],
            "hit_rate": best["hit"], "fail_streak": 0, "history": [], "version": 1
        }

    fmap = {f.name: f for f in formulas}
    
    for i in range(start_index, len(data)):
        row = features.iloc[i].to_dict()
        draw_result = {"Index": i + 1, "Actual": data.iloc[i]["3D"]}

        for pos in POSITIONS:
            if pos not in working_lock:
                draw_result[f"{pos}_Pred"] = None
                draw_result[f"{pos}_Hit"] = False
                continue

            fname = working_lock[pos]["formula"]
            f = fmap.get(fname)
            pred = f.calc(row) if f is not None else None
            actual = target_digit(data, i, pos)
            hit = (pred is not None and int(pred) == int(actual))

            draw_result[f"{pos}_Pred"] = pred
            draw_result[f"{pos}_Actual"] = actual
            draw_result[f"{pos}_Hit"] = hit

            if hit: working_lock[pos]["fail_streak"] = 0
            else: working_lock[pos]["fail_streak"] += 1

        for pos in POSITIONS:
            if pos in working_lock:
                draw_result[f"{pos}_Formula"] = working_lock[pos]["formula"]
                draw_result[f"{pos}_Fail"] = working_lock[pos]["fail_streak"]

        history_rows.append(draw_result)

        for pos in POSITIONS:
            if pos in working_lock and working_lock[pos]["fail_streak"] >= FAIL_LIMIT:
                candidates = discover_best_10(data, features, formulas, pos, end_index=i + 1, top_candidates=30)
                if candidates:
                    old = working_lock[pos]["formula"]
                    selected = next((c for c in candidates if c["formula"] != old), candidates[0])

                    working_lock[pos] = {
                        "formula": selected["formula"], "hits": selected["hits"], "total": selected["total"],
                        "hit_rate": selected["hit"], "fail_streak": 0, "history": [],
                        "version": working_lock[pos].get("version", 1) + 1, "replaced_from": old
                    }
                    replacement_log.append({
                        "Index": i + 1, "Position": pos, "Old Formula": old,
                        "New Formula": selected["formula"], "New 10D Hits": selected["hits"],
                        "New 10D %": selected["hit"] * 100
                    })

    return pd.DataFrame(history_rows), pd.DataFrame(replacement_log)

# ============================================================
# UI RENDER
# ============================================================

st.title("🧠 LOTTO AI — AUTO SYMBOLIC EQUATION V4")
st.caption("Chronological Order Fix • Overfitting Avoidance • Formula Lock • 2-Fail Auto Replacement")
st.info(f"แหล่งข้อมูล: **{category}**\n\n{BLOG_URLS[category]}")

if st.button("🌐 ดึงข้อมูลจาก Blogspot", type="primary", use_container_width=True):
    with st.spinner("กำลังอ่าน Blogspot และค้นหาโพสต์ย้อนหลัง..."):
        raw = crawl_blogspot(BLOG_URLS[category], category, max_pages=max_pages)
        data = clean_history(raw)
        
        st.session_state["blog_data"] = data
        st.session_state["blog_category"] = category
        for key in ["formula_lock", "results", "features", "formulas"]:
            st.session_state.pop(key, None)
    st.success(f"ดึงข้อมูลได้ {len(data):,} รายการ และจัดเรียงงวดตามเวลาแล้ว")

with st.expander("🔗 เพิ่ม URL Blogspot เอง"):
    custom_url = st.text_input("URL ของหน้า Blogspot")
    if st.button("ดึง URL นี้"):
        if custom_url.strip():
            try:
                raw_rows, _, _, pub_date = parse_page(custom_url.strip())
                custom_df = pd.DataFrame([
                    {
                        "source_url": custom_url.strip(), 
                        "published_date": r["date"] or pub_date, 
                        "3D": r["3D"], 
                        "2D": r["2D"]
                    }
                    for r in raw_rows
                ])
                st.session_state["custom_data"] = clean_history(custom_df)
                st.success(f"พบ {len(custom_df)} รายการ")
            except Exception as e:
                st.error(f"อ่าน URL ไม่สำเร็จ: {e}")

data = st.session_state.get("blog_data", pd.DataFrame())
if st.session_state.get("blog_category") != category: data = pd.DataFrame()
custom_data = st.session_state.get("custom_data", pd.DataFrame())

if not custom_data.empty:
    if data.empty: data = custom_data.copy()
    else:
        data = pd.concat([data, custom_data], ignore_index=True)
        data = data.sort_values(by="published_date", ascending=True)
        data = data.drop_duplicates(subset=["published_date", "3D", "2D"], keep='last').reset_index(drop=True)

if not data.empty:
    st.subheader(f"📊 ข้อมูลที่ดึงได้ {len(data):,} รายการ (เรียงตามเวลาแล้ว)")
    disp_data = data.copy()
    if 'published_date' in disp_data.columns:
        disp_data['Date'] = disp_data['published_date'].dt.strftime('%Y-%m-%d').replace('1970-01-01', 'Unknown')
    
    # 🌟 เลื่อนคอลัมน์ Date มาไว้ด้านหน้าสุดเพื่อให้ดูง่ายขึ้น
    cols = ['Date', '3D', '2D', 'source_url']
    cols = [c for c in cols if c in disp_data.columns] + [c for c in disp_data.columns if c not in cols]
    
    st.dataframe(disp_data[cols].head(100), use_container_width=True, hide_index=True)

    if len(data) < min_history:
        st.warning(f"ข้อมูลมีเพียง {len(data)} งวด ต้องการอย่างน้อย {min_history} งวด")

if not data.empty and len(data) >= min_history:
    st.markdown("---")
    st.header("🧠 SYMBOLIC ENGINE")
    
    if st.button(f"🚀 สร้างสูตร + คัด {LOCK_WINDOW} งวด", use_container_width=True):
        with st.spinner("กำลังสร้างสูตรและทดสอบ..."):
            features = build_features(data)
            formulas = generate_formulas(max_formulas=max_formulas)
            st.session_state["features"] = features
            st.session_state["formulas"] = formulas
            st.session_state["formula_lock"] = create_formula_lock(data, features, formulas)
        st.success(f"สร้างสูตร {len(formulas):,} สูตร และ LOCK สูตร {LOCK_WINDOW} งวดเรียบร้อย")

if "formula_lock" in st.session_state and "formulas" in st.session_state:
    lock = st.session_state["formula_lock"]
    formulas = st.session_state["formulas"]
    st.markdown("---")
    st.header("🔒 LOCKED FORMULAS")
    
    lock_rows = []
    for pos in POSITIONS:
        if pos not in lock: continue
        x = lock[pos]
        status = "🔴 REPLACE" if x["fail_streak"] >= FAIL_LIMIT else "🟠 FAIL 1/2" if x["fail_streak"] == 1 else "🟢 LOCKED"
        lock_rows.append({
            "Position": pos, "Formula": x["formula"],
            f"{LOCK_WINDOW}D Hits": f'{x["hits"]}/{x["total"]}',
            f"{LOCK_WINDOW}D Hit %": round(x["hit_rate"] * 100, 2),
            "Fail Streak": x["fail_streak"], "Version": x.get("version", 1), "Status": status
        })
    st.dataframe(pd.DataFrame(lock_rows), use_container_width=True, hide_index=True)

if "formula_lock" in st.session_state and "formulas" in st.session_state and not data.empty:
    lock = st.session_state["formula_lock"]
    formulas = st.session_state["formulas"]
    next_features = build_next_features(data)
    locked_pred = predict_locked(lock, formulas, next_features)
    
    st.markdown("---")
    st.header("🎯 งวดถัดไป — สูตรที่ LOCK")
    c1, c2, c3 = st.columns(3)
    
    for col, pos, title in zip([c1, c2, c3], POSITIONS, ["🔴 หลักร้อย H", "🟢 หลักสิบ T", "🔵 หลักหน่วย O"]):
        with col:
            digit = locked_pred.get(pos)
            st.subheader(title)
            if digit is None: st.warning("ไม่มีผล")
            else:
                st.metric("Digit", str(digit))
                st.caption(lock[pos]["formula"])

    if all(locked_pred.get(pos) is not None for pos in POSITIONS):
        number = "".join(str(locked_pred[pos]) for pos in POSITIONS)
        st.success(f"🔒 เลขจากสูตร LOCK: **{number}**")

if "formula_lock" in st.session_state and "formulas" in st.session_state and not data.empty:
    st.markdown("---")
    st.header(f"🧪 ตรวจสอบสูตร LOCK — {LOCK_WINDOW} งวดล่าสุด")
    lock = st.session_state["formula_lock"]
    formulas = st.session_state["formulas"]
    features = st.session_state["features"]
    fmap = {f.name: f for f in formulas}
    
    rows = []
    n = min(LOCK_WINDOW, len(data))
    for i in range(len(data) - n, len(data)):
        row = features.iloc[i].to_dict()
        item = {"Index": i + 1, "Actual": data.iloc[i]["3D"]}
        for pos in POSITIONS:
            if pos not in lock: continue
            fname = lock[pos]["formula"]
            f = fmap.get(fname)
            pred = f.calc(row) if f is not None else None
            actual = target_digit(data, i, pos)
            item[f"{pos} Pred"] = pred
            item[f"{pos} Actual"] = actual
            item[f"{pos} Hit"] = (pred == actual)
        rows.append(item)
    
    check = pd.DataFrame(rows)
    st.dataframe(check, use_container_width=True, hide_index=True)

if "formula_lock" in st.session_state and "formulas" in st.session_state and not data.empty:
    st.markdown("---")
    st.header("🔄 ADAPTIVE LOCK CONTROL")
    st.caption("ระบบจะเพิ่ม Fail Streak จากผลจริง และเปลี่ยนเฉพาะหลักที่ผิด 2 งวดติด")
    
    if st.button("🔄 ประมวลผลผลล่าสุด + ตรวจ 2 งวดติด", use_container_width=True):
        lock = st.session_state["formula_lock"]
        formulas = st.session_state["formulas"]
        features = st.session_state["features"]
        last_processed = st.session_state.get("last_processed_index")
        current_index = len(data) - 1

        if last_processed == current_index:
            st.warning("งวดล่าสุดถูกประมวลผลไปแล้ว")
        else:
            lock, check_result = update_lock_after_draw(lock, formulas, features, data, current_index)
            lock, replaced = refresh_failed_formulas(lock, data, features, formulas)
            st.session_state["formula_lock"] = lock
            st.session_state["last_processed_index"] = current_index

            if check_result:
                check_rows = []
                for pos, x in check_result.items():
                    check_rows.append({
                        "Position": pos, "Prediction": x["prediction"], "Actual": x["actual"],
                        "Result": "✅ HIT" if x["hit"] else "❌ MISS", "Fail Streak": lock[pos]["fail_streak"]
                    })
                st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)

            if replaced:
                st.warning("⚠️ มีสูตรที่หลุด 2 งวดติด ระบบเปลี่ยนสูตรเฉพาะหลักนั้นแล้ว")
                st.dataframe(pd.DataFrame(replaced), use_container_width=True, hide_index=True)
            else:
                st.success("🟢 ยังไม่มีหลักใดหลุด 2 งวดติด — สูตรเดิมยัง LOCK")

if not data.empty and "features" in st.session_state and "formulas" in st.session_state:
    st.markdown("---")
    if st.button(f"🧹 Reset LOCK แล้วคัดสูตรใหม่จาก {LOCK_WINDOW} งวด", use_container_width=True):
        st.session_state["formula_lock"] = create_formula_lock(data, st.session_state["features"], st.session_state["formulas"])
        st.session_state.pop("last_processed_index", None)
        st.success(f"สร้าง LOCK ใหม่จาก {LOCK_WINDOW} งวดล่าสุดแล้ว")

if not data.empty and len(data) >= 25 and "features" in st.session_state and "formulas" in st.session_state:
    st.markdown("---")
    st.header("📈 FULL WALK-FORWARD BACKTEST")
    
    backtest_start = st.slider(
        "เริ่มทดสอบที่งวด", LOCK_WINDOW, max(LOCK_WINDOW + 1, len(data) - 1),
        min(max(LOCK_WINDOW, len(data) // 3), max(LOCK_WINDOW + 1, len(data) - 1))
    )
    
    if st.button(f"🧪 RUN {LOCK_WINDOW}-DRAW LOCK BACKTEST", use_container_width=True):
        with st.spinner("กำลังทำ Walk-forward Backtest..."):
            bt, replacements = run_lock_backtest(data, st.session_state["features"], st.session_state["formulas"], backtest_start)
        
        if not bt.empty:
            st.subheader("ผล Walk-forward")
            st.dataframe(bt, use_container_width=True, hide_index=True)
            metric_cols = st.columns(4)
            with metric_cols[0]: st.metric("H Hit %", f"{(bt['H_Hit'].mean() * 100):.2f}%")
            with metric_cols[1]: st.metric("T Hit %", f"{(bt['T_Hit'].mean() * 100):.2f}%")
            with metric_cols[2]: st.metric("O Hit %", f"{(bt['O_Hit'].mean() * 100):.2f}%")
            with metric_cols[3]:
                exact = (bt["H_Hit"] & bt["T_Hit"] & bt["O_Hit"]).mean() * 100
                st.metric("3D Exact %", f"{exact:.2f}%")

            if not replacements.empty:
                st.subheader("🔄 Formula Replacement Log")
                st.dataframe(replacements, use_container_width=True, hide_index=True)
            else:
                st.info("ไม่พบการเปลี่ยนสูตรระหว่าง Backtest")
