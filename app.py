import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
from datetime import datetime, timedelta
import joblib
import hashlib
import os
import glob
import time
import scipy.stats as stats
import tempfile

# --- Machine Learning Modules ---
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# ป้องกันปัญหา Deploy บน Cloud ไม่ผ่าน
HAS_LGBM = False 

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. ตั้งค่าหน้าเพจ & ระบบจัดการฟอนต์ภาษาไทย & Cache
# ==========================================
st.set_page_config(page_title="Ultimate Ensemble V.Max", page_icon="🎯", layout="wide")

@st.cache_resource
def setup_thai_font():
    font_path = 'thsarabunnew-webfont.ttf'
    if not os.path.exists(font_path):
        try: 
            urllib.request.urlretrieve("https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf", font_path)
        except Exception: 
            pass

    if os.path.exists(font_path):  
        fm.fontManager.addfont(font_path)  
        plt.rc('font', family='TH Sarabun New', size=14)  
    else:  
        plt.rc('font', family='Tahoma', size=12)

setup_thai_font()

# ใช้ tempfile เพื่อความเสถียรบน Streamlit Cloud
CACHE_DIR = os.path.join(tempfile.gettempdir(), "lotto_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_old_cache(directory, days=7):
    if not os.path.exists(directory): return
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if os.stat(filepath).st_mtime < now - days * 86400:
                try: os.remove(filepath)
                except: pass

clean_old_cache(CACHE_DIR, 7)

LOTTERY_SOURCES = {
    "1. หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "2. หวยธกส.": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "3. หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "4. หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "5. หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "6. หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "7. หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "8. หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "9. หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "10. หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}

# ==========================================
# 1. ระบบจัดการข้อมูล & Feature Engineering (พร้อม Retry)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_and_clean_data(url):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"ไม่สามารถเชื่อมต่อเว็บไซต์ได้ (Network Error): {str(e)}")

    try: soup = BeautifulSoup(response.text, 'lxml')  
    except: soup = BeautifulSoup(response.text, 'html.parser')  
          
    main_content = soup.find('div', class_=re.compile(r'post-body|entry-content|post-content|content'))  
    if not main_content: main_content = soup  

    text_lines = main_content.get_text(separator='\n').split('\n')  
    extracted = []  
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')  
    num_pattern = re.compile(r'\b(\d{3})\b.*?\b(\d{2})\b|\b(\d{5,6})\b.*?\b(\d{2})\b')  
    current_date = datetime.now().strftime('%Y-%m-%d')  

    for line in text_lines:  
        line = line.strip()  
        if not line: continue  

        date_match = date_pattern.search(line)  
        if date_match: current_date = date_match.group(1).replace('/', '-')  

        num_match = num_pattern.search(line)  
        if num_match:  
            if num_match.group(1) and num_match.group(2): res3d, res2d = num_match.group(1), num_match.group(2)  
            elif num_match.group(3) and num_match.group(4): res3d, res2d = num_match.group(3)[-3:], num_match.group(4)  
            else: continue  
            extracted.append({'Date': current_date, 'Result_3D': res3d, 'Result_2D': res2d})  

    if len(extracted) < 10: raise Exception("ข้อมูลบนเว็บมีน้อยเกินไป (ต่ำกว่า 10 งวด)")  
      
    df = pd.DataFrame(extracted)  
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  
    return df.dropna().sort_values('Date').reset_index(drop=True)  

@st.cache_data
def build_features(df, lags, rolls):
    df_feat = df.copy()
    new_features = {}

    df_feat['H'] = df_feat['Result_3D'].str[0].astype(int)  
    df_feat['T'] = df_feat['Result_3D'].str[1].astype(int)  
    df_feat['O'] = df_feat['Result_3D'].str[2].astype(int)  
    df_feat['T2'] = df_feat['Result_2D'].str[0].astype(int)  
    df_feat['O2'] = df_feat['Result_2D'].str[1].astype(int)  

    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek  
    df_feat['DrawIndex'] = df_feat.index  

    new_features['DigitSum_3D'] = (df_feat['H'].shift(1) + df_feat['T'].shift(1) + df_feat['O'].shift(1)).fillna(0) % 10  
    new_features['Sum_2D'] = (df_feat['T2'].shift(1) + df_feat['O2'].shift(1)).fillna(0) % 10  
    primes = [2, 3, 5, 7]  

    for pos in ['H', 'T', 'O', 'T2', 'O2']:  
        prev = df_feat[pos].shift(1)  

        new_features[f'OddEven_{pos}'] = (prev % 2).fillna(0).astype(int)  
        new_features[f'HighLow_{pos}'] = (prev >= 5).fillna(0).astype(int)  
        new_features[f'Is_Prime_{pos}'] = prev.isin(primes).astype(int)  

        for lag in lags: new_features[f'Lag_{lag}_{pos}'] = df_feat[pos].shift(lag)  

        if f'Lag_1_{pos}' in new_features and f'Lag_2_{pos}' in new_features:  
            new_features[f'Repeat_{pos}'] = (new_features[f'Lag_1_{pos}'] == new_features[f'Lag_2_{pos}']).astype(int)  
            new_features[f'Diff_{pos}'] = (new_features[f'Lag_1_{pos}'] - new_features[f'Lag_2_{pos}']).fillna(0)  

        for w in rolls:  
            new_features[f'EMA_{w}_{pos}'] = prev.ewm(span=w, adjust=False).mean()  
            new_features[f'Roll_Med_{w}_{pos}'] = prev.rolling(w).median().fillna(-1)  
            new_features[f'Roll_Std_{w}_{pos}'] = prev.rolling(w).std().fillna(-1)  
            new_features[f'Momentum_{w}_{pos}'] = (prev - prev.rolling(w).mean()).fillna(0)  

        skips = np.zeros(len(df_feat))  
        ranks = np.zeros(len(df_feat))  
        last_seen = {}  
        pos_values = df_feat[pos].values  
          
        for i in range(len(df_feat)):  
            if i == 0:   
                skips[i] = 100  
                ranks[i] = 5  
                last_seen[pos_values[i]] = i  
            else:  
                val_prev = pos_values[i-1]  
                if val_prev in last_seen: skips[i] = (i - 1) - last_seen[val_prev]  
                else: skips[i] = 100  
                      
                seen_distances = {v: ((i - 1) - last_seen.get(v, -100)) for v in range(10)}  
                sorted_by_dist = sorted(seen_distances.items(), key=lambda x: x[1])  
                rank_dict = {v[0]: rank for rank, v in enumerate(sorted_by_dist)}  
                ranks[i] = rank_dict.get(val_prev, 5)  
                last_seen[val_prev] = i - 1  

        new_features[f'Skip_{pos}'] = skips  
        new_features[f'LastRank_{pos}'] = ranks  

    df_new = pd.DataFrame(new_features, index=df_feat.index)  
    df_feat = pd.concat([df_feat, df_new], axis=1)  
    return df_feat.fillna(-1)

# ==========================================
# 2. ระบบวิเคราะห์ 5 สำนัก (Bayesian & Entropy)
# ==========================================
def get_entropy_weight(probs):
    h = stats.entropy(probs + 1e-9)
    max_h = np.log(10)
    conf = 1.0 - (h / max_h)
    return max(0.1, conf**1.5)

class PositionalEquation:
    def analyze(self, df):
        latest = df.iloc[-1]
        H, T, O = latest['H'], latest['T'], latest['O']
        probs = np.zeros(10)
        for v in [(H + T) % 10, (T + O) % 10, abs(H - O) % 10, (H * T) % 10]: probs[int(v)] += 1.0
        return (probs + 0.1) / (probs + 0.1).sum()

class FrequencyEngine:
    def analyze(self, df, pos):
        series = df[pos].dropna()
        probs = np.zeros(10)
        freq_all = series.value_counts(normalize=True).to_dict()
        freq_10 = series.tail(10).value_counts(normalize=True).to_dict()
        for i in range(10):
            idxs = np.where(series == i)[0]
            skip = (len(series) - 1 - idxs[-1]) if len(idxs) > 0 else len(series)
            probs[i] = (freq_all.get(i, 0) * 0.3) + (freq_10.get(i, 0) * 0.5) + ((1.0 / (skip + 1)) * 0.2)
        return (probs + 0.01) / (probs + 0.01).sum()

class ConditionalSystem:
    def analyze(self, df, pos, next_date):
        probs = np.zeros(10)
        subset = df[(df['DayOfWeek'] == next_date.dayofweek)]
        if len(subset) == 0: subset = df
        freq = subset[pos].value_counts(normalize=True).to_dict()
        for i in range(10): probs[i] = freq.get(i, 0)
        return (probs + 0.01) / (probs + 0.01).sum()

class MarkovChainSystem:
    def analyze(self, df, pos):
        series = df[pos].dropna().values
        n = len(series)

        global_freq = pd.Series(series).value_counts(normalize=True).reindex(range(10), fill_value=0.1).values  
        alpha = 3.0   
          
        trans_1 = np.zeros((10, 10))  
        trans_2 = np.zeros((10, 10, 10))  
        trans_3 = np.zeros((10, 10, 10, 10))  
          
        if n < 2: return np.ones(10) / 10  
          
        for i in range(n-1): trans_1[int(series[i]), int(series[i+1])] += 1.0  
        for i in range(n-2): trans_2[int(series[i]), int(series[i+1]), int(series[i+2])] += 1.0  
        for i in range(n-3): trans_3[int(series[i]), int(series[i+1]), int(series[i+2]), int(series[i+3])] += 1.0  
              
        for i in range(10):  
            trans_1[i] = (trans_1[i] + alpha * global_freq) / (trans_1[i].sum() + alpha)  
            for j in range(10):  
                trans_2[i, j] = (trans_2[i, j] + alpha * global_freq) / (trans_2[i, j].sum() + alpha)  
                for k in range(10):  
                    trans_3[i, j, k] = (trans_3[i, j, k] + alpha * global_freq) / (trans_3[i, j, k].sum() + alpha)  
          
        last_1 = int(series[-1])  
        last_2 = int(series[-2]) if n >= 2 else 0  
        last_3 = int(series[-3]) if n >= 3 else 0  
          
        p1 = trans_1[last_1]  
        p2 = trans_2[last_2, last_1] if n >= 3 else p1  
        p3 = trans_3[last_3, last_2, last_1] if n >= 4 else p2  

        if n >= 500: w1, w2, w3 = 0.20, 0.35, 0.45  
        elif n >= 200: w1, w2, w3 = 0.35, 0.45, 0.20  
        else: w1, w2, w3 = 0.60, 0.40, 0.0  

        probs = (p3 * w3) + (p2 * w2) + (p1 * w1)  
        return probs / probs.sum()

class PatternBacktestSystem:
    def analyze(self, df, pos):
        probs = np.zeros(10)
        if len(df) < 3: return np.ones(10) / 10
        l1, l2 = df[pos].iloc[-1], df[pos].iloc[-2]
        subset = df[(df[f'Lag_1_{pos}'] == l1) & (df[f'Lag_2_{pos}'] == l2)]
        if len(subset) == 0: subset = df[df[f'Lag_1_{pos}'] == l1]
        if len(subset) > 0:
            freq = subset[pos].value_counts(normalize=True).to_dict()
            for i in range(10): probs[i] = freq.get(i, 0)
        return (probs + 0.01) / (probs + 0.01).sum()

# ==========================================
# 3. AI System (Fast TS-Duel & Calibration)
# ==========================================
class AISystem:
    def __init__(self, lottery_id, data_length):
        self.lottery_id = lottery_id
        self.data_length = data_length

        if data_length >= 700: self.trees, self.depth = 120, 6  
        elif data_length >= 400: self.trees, self.depth = 100, 5  
        elif data_length >= 200: self.trees, self.depth = 80, 4  
        else: self.trees, self.depth = 60, 3  

        self.estimators = [  
            ('hgb', HistGradientBoostingClassifier(max_iter=self.trees, max_leaf_nodes=15, min_samples_leaf=3, random_state=42)),  
            ('xgb', XGBClassifier(n_estimators=self.trees, max_depth=max(1, self.depth-1), learning_rate=0.05, subsample=0.8, tree_method="hist", verbosity=0, random_state=42, n_jobs=-1)),  
            ('et', ExtraTreesClassifier(n_estimators=self.trees//2, max_depth=self.depth, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_estimators=self.trees//2, max_depth=self.depth, class_weight='balanced', random_state=42, n_jobs=-1))  
        ]  

        self.model = None  
        self.model_name = "Turbo Calibrated AI"  

    def analyze(self, X_train, y_train, X_next, pos, data_hash, sample_weight=None):  
        model_path = os.path.join(CACHE_DIR, f"m_ai_calib_turbo_{self.lottery_id}_{pos}_{data_hash}.joblib")  

        if not os.path.exists(model_path):  
            for old_file in glob.glob(os.path.join(CACHE_DIR, f"m_ai_calib_turbo_{self.lottery_id}_{pos}_*.joblib")):  
                try: os.remove(old_file)  
                except: pass  

            if len(X_train) > 100:  
                tscv = TimeSeriesSplit(n_splits=2)  
                score_v, score_s = 0, 0  
                  
                for train_idx, val_idx in tscv.split(X_train):  
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]  
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]  
                      
                    voting = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)  
                    voting.fit(X_tr, y_tr)  
                    score_v += log_loss(y_val, voting.predict_proba(X_val), labels=np.arange(10))  
                      
                    stacking = StackingClassifier(estimators=self.estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=50), cv=2, n_jobs=-1)  
                    stacking.fit(X_tr, y_tr)  
                    score_s += log_loss(y_val, stacking.predict_proba(X_val), labels=np.arange(10))  
                  
                if score_v <= score_s:  
                    best_base = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)  
                else:  
                    best_base = StackingClassifier(estimators=self.estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=100), cv=2, n_jobs=-1)  
            else:  
                best_base = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)  

            calib_method = 'isotonic' if len(X_train) >= 200 else 'sigmoid'  
            calib_cv = 3 if len(X_train) >= 150 else 2  
              
            self.model = CalibratedClassifierCV(best_base, method=calib_method, cv=calib_cv)  
              
            try: self.model.fit(X_train, y_train, sample_weight=sample_weight)  
            except: self.model.fit(X_train, y_train)  

            joblib.dump(self.model, model_path)  
        else:  
            self.model = joblib.load(model_path)  

        probs = self.model.predict_proba(X_next)[0]  
        res = np.zeros(10)  
        for c, p in zip(self.model.classes_, probs): res[int(c)] = p  
        return res / res.sum()

# ==========================================
# 4. Ensemble Engine
# ==========================================
class EnsembleEngine:
    def __init__(self, df_raw, lottery_name, target_dow=None):
        self.df_raw = df_raw
        self.target_dow = target_dow
        self.lottery_name = lottery_name
        self.lottery_id = lottery_name.split(".")[0].strip()
        n = len(df_raw)

        if n >= 700: self.test_size = 15  
        elif n >= 400: self.test_size = 12  
        elif n >= 200: self.test_size = 8  
        else: self.test_size = 4  

        if n < 100: self.test_size = min(3, max(0, n - 30))  

        self.lags = [1, 2, 3] if n < 200 else [1, 2, 3, 5]  
        self.rolls = [3, 5, 10]  

        self.features = ['DayOfWeek', 'DrawIndex', 'DigitSum_3D', 'Sum_2D']  
                           
        for pos in ['H', 'T', 'O', 'T2', 'O2']:  
            self.features.extend([f'OddEven_{pos}', f'HighLow_{pos}', f'Is_Prime_{pos}', f'Skip_{pos}', f'LastRank_{pos}'])  
            for lag in self.lags: self.features.append(f'Lag_{lag}_{pos}')  
            self.features.extend([f'Diff_{pos}', f'Repeat_{pos}'])  
            for w in self.rolls:   
                self.features.extend([f'EMA_{w}_{pos}', f'Roll_Med_{w}_{pos}', f'Roll_Std_{w}_{pos}', f'Momentum_{w}_{pos}'])  

        hash_array = pd.util.hash_pandas_object(df_raw[['Result_3D', 'Result_2D']], index=False).values  
        base_hash = hashlib.md5(hash_array).hexdigest()  
        self.data_hash = f"{base_hash}_{self.test_size}_turbo_vmax"  

        self.pos_sys, self.freq_sys = PositionalEquation(), FrequencyEngine()  
        self.cond_sys, self.markov_sys = ConditionalSystem(), MarkovChainSystem()  
        self.ptn_sys = PatternBacktestSystem()  
          
        self.ai_sys = AISystem(self.lottery_id, n)  
        self.base_weights = {'AI': 0.40, 'Freq': 0.15, 'Markov': 0.15, 'Cal': 0.10, 'BT': 0.10, 'Eq': 0.10}  

    def _process_single_position(self, pos, df_hist, X_all, next_x, next_date):  
        bt_size = self.test_size  
        cache_key = os.path.join(CACHE_DIR, f"bt_turbo_{self.lottery_id}_{pos}_{self.data_hash}.joblib")  
        n = len(df_hist)  

        train_len = len(X_all) - bt_size if bt_size > 0 else len(X_all)  
        X_train_full = X_all.iloc[:train_len]  
        y_train_full = df_hist[pos].iloc[:train_len]  

        valid_features = [f for f in X_all.columns if f in X_train_full.columns]  
          
        mi_scores = mutual_info_classif(X_train_full, y_train_full, random_state=42)  
        mi_series = pd.Series(mi_scores, index=valid_features)  
          
        if n >= 700: mi_thresh = 0.010  
        elif n >= 400: mi_thresh = 0.008  
        elif n >= 200: mi_thresh = 0.005  
        else: mi_thresh = 0.002  
          
        target_feats = min(len(valid_features), max(30, int(n * 0.15)))  
        pre_selected = mi_series[mi_series > mi_thresh].sort_values(ascending=False).head(target_feats * 2).index  
        if len(pre_selected) < 10: pre_selected = valid_features[:target_feats]  
          
        corr_thresh = max(0.75, 0.95 - (n / 5000.0))  
          
        corr_matrix = X_all[pre_selected].corr().abs()  
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  
        to_drop = set()  
        for col in upper_tri.columns:  
            high_corr = upper_tri.index[upper_tri[col] > corr_thresh].tolist()  
            for r in high_corr:  
                if mi_series[col] > mi_series[r]: to_drop.add(r)  
                else: to_drop.add(col)  
          
        selected_feats = [f for f in pre_selected if f not in to_drop][:target_feats]  
        self.final_feat_count = len(selected_feats)  
          
        X_all_fs = X_all[selected_feats]  
        next_x_fs = next_x[selected_feats]  

        if n >= 700: decay_factor = 2.5  
        elif n >= 400: decay_factor = 2.0  
        elif n >= 200: decay_factor = 1.6  
        else: decay_factor = 1.2  
          
        full_sample_weights = np.exp(np.linspace(-decay_factor, 0, len(X_all_fs)))  

        if os.path.exists(cache_key):  
            norm_weights, bt_msg = joblib.load(cache_key)  
        elif len(df_hist) < bt_size + 30 or bt_size <= 0:  
            norm_weights, bt_msg = self.base_weights, "(ข้อมูลน้อย ข้าม Backtest)"  
        else:  
            scores = {k: 0.0 for k in self.base_weights.keys()}  
            total_steps_weight = 0.0  
              
            lite_trees = max(20, self.ai_sys.trees // 3)  
            lite_estimators = [  
                ('hgb', HistGradientBoostingClassifier(max_iter=lite_trees, max_leaf_nodes=15, min_samples_leaf=3, random_state=42)),  
                ('xgb', XGBClassifier(n_estimators=lite_trees, max_depth=max(1, self.ai_sys.depth-1), learning_rate=0.05, subsample=0.8, tree_method="hist", verbosity=0, random_state=42, n_jobs=-1)),  
                ('et', ExtraTreesClassifier(n_estimators=lite_trees, max_depth=self.ai_sys.depth, class_weight='balanced', random_state=42, n_jobs=-1))  
            ]  
              
            if n >= 500:  
                bt_ai_base = StackingClassifier(estimators=lite_estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=50), cv=2, n_jobs=-1)  
            else:  
                bt_ai_base = VotingClassifier(estimators=lite_estimators, voting='soft', n_jobs=-1)  
                  
            calib_method = 'isotonic' if n >= 200 else 'sigmoid'  
            bt_ai_model = CalibratedClassifierCV(bt_ai_base, method=calib_method, cv=2)  

            for i in range(bt_size):  
                curr_train_len = len(X_all_fs) - bt_size + i  
                X_train_step = X_all_fs.iloc[:curr_train_len]  
                y_train_step = df_hist[pos].iloc[:curr_train_len]  
                step_weights = full_sample_weights[:curr_train_len]  
                X_test_step = X_all_fs.iloc[[curr_train_len]]  
                actual_val = df_hist[pos].iloc[curr_train_len]  

                try: bt_ai_model.fit(X_train_step, y_train_step, sample_weight=step_weights)  
                except: bt_ai_model.fit(X_train_step, y_train_step)  

                probs_ai = bt_ai_model.predict_proba(X_test_step)[0]  
                ai_res = np.zeros(10)  
                for idx, c in enumerate(bt_ai_model.classes_): ai_res[int(c)] = probs_ai[idx]  

                curr_df = df_hist.iloc[:curr_train_len]  
                target_date = df_hist.iloc[curr_train_len]['Date']  

                sys_probs = {  
                    'AI': ai_res,  
                    'Freq': self.freq_sys.analyze(curr_df, pos),  
                    'Cal': self.cond_sys.analyze(curr_df, pos, target_date),  
                    'Markov': self.markov_sys.analyze(curr_df, pos),  
                    'BT': self.ptn_sys.analyze(curr_df, pos),  
                    'Eq': self.pos_sys.analyze(curr_df)  
                }  
                  
                step_weight = np.exp((i - bt_size + 1) * 0.15)   
                total_steps_weight += step_weight  

                for sys_name, p in sys_probs.items():  
                    ranked = np.argsort(p)[::-1]  
                    top1 = 1 if actual_val == ranked[0] else 0  
                    top3 = 1 if actual_val in ranked[:3] else 0  
                    top5 = 1 if actual_val in ranked[:5] else 0  
                      
                    ll = -np.log(p[actual_val] + 1e-9)  
                    brier = np.sum((p - np.eye(10)[actual_val])**2)  
                      
                    metric_score = (top1*0.4 + top3*0.3 + top5*0.1) + (1.0 / (ll + 1.0))*0.1 + (1.0 - (brier/2.0))*0.1  
                    scores[sys_name] += (metric_score * step_weight)  

            total_score = sum(scores.values())  
            norm_weights = {k: v/total_score for k, v in scores.items()}  
            msg = f"(Turbo-BT: AI {norm_weights['AI']*100:.1f}% | Bayesian Markov {norm_weights['Markov']*100:.1f}%)"  

            joblib.dump((norm_weights, msg), cache_key)  
            bt_msg = msg  

        p_ai = self.ai_sys.analyze(X_all_fs, df_hist[pos], next_x_fs, pos, self.data_hash, sample_weight=full_sample_weights)  
        p_fq = self.freq_sys.analyze(df_hist, pos)  
        p_cal = self.cond_sys.analyze(df_hist, pos, next_date)  
        p_mk = self.markov_sys.analyze(df_hist, pos)  
        p_bt = self.ptn_sys.analyze(df_hist, pos)  
        p_eq = self.pos_sys.analyze(df_hist)  

        ent_weights = {  
            'AI': get_entropy_weight(p_ai),  
            'Freq': get_entropy_weight(p_fq),  
            'Cal': get_entropy_weight(p_cal),  
            'Markov': get_entropy_weight(p_mk),  
            'BT': get_entropy_weight(p_bt),  
            'Eq': get_entropy_weight(p_eq)  
        }  

        W = norm_weights.copy()  
        W = {k: W[k] * ent_weights[k] for k in W}  
        total_w = sum(W.values())  
        W = {k: v/total_w for k, v in W.items()}  

        final_score = (W['AI']*p_ai + W['Freq']*p_fq + W['Cal']*p_cal + W['Markov']*p_mk + W['BT']*p_bt + W['Eq']*p_eq)  
        final_score = final_score / final_score.sum()  

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]  

        calib_msg = f" [Turbo Applied]"  

        return pos, {  
            'AI': get_top5(p_ai),  
            'Calendar': get_top5(p_cal),  
            'Markov': get_top5(p_mk),  
            'Final': get_top5(final_score),  
            'Probs_For_Graph': final_score,  
            'BT_Msg': bt_msg + calib_msg,  
            'Feat_Count': getattr(self, 'final_feat_count', 0)  
        }  

    def predict_all(self, st_progress_bar):  
        last_date = self.df_raw['Date'].iloc[-1]  
        if self.target_dow is not None:  
            days_ahead = self.target_dow - last_date.dayofweek  
            if days_ahead <= 0: days_ahead += 7  
            next_date = last_date + timedelta(days=days_ahead)  
        else:  
            next_date = last_date + timedelta(days=7 if len(self.df_raw) <= 1 else (last_date - self.df_raw['Date'].iloc[-2]).days)  

        dummy = pd.DataFrame([{'Date': next_date, 'Result_3D': '000', 'Result_2D': '00'}])  
        df_ext = pd.concat([self.df_raw, dummy], ignore_index=True)  

        df_ext = build_features(df_ext, self.lags, self.rolls)  
        next_x = df_ext.iloc[[-1]][self.features]  
        df_hist = df_ext.iloc[:-1]  
        X_all = df_hist[self.features]  

        results = []  
        positions = ['H', 'T', 'O', 'T2', 'O2']  
          
        for i, pos in enumerate(positions):  
            progress_percent = int(((i + 1) / 5) * 100)
            st_progress_bar.progress(progress_percent, text=f'กำลังประมวลผลโมเดล {self.ai_sys.model_name}: ตำแหน่ง {pos}...')
            res = self._process_single_position(pos, df_hist, X_all, next_x, next_date)  
            results.append(res)  
              
        predictions = {pos: data for pos, data in results}  
        return predictions, next_date

# ==========================================
# 5. Dashboard (UI ของ Streamlit)
# ==========================================
st.title("🎯 ระบบวิเคราะห์เลขเด่น Ultimate Ensemble V.Max")
st.markdown("*(Turbo Quantum Edition)*")

col1, col2 = st.columns(2)
with col1:
    selected_lotto = st.selectbox("🎯 เลือกหวย:", list(LOTTERY_SOURCES.keys()))
with col2:
    day_options = {
        'อัตโนมัติ (คำนวณจากงวดล่าสุด)': None, 'วันจันทร์': 0, 'วันอังคาร': 1, 
        'วันพุธ': 2, 'วันพฤหัสบดี': 3, 'วันศุกร์': 4, 'วันเสาร์': 5, 'วันอาทิตย์': 6
    }
    selected_day_name = st.selectbox("📅 ออกวัน:", list(day_options.keys()))
    target_dow = day_options[selected_day_name]

if st.button("🚀 วิเคราะห์เลขเด่น (Turbo Speed)", type="primary", use_container_width=True):
    url = LOTTERY_SOURCES[selected_lotto]
    
    try:
        with st.spinner("กำลังดึงและเตรียมข้อมูลจากแหล่งอ้างอิง..."):
            df_raw = fetch_and_clean_data(url)
            engine = EnsembleEngine(df_raw, selected_lotto, target_dow=target_dow)

        st.write("---")
        progress_bar = st.progress(0, text="เตรียมเริ่มการวิเคราะห์โมเดล AI...")
        
        preds, next_date = engine.predict_all(progress_bar)
        
        # ทำให้ Progress Bar หายไปเมื่อทำเสร็จ
        progress_bar.empty()
        
        st.success("✨ วิเคราะห์เสร็จสิ้นสมบูรณ์!")
        
        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]  
        labels = {'H': 'หลักร้อย (บน)', 'T': 'หลักสิบ (บน)', 'O': 'หลักหน่วย (บน)', 'T2': 'หลักสิบ (ล่าง)', 'O2': 'หลักหน่วย (ล่าง)'}  

        probs_top = (preds['H']['Probs_For_Graph'] + preds['T']['Probs_For_Graph'] + preds['O']['Probs_For_Graph']) / 3  
        probs_bot = (preds['T2']['Probs_For_Graph'] + preds['O2']['Probs_For_Graph']) / 2  

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]  
        top5_top = get_top5(probs_top)  
        top5_bot = get_top5(probs_bot)  

        st.subheader("🔥 สรุปฟันธง เลขเด่นมาแรง (Quantum Computed Probabilities)")
        top_str = " , ".join([str(x[0]) for x in top5_top])
        bot_str = " , ".join([str(x[0]) for x in top5_bot])
        st.info(f"**🚀 เด่นบนรวม (ร้อย-สิบ-หน่วย) : {top_str}**")
        st.info(f"**⬇️ เด่นล่างรวม (สิบ-หน่วย) : {bot_str}**")
        
        st.write(f"🔮 ผลการวิเคราะห์ระดับลึก ประจำวัน{dow_names[next_date.dayofweek]}ที่ {next_date.strftime('%d-%m-%Y')} (ใช้ข้อมูล {len(df_raw)} งวด)")

        for pos in ['H', 'T', 'O', 'T2', 'O2']:  
            with st.expander(f"📍 เจาะลึกตำแหน่ง: {labels[pos]}"):
                st.caption(f"คัดเฉพาะฟีเจอร์เด่นสุด {preds[pos]['Feat_Count']} ตัว | {preds[pos]['BT_Msg']}")
                
                nums_ai = ", ".join([str(num) for num, prob in preds[pos]['AI']])  
                nums_day = ", ".join([str(num) for num, prob in preds[pos]['Calendar']])  
                nums_mk = ", ".join([str(num) for num, prob in preds[pos]['Markov']])  
                nums_final = ", ".join([str(num) for num, prob in preds[pos]['Final']])  

                st.markdown(f"- 🧠 **เลขเด่น Quantum AI:** {nums_ai}")  
                st.markdown(f"- 🔗 **เลขเด่น มาร์คอฟแบบเบย์:** {nums_mk}")  
                st.markdown(f"- 📅 **เลขเด่น กำลังวัน:** {nums_day}")  
                st.markdown(f"- 🌟 **เด่นสรุปรวม 5 ตัว:** {nums_final}")  

        # กราฟ Matplotlib (พร้อมเพิ่ม plt.close(fig) ป้องกันหน่วยความจำสะสม)
        st.subheader("📊 กราฟโอกาสความน่าจะเป็น (Probabilities)")
        fig = plt.figure(figsize=(12, 8))  
        fig.suptitle(f'Quantum Precision Probabilities - {selected_lotto}', fontsize=14, fontweight='bold')  
        colors_list = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  
        
        for idx, pos in enumerate(['H', 'T', 'O', 'T2', 'O2']):  
            ax = plt.subplot(2, 3, idx + 1)  
            top_5_items = preds[pos]['Final']  
            ax.bar([str(x[0]) for x in top_5_items], [x[1]*100 for x in top_5_items], color=colors_list)  
            ax.set_title(labels[pos])  
            ax.set_ylabel('โอกาส (%)')  
            
        plt.tight_layout()  
        st.pyplot(fig)  
        plt.close(fig) # ป้องกันหน่วยความจำค้าง

    except requests.exceptions.RequestException as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อเว็บไซต์ (Network Error): {str(e)}")
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดของระบบ: {str(e)}")
# 0. ตั้งค่าหน้าเพจ & ระบบจัดการฟอนต์ภาษาไทย
# ==========================================
st.set_page_config(page_title="Ultimate Ensemble V.Max", page_icon="🎯", layout="wide")

@st.cache_resource
def setup_thai_font():
    font_path = 'thsarabunnew-webfont.ttf'
    if not os.path.exists(font_path):
        try: 
            urllib.request.urlretrieve("https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf", font_path)
        except Exception: 
            pass

    if os.path.exists(font_path):  
        fm.fontManager.addfont(font_path)  
        plt.rc('font', family='TH Sarabun New', size=14)  
    else:  
        plt.rc('font', family='Tahoma', size=12)

setup_thai_font()

CACHE_DIR = 'model_cache_vmax_turbo'
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_old_cache(directory, days=7):
    if not os.path.exists(directory): return
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if os.stat(filepath).st_mtime < now - days * 86400:
                try: os.remove(filepath)
                except: pass

clean_old_cache(CACHE_DIR, 7)

LOTTERY_SOURCES = {
    "1. หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "2. หวยธกส.": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "3. หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "4. หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "5. หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "6. หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "7. หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "8. หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "9. หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "10. หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}

# ==========================================
# 1. ระบบจัดการข้อมูล & Feature Engineering
# ==========================================
@st.cache_data(ttl=3600)
def fetch_and_clean_data(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    try: soup = BeautifulSoup(response.text, 'lxml')  
    except: soup = BeautifulSoup(response.text, 'html.parser')  
          
    main_content = soup.find('div', class_=re.compile(r'post-body|entry-content|post-content|content'))  
    if not main_content: main_content = soup  

    text_lines = main_content.get_text(separator='\n').split('\n')  
    extracted = []  
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')  
    num_pattern = re.compile(r'\b(\d{3})\b.*?\b(\d{2})\b|\b(\d{5,6})\b.*?\b(\d{2})\b')  
    current_date = datetime.now().strftime('%Y-%m-%d')  

    for line in text_lines:  
        line = line.strip()  
        if not line: continue  

        date_match = date_pattern.search(line)  
        if date_match: current_date = date_match.group(1).replace('/', '-')  

        num_match = num_pattern.search(line)  
        if num_match:  
            if num_match.group(1) and num_match.group(2): res3d, res2d = num_match.group(1), num_match.group(2)  
            elif num_match.group(3) and num_match.group(4): res3d, res2d = num_match.group(3)[-3:], num_match.group(4)  
            else: continue  
            extracted.append({'Date': current_date, 'Result_3D': res3d, 'Result_2D': res2d})  

    if len(extracted) < 10: raise Exception("ข้อมูลบนเว็บมีน้อยเกินไป (ต่ำกว่า 10 งวด)")  
      
    df = pd.DataFrame(extracted)  
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  
    return df.dropna().sort_values('Date').reset_index(drop=True)  

@st.cache_data
def build_features(df, lags, rolls):
    df_feat = df.copy()
    new_features = {}

    df_feat['H'] = df_feat['Result_3D'].str[0].astype(int)  
    df_feat['T'] = df_feat['Result_3D'].str[1].astype(int)  
    df_feat['O'] = df_feat['Result_3D'].str[2].astype(int)  
    df_feat['T2'] = df_feat['Result_2D'].str[0].astype(int)  
    df_feat['O2'] = df_feat['Result_2D'].str[1].astype(int)  

    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek  
    df_feat['DrawIndex'] = df_feat.index  

    new_features['DigitSum_3D'] = (df_feat['H'].shift(1) + df_feat['T'].shift(1) + df_feat['O'].shift(1)).fillna(0) % 10  
    new_features['Sum_2D'] = (df_feat['T2'].shift(1) + df_feat['O2'].shift(1)).fillna(0) % 10  
    primes = [2, 3, 5, 7]  

    for pos in ['H', 'T', 'O', 'T2', 'O2']:  
        prev = df_feat[pos].shift(1)  

        new_features[f'OddEven_{pos}'] = (prev % 2).fillna(0).astype(int)  
        new_features[f'HighLow_{pos}'] = (prev >= 5).fillna(0).astype(int)  
        new_features[f'Is_Prime_{pos}'] = prev.isin(primes).astype(int)  

        for lag in lags: new_features[f'Lag_{lag}_{pos}'] = df_feat[pos].shift(lag)  

        if f'Lag_1_{pos}' in new_features and f'Lag_2_{pos}' in new_features:  
            new_features[f'Repeat_{pos}'] = (new_features[f'Lag_1_{pos}'] == new_features[f'Lag_2_{pos}']).astype(int)  
            new_features[f'Diff_{pos}'] = (new_features[f'Lag_1_{pos}'] - new_features[f'Lag_2_{pos}']).fillna(0)  

        for w in rolls:  
            new_features[f'EMA_{w}_{pos}'] = prev.ewm(span=w, adjust=False).mean()  
            new_features[f'Roll_Med_{w}_{pos}'] = prev.rolling(w).median().fillna(-1)  
            new_features[f'Roll_Std_{w}_{pos}'] = prev.rolling(w).std().fillna(-1)  
            new_features[f'Momentum_{w}_{pos}'] = (prev - prev.rolling(w).mean()).fillna(0)  

        skips = np.zeros(len(df_feat))  
        ranks = np.zeros(len(df_feat))  
        last_seen = {}  
        pos_values = df_feat[pos].values  
          
        for i in range(len(df_feat)):  
            if i == 0:   
                skips[i] = 100  
                ranks[i] = 5  
                last_seen[pos_values[i]] = i  
            else:  
                val_prev = pos_values[i-1]  
                if val_prev in last_seen: skips[i] = (i - 1) - last_seen[val_prev]  
                else: skips[i] = 100  
                      
                seen_distances = {v: ((i - 1) - last_seen.get(v, -100)) for v in range(10)}  
                sorted_by_dist = sorted(seen_distances.items(), key=lambda x: x[1])  
                rank_dict = {v[0]: rank for rank, v in enumerate(sorted_by_dist)}  
                ranks[i] = rank_dict.get(val_prev, 5)  
                last_seen[val_prev] = i - 1  

        new_features[f'Skip_{pos}'] = skips  
        new_features[f'LastRank_{pos}'] = ranks  

    df_new = pd.DataFrame(new_features, index=df_feat.index)  
    df_feat = pd.concat([df_feat, df_new], axis=1)  
    return df_feat.fillna(-1)

# ==========================================
# 2. ระบบวิเคราะห์ 5 สำนัก (Bayesian & Entropy)
# ==========================================
def get_entropy_weight(probs):
    h = stats.entropy(probs + 1e-9)
    max_h = np.log(10)
    conf = 1.0 - (h / max_h)
    return max(0.1, conf**1.5)

class PositionalEquation:
    def analyze(self, df):
        latest = df.iloc[-1]
        H, T, O = latest['H'], latest['T'], latest['O']
        probs = np.zeros(10)
        for v in [(H + T) % 10, (T + O) % 10, abs(H - O) % 10, (H * T) % 10]: probs[int(v)] += 1.0
        return (probs + 0.1) / (probs + 0.1).sum()

class FrequencyEngine:
    def analyze(self, df, pos):
        series = df[pos].dropna()
        probs = np.zeros(10)
        freq_all = series.value_counts(normalize=True).to_dict()
        freq_10 = series.tail(10).value_counts(normalize=True).to_dict()
        for i in range(10):
            idxs = np.where(series == i)[0]
            skip = (len(series) - 1 - idxs[-1]) if len(idxs) > 0 else len(series)
            probs[i] = (freq_all.get(i, 0) * 0.3) + (freq_10.get(i, 0) * 0.5) + ((1.0 / (skip + 1)) * 0.2)
        return (probs + 0.01) / (probs + 0.01).sum()

class ConditionalSystem:
    def analyze(self, df, pos, next_date):
        probs = np.zeros(10)
        subset = df[(df['DayOfWeek'] == next_date.dayofweek)]
        if len(subset) == 0: subset = df
        freq = subset[pos].value_counts(normalize=True).to_dict()
        for i in range(10): probs[i] = freq.get(i, 0)
        return (probs + 0.01) / (probs + 0.01).sum()

class MarkovChainSystem:
    def analyze(self, df, pos):
        series = df[pos].dropna().values
        n = len(series)

        global_freq = pd.Series(series).value_counts(normalize=True).reindex(range(10), fill_value=0.1).values  
        alpha = 3.0   
          
        trans_1 = np.zeros((10, 10))  
        trans_2 = np.zeros((10, 10, 10))  
        trans_3 = np.zeros((10, 10, 10, 10))  
          
        if n < 2: return np.ones(10) / 10  
          
        for i in range(n-1): trans_1[int(series[i]), int(series[i+1])] += 1.0  
        for i in range(n-2): trans_2[int(series[i]), int(series[i+1]), int(series[i+2])] += 1.0  
        for i in range(n-3): trans_3[int(series[i]), int(series[i+1]), int(series[i+2]), int(series[i+3])] += 1.0  
              
        for i in range(10):  
            trans_1[i] = (trans_1[i] + alpha * global_freq) / (trans_1[i].sum() + alpha)  
            for j in range(10):  
                trans_2[i, j] = (trans_2[i, j] + alpha * global_freq) / (trans_2[i, j].sum() + alpha)  
                for k in range(10):  
                    trans_3[i, j, k] = (trans_3[i, j, k] + alpha * global_freq) / (trans_3[i, j, k].sum() + alpha)  
          
        last_1 = int(series[-1])  
        last_2 = int(series[-2]) if n >= 2 else 0  
        last_3 = int(series[-3]) if n >= 3 else 0  
          
        p1 = trans_1[last_1]  
        p2 = trans_2[last_2, last_1] if n >= 3 else p1  
        p3 = trans_3[last_3, last_2, last_1] if n >= 4 else p2  

        if n >= 500: w1, w2, w3 = 0.20, 0.35, 0.45  
        elif n >= 200: w1, w2, w3 = 0.35, 0.45, 0.20  
        else: w1, w2, w3 = 0.60, 0.40, 0.0  

        probs = (p3 * w3) + (p2 * w2) + (p1 * w1)  
        return probs / probs.sum()

class PatternBacktestSystem:
    def analyze(self, df, pos):
        probs = np.zeros(10)
        if len(df) < 3: return np.ones(10) / 10
        l1, l2 = df[pos].iloc[-1], df[pos].iloc[-2]
        subset = df[(df[f'Lag_1_{pos}'] == l1) & (df[f'Lag_2_{pos}'] == l2)]
        if len(subset) == 0: subset = df[df[f'Lag_1_{pos}'] == l1]
        if len(subset) > 0:
            freq = subset[pos].value_counts(normalize=True).to_dict()
            for i in range(10): probs[i] = freq.get(i, 0)
        return (probs + 0.01) / (probs + 0.01).sum()

# ==========================================
# 3. AI System (Fast TS-Duel & Calibration)
# ==========================================
class AISystem:
    def __init__(self, lottery_id, data_length):
        self.lottery_id = lottery_id
        self.data_length = data_length

        if data_length >= 700: self.trees, self.depth = 120, 6  
        elif data_length >= 400: self.trees, self.depth = 100, 5  
        elif data_length >= 200: self.trees, self.depth = 80, 4  
        else: self.trees, self.depth = 60, 3  

        self.estimators = [  
            ('hgb', HistGradientBoostingClassifier(max_iter=self.trees, max_leaf_nodes=15, min_samples_leaf=3, random_state=42)),  
            ('xgb', XGBClassifier(n_estimators=self.trees, max_depth=max(1, self.depth-1), learning_rate=0.05, subsample=0.8, tree_method="hist", verbosity=0, random_state=42, n_jobs=-1)),  
            ('et', ExtraTreesClassifier(n_estimators=self.trees//2, max_depth=self.depth, class_weight='balanced', random_state=42, n_jobs=-1))  
        ]  
          
        if HAS_LGBM:  
            self.estimators.insert(0, ('lgbm', LGBMClassifier(n_estimators=self.trees, max_depth=self.depth, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)))  
        else:  
            self.estimators.insert(0, ('rf', RandomForestClassifier(n_estimators=self.trees//2, max_depth=self.depth, class_weight='balanced', random_state=42, n_jobs=-1)))  

        self.model = None  
        self.model_name = "Turbo Calibrated AI"  

    def analyze(self, X_train, y_train, X_next, pos, data_hash, sample_weight=None):  
        model_path = os.path.join(CACHE_DIR, f"m_ai_calib_turbo_{self.lottery_id}_{pos}_{data_hash}.joblib")  

        if not os.path.exists(model_path):  
            for old_file in glob.glob(os.path.join(CACHE_DIR, f"m_ai_calib_turbo_{self.lottery_id}_{pos}_*.joblib")):  
                try: os.remove(old_file)  
                except: pass  

            if len(X_train) > 100:  
                tscv = TimeSeriesSplit(n_splits=2)  
                score_v, score_s = 0, 0  
                  
                for train_idx, val_idx in tscv.split(X_train):  
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]  
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]  
                      
                    voting = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)  
                    voting.fit(X_tr, y_tr)  
                    score_v += log_loss(y_val, voting.predict_proba(X_val), labels=np.arange(10))  
                      
                    stacking = StackingClassifier(estimators=self.estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=50), cv=2, n_jobs=-1)  
                    stacking.fit(X_tr, y_tr)  
                    score_s += log_loss(y_val, stacking.predict_proba(X_val), labels=np.arange(10))  
                  
                if score_v <= score_s:  
                    best_base = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)  
                else:  
                    best_base = StackingClassifier(estimators=self.estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=100), cv=2, n_jobs=-1)  
            else:  
                best_base = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)  

            calib_method = 'isotonic' if len(X_train) >= 200 else 'sigmoid'  
            calib_cv = 3 if len(X_train) >= 150 else 2  
              
            self.model = CalibratedClassifierCV(best_base, method=calib_method, cv=calib_cv)  
              
            try: self.model.fit(X_train, y_train, sample_weight=sample_weight)  
            except: self.model.fit(X_train, y_train)  

            joblib.dump(self.model, model_path)  
        else:  
            self.model = joblib.load(model_path)  

        probs = self.model.predict_proba(X_next)[0]  
        res = np.zeros(10)  
        for c, p in zip(self.model.classes_, probs): res[int(c)] = p  
        return res / res.sum()

# ==========================================
# 4. Ensemble Engine
# ==========================================
class EnsembleEngine:
    def __init__(self, df_raw, lottery_name, target_dow=None):
        self.df_raw = df_raw
        self.target_dow = target_dow
        self.lottery_name = lottery_name
        self.lottery_id = lottery_name.split(".")[0].strip()
        n = len(df_raw)

        if n >= 700: self.test_size = 15  
        elif n >= 400: self.test_size = 12  
        elif n >= 200: self.test_size = 8  
        else: self.test_size = 4  

        if n < 100: self.test_size = min(3, max(0, n - 30))  

        self.lags = [1, 2, 3] if n < 200 else [1, 2, 3, 5]  
        self.rolls = [3, 5, 10]  

        self.features = ['DayOfWeek', 'DrawIndex', 'DigitSum_3D', 'Sum_2D']  
                           
        for pos in ['H', 'T', 'O', 'T2', 'O2']:  
            self.features.extend([f'OddEven_{pos}', f'HighLow_{pos}', f'Is_Prime_{pos}', f'Skip_{pos}', f'LastRank_{pos}'])  
            for lag in self.lags: self.features.append(f'Lag_{lag}_{pos}')  
            self.features.extend([f'Diff_{pos}', f'Repeat_{pos}'])  
            for w in self.rolls:   
                self.features.extend([f'EMA_{w}_{pos}', f'Roll_Med_{w}_{pos}', f'Roll_Std_{w}_{pos}', f'Momentum_{w}_{pos}'])  

        hash_array = pd.util.hash_pandas_object(df_raw[['Result_3D', 'Result_2D']], index=False).values  
        base_hash = hashlib.md5(hash_array).hexdigest()  
        self.data_hash = f"{base_hash}_{self.test_size}_turbo_vmax"  

        self.pos_sys, self.freq_sys = PositionalEquation(), FrequencyEngine()  
        self.cond_sys, self.markov_sys = ConditionalSystem(), MarkovChainSystem()  
        self.ptn_sys = PatternBacktestSystem()  
          
        self.ai_sys = AISystem(self.lottery_id, n)  
        self.base_weights = {'AI': 0.40, 'Freq': 0.15, 'Markov': 0.15, 'Cal': 0.10, 'BT': 0.10, 'Eq': 0.10}  

    def _process_single_position(self, pos, df_hist, X_all, next_x, next_date):  
        bt_size = self.test_size  
        cache_key = os.path.join(CACHE_DIR, f"bt_turbo_{self.lottery_id}_{pos}_{self.data_hash}.joblib")  
        n = len(df_hist)  

        train_len = len(X_all) - bt_size if bt_size > 0 else len(X_all)  
        X_train_full = X_all.iloc[:train_len]  
        y_train_full = df_hist[pos].iloc[:train_len]  

        valid_features = [f for f in X_all.columns if f in X_train_full.columns]  
          
        mi_scores = mutual_info_classif(X_train_full, y_train_full, random_state=42)  
        mi_series = pd.Series(mi_scores, index=valid_features)  
          
        if n >= 700: mi_thresh = 0.010  
        elif n >= 400: mi_thresh = 0.008  
        elif n >= 200: mi_thresh = 0.005  
        else: mi_thresh = 0.002  
          
        target_feats = min(len(valid_features), max(30, int(n * 0.15)))  
        pre_selected = mi_series[mi_series > mi_thresh].sort_values(ascending=False).head(target_feats * 2).index  
        if len(pre_selected) < 10: pre_selected = valid_features[:target_feats]  
          
        corr_thresh = max(0.75, 0.95 - (n / 5000.0))  
          
        corr_matrix = X_all[pre_selected].corr().abs()  
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  
        to_drop = set()  
        for col in upper_tri.columns:  
            high_corr = upper_tri.index[upper_tri[col] > corr_thresh].tolist()  
            for r in high_corr:  
                if mi_series[col] > mi_series[r]: to_drop.add(r)  
                else: to_drop.add(col)  
          
        selected_feats = [f for f in pre_selected if f not in to_drop][:target_feats]  
        self.final_feat_count = len(selected_feats)  
          
        X_all_fs = X_all[selected_feats]  
        next_x_fs = next_x[selected_feats]  

        if n >= 700: decay_factor = 2.5  
        elif n >= 400: decay_factor = 2.0  
        elif n >= 200: decay_factor = 1.6  
        else: decay_factor = 1.2  
          
        full_sample_weights = np.exp(np.linspace(-decay_factor, 0, len(X_all_fs)))  

        if os.path.exists(cache_key):  
            norm_weights, bt_msg = joblib.load(cache_key)  
        elif len(df_hist) < bt_size + 30 or bt_size <= 0:  
            norm_weights, bt_msg = self.base_weights, "(ข้อมูลน้อย ข้าม Backtest)"  
        else:  
            scores = {k: 0.0 for k in self.base_weights.keys()}  
            total_steps_weight = 0.0  
              
            lite_trees = max(20, self.ai_sys.trees // 3)  
            lite_estimators = [  
                ('hgb', HistGradientBoostingClassifier(max_iter=lite_trees, max_leaf_nodes=15, min_samples_leaf=3, random_state=42)),  
                ('xgb', XGBClassifier(n_estimators=lite_trees, max_depth=max(1, self.ai_sys.depth-1), learning_rate=0.05, subsample=0.8, tree_method="hist", verbosity=0, random_state=42, n_jobs=-1)),  
                ('et', ExtraTreesClassifier(n_estimators=lite_trees, max_depth=self.ai_sys.depth, class_weight='balanced', random_state=42, n_jobs=-1))  
            ]  
            if HAS_LGBM: lite_estimators.insert(0, ('lgbm', LGBMClassifier(n_estimators=lite_trees, max_depth=self.ai_sys.depth, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)))  
              
            if n >= 500:  
                bt_ai_base = StackingClassifier(estimators=lite_estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=50), cv=2, n_jobs=-1)  
            else:  
                bt_ai_base = VotingClassifier(estimators=lite_estimators, voting='soft', n_jobs=-1)  
                  
            calib_method = 'isotonic' if n >= 200 else 'sigmoid'  
            bt_ai_model = CalibratedClassifierCV(bt_ai_base, method=calib_method, cv=2)  

            for i in range(bt_size):  
                curr_train_len = len(X_all_fs) - bt_size + i  
                X_train_step = X_all_fs.iloc[:curr_train_len]  
                y_train_step = df_hist[pos].iloc[:curr_train_len]  
                step_weights = full_sample_weights[:curr_train_len]  
                X_test_step = X_all_fs.iloc[[curr_train_len]]  
                actual_val = df_hist[pos].iloc[curr_train_len]  

                try: bt_ai_model.fit(X_train_step, y_train_step, sample_weight=step_weights)  
                except: bt_ai_model.fit(X_train_step, y_train_step)  

                probs_ai = bt_ai_model.predict_proba(X_test_step)[0]  
                ai_res = np.zeros(10)  
                for idx, c in enumerate(bt_ai_model.classes_): ai_res[int(c)] = probs_ai[idx]  

                curr_df = df_hist.iloc[:curr_train_len]  
                target_date = df_hist.iloc[curr_train_len]['Date']  

                sys_probs = {  
                    'AI': ai_res,  
                    'Freq': self.freq_sys.analyze(curr_df, pos),  
                    'Cal': self.cond_sys.analyze(curr_df, pos, target_date),  
                    'Markov': self.markov_sys.analyze(curr_df, pos),  
                    'BT': self.ptn_sys.analyze(curr_df, pos),  
                    'Eq': self.pos_sys.analyze(curr_df)  
                }  
                  
                step_weight = np.exp((i - bt_size + 1) * 0.15)   
                total_steps_weight += step_weight  

                for sys_name, p in sys_probs.items():  
                    ranked = np.argsort(p)[::-1]  
                    top1 = 1 if actual_val == ranked[0] else 0  
                    top3 = 1 if actual_val in ranked[:3] else 0  
                    top5 = 1 if actual_val in ranked[:5] else 0  
                      
                    ll = -np.log(p[actual_val] + 1e-9)  
                    brier = np.sum((p - np.eye(10)[actual_val])**2)  
                      
                    metric_score = (top1*0.4 + top3*0.3 + top5*0.1) + (1.0 / (ll + 1.0))*0.1 + (1.0 - (brier/2.0))*0.1  
                    scores[sys_name] += (metric_score * step_weight)  

            total_score = sum(scores.values())  
            norm_weights = {k: v/total_score for k, v in scores.items()}  
            msg = f"(Turbo-BT: AI {norm_weights['AI']*100:.1f}% | Bayesian Markov {norm_weights['Markov']*100:.1f}%)"  

            joblib.dump((norm_weights, msg), cache_key)  
            bt_msg = msg  

        p_ai = self.ai_sys.analyze(X_all_fs, df_hist[pos], next_x_fs, pos, self.data_hash, sample_weight=full_sample_weights)  
        p_fq = self.freq_sys.analyze(df_hist, pos)  
        p_cal = self.cond_sys.analyze(df_hist, pos, next_date)  
        p_mk = self.markov_sys.analyze(df_hist, pos)  
        p_bt = self.ptn_sys.analyze(df_hist, pos)  
        p_eq = self.pos_sys.analyze(df_hist)  

        ent_weights = {  
            'AI': get_entropy_weight(p_ai),  
            'Freq': get_entropy_weight(p_fq),  
            'Cal': get_entropy_weight(p_cal),  
            'Markov': get_entropy_weight(p_mk),  
            'BT': get_entropy_weight(p_bt),  
            'Eq': get_entropy_weight(p_eq)  
        }  

        W = norm_weights.copy()  
        W = {k: W[k] * ent_weights[k] for k in W}  
        total_w = sum(W.values())  
        W = {k: v/total_w for k, v in W.items()}  

        final_score = (W['AI']*p_ai + W['Freq']*p_fq + W['Cal']*p_cal + W['Markov']*p_mk + W['BT']*p_bt + W['Eq']*p_eq)  
        final_score = final_score / final_score.sum()  

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]  

        calib_msg = f" [Turbo Applied]"  

        return pos, {  
            'AI': get_top5(p_ai),  
            'Calendar': get_top5(p_cal),  
            'Markov': get_top5(p_mk),  
            'Final': get_top5(final_score),  
            'Probs_For_Graph': final_score,  
            'BT_Msg': bt_msg + calib_msg,  
            'Feat_Count': getattr(self, 'final_feat_count', 0)  
        }  

    def predict_all(self, st_progress_bar):  
        last_date = self.df_raw['Date'].iloc[-1]  
        if self.target_dow is not None:  
            days_ahead = self.target_dow - last_date.dayofweek  
            if days_ahead <= 0: days_ahead += 7  
            next_date = last_date + timedelta(days=days_ahead)  
        else:  
            next_date = last_date + timedelta(days=7 if len(self.df_raw) <= 1 else (last_date - self.df_raw['Date'].iloc[-2]).days)  

        dummy = pd.DataFrame([{'Date': next_date, 'Result_3D': '000', 'Result_2D': '00'}])  
        df_ext = pd.concat([self.df_raw, dummy], ignore_index=True)  

        df_ext = build_features(df_ext, self.lags, self.rolls)  
        next_x = df_ext.iloc[[-1]][self.features]  
        df_hist = df_ext.iloc[:-1]  
        X_all = df_hist[self.features]  

        results = []  
        positions = ['H', 'T', 'O', 'T2', 'O2']  
          
        for i, pos in enumerate(positions):  
            progress_percent = int(((i + 1) / 5) * 100)
            st_progress_bar.progress(progress_percent, text=f'กำลังประมวลผลโมเดล {self.ai_sys.model_name}: ตำแหน่ง {pos}...')
            res = self._process_single_position(pos, df_hist, X_all, next_x, next_date)  
            results.append(res)  
              
        predictions = {pos: data for pos, data in results}  
        return predictions, next_date

# ==========================================
# 5. Dashboard (UI ของ Streamlit)
# ==========================================
st.title("🎯 ระบบวิเคราะห์เลขเด่น Ultimate Ensemble V.Max")
st.markdown("*(Turbo Quantum Edition)*")

col1, col2 = st.columns(2)
with col1:
    selected_lotto = st.selectbox("🎯 เลือกหวย:", list(LOTTERY_SOURCES.keys()))
with col2:
    day_options = {
        'อัตโนมัติ (คำนวณจากงวดล่าสุด)': None, 'วันจันทร์': 0, 'วันอังคาร': 1, 
        'วันพุธ': 2, 'วันพฤหัสบดี': 3, 'วันศุกร์': 4, 'วันเสาร์': 5, 'วันอาทิตย์': 6
    }
    selected_day_name = st.selectbox("📅 ออกวัน:", list(day_options.keys()))
    target_dow = day_options[selected_day_name]

if st.button("🚀 วิเคราะห์เลขเด่น (Turbo Speed)", type="primary", use_container_width=True):
    url = LOTTERY_SOURCES[selected_lotto]
    
    try:
        # ดึงข้อมูล
        with st.spinner("กำลังดึงและเตรียมข้อมูลจากแหล่งอ้างอิง..."):
            df_raw = fetch_and_clean_data(url)
            engine = EnsembleEngine(df_raw, selected_lotto, target_dow=target_dow)

        # พื้นที่แสดงการโหลด
        st.write("---")
        progress_bar = st.progress(0, text="เตรียมเริ่มการวิเคราะห์โมเดล AI...")
        
        # รันการทำนาย
        preds, next_date = engine.predict_all(progress_bar)
        
        st.success("✨ วิเคราะห์เสร็จสิ้นสมบูรณ์!")
        
        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]  
        labels = {'H': 'หลักร้อย (บน)', 'T': 'หลักสิบ (บน)', 'O': 'หลักหน่วย (บน)', 'T2': 'หลักสิบ (ล่าง)', 'O2': 'หลักหน่วย (ล่าง)'}  

        # สรุปฟันธง
        probs_top = (preds['H']['Probs_For_Graph'] + preds['T']['Probs_For_Graph'] + preds['O']['Probs_For_Graph']) / 3  
        probs_bot = (preds['T2']['Probs_For_Graph'] + preds['O2']['Probs_For_Graph']) / 2  

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]  
        top5_top = get_top5(probs_top)  
        top5_bot = get_top5(probs_bot)  

        st.subheader("🔥 สรุปฟันธง เลขเด่นมาแรง (Quantum Computed Probabilities)")
        top_str = " , ".join([str(x[0]) for x in top5_top])
        bot_str = " , ".join([str(x[0]) for x in top5_bot])
        st.info(f"**🚀 เด่นบนรวม (ร้อย-สิบ-หน่วย) : {top_str}**")
        st.info(f"**⬇️ เด่นล่างรวม (สิบ-หน่วย) : {bot_str}**")
        
        st.write(f"🔮 ผลการวิเคราะห์ระดับลึก ประจำวัน{dow_names[next_date.dayofweek]}ที่ {next_date.strftime('%d-%m-%Y')} (ใช้ข้อมูล {len(df_raw)} งวด)")

        # รายละเอียดแต่ละหลัก
        for pos in ['H', 'T', 'O', 'T2', 'O2']:  
            with st.expander(f"📍 เจาะลึกตำแหน่ง: {labels[pos]}"):
                st.caption(f"คัดเฉพาะฟีเจอร์เด่นสุด {preds[pos]['Feat_Count']} ตัว | {preds[pos]['BT_Msg']}")
                
                nums_ai = ", ".join([str(num) for num, prob in preds[pos]['AI']])  
                nums_day = ", ".join([str(num) for num, prob in preds[pos]['Calendar']])  
                nums_mk = ", ".join([str(num) for num, prob in preds[pos]['Markov']])  
                nums_final = ", ".join([str(num) for num, prob in preds[pos]['Final']])  

                st.markdown(f"- 🧠 **เลขเด่น Quantum AI:** {nums_ai}")  
                st.markdown(f"- 🔗 **เลขเด่น มาร์คอฟแบบเบย์:** {nums_mk}")  
                st.markdown(f"- 📅 **เลขเด่น กำลังวัน:** {nums_day}")  
                st.markdown(f"- 🌟 **เด่นสรุปรวม 5 ตัว:** {nums_final}")  

        # กราฟ Matplotlib
        st.subheader("📊 กราฟโอกาสความน่าจะเป็น (Probabilities)")
        fig = plt.figure(figsize=(12, 8))  
        fig.suptitle(f'Quantum Precision Probabilities - {selected_lotto}', fontsize=14, fontweight='bold')  
        colors_list = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  
        
        for idx, pos in enumerate(['H', 'T', 'O', 'T2', 'O2']):  
            ax = plt.subplot(2, 3, idx + 1)  
            top_5_items = preds[pos]['Final']  
            ax.bar([str(x[0]) for x in top_5_items], [x[1]*100 for x in top_5_items], color=colors_list)  
            ax.set_title(labels[pos])  
            ax.set_ylabel('โอกาส (%)')  
            
        plt.tight_layout()  
        st.pyplot(fig)  

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดของระบบ: {str(e)}")
