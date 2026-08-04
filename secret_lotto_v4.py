import requests
import warnings
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import Memory
import copy
warnings.filterwarnings('ignore')

memory = Memory(location='/tmp/lotto_cache_v4', verbose=0)
global_model_cache = {}
global_backtest_cache = {}

class LotteryScraper:
    def __init__(self):
        self.urls = {
            'หวยไทย': 'https://suksan18190.blogspot.com/2026/07/blog-post_07.html',
            'หวยธกส': 'https://suksan18190.blogspot.com/2026/07/blog-post_12.html',
            'หวยออมสิน': 'https://suksan18190.blogspot.com/2026/07/blog-post_525.html',
            'หวยลาว': 'https://suksan18190.blogspot.com/2026/07/blog-post.html',
            'หวยฮานอย': 'https://suksan18190.blogspot.com/2026/07/blog-post_08.html',
            'หวยมาเลย์': 'https://suksan18190.blogspot.com/2026/07/blog-post_10.html',
            'หวยหุ้นไทยเย็น': 'https://suksan18190.blogspot.com/2026/07/blog-post_11.html',
            'หวยหุ้นนิเคอิบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_412.html',
            'หวยหุ้นฮั่งเส็งบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_229.html',
            'หวยหุ้นจีนบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_162.html'
        }

    @staticmethod
    @memory.cache
    def _fetch_url_content(url):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            return response.content
        except:
            return None

    def fetch_data(self, lotto_name):
        if lotto_name not in self.urls: return None
        content = self._fetch_url_content(self.urls[lotto_name])
        if not content: return None
        try:
            soup = BeautifulSoup(content, 'html.parser')
            post_body = soup.find('div', class_=re.compile(r'post-body|entry-content'))
            if not post_body: return None
            text_content = post_body.get_text()
            pattern = r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})"
            matches = re.findall(pattern, text_content)
            data = []
            for date_str, prize1, bot2 in matches:
                p1_str = str(prize1).zfill(3)
                bot2_str = str(bot2).zfill(2)
                data.append({
                    'date': date_str, 'draw_num': prize1,
                    'hundred': int(p1_str[-3]), 'ten': int(p1_str[-2]), 'unit': int(p1_str[-1]),
                    'bot_ten': int(bot2_str[0]), 'bot_unit': int(bot2_str[1])
                })
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
        except Exception:
            return None

@memory.cache
def build_features_adaptive(df, col, lags, rolls):
    df_feat = df.copy()
    n = len(df)
    df_feat['prev_val'] = df_feat[col].shift(1)
    df_feat['mirror'] = (df_feat['prev_val'] + 5) % 10
    df_feat['is_even'] = (df_feat['prev_val'] % 2 == 0).astype(int)
    df_feat['is_high'] = (df_feat['prev_val'] >= 5).astype(int)
    df_feat['mod3'] = (df_feat['prev_val'] % 3).fillna(0).astype(int)
    df_feat['weekday'] = df_feat['date'].dt.weekday
    for lag in lags: df_feat[f'lag_{lag}'] = df_feat[col].shift(lag)
    if 'lag_1' in df_feat.columns and 'lag_2' in df_feat.columns:
        df_feat['repeat_2'] = (df_feat['lag_1'] == df_feat['lag_2']).astype(int)
        if 'lag_3' in df_feat.columns:
            df_feat['repeat_3'] = ((df_feat['lag_1'] == df_feat['lag_2']) & (df_feat['lag_2'] == df_feat['lag_3'])).astype(int)
    for w in rolls:
        df_feat[f'rolling_mean_{w}'] = df_feat[col].shift(1).rolling(w).mean()
        df_feat[f'rolling_std_{w}'] = df_feat[col].shift(1).rolling(w).std()
    history = df_feat[col].values
    hc_windows = list(rolls)
    if n >= 500 and 50 not in hc_windows: hc_windows.append(50)
    stats_cols = {f'{typ}{w}_{d}': np.zeros(n) for typ in ['hot', 'cold'] for w in hc_windows for d in range(10) if not (typ=='cold' and w>=50)}
    skip_cols = {f'skip_{d}': np.full(n, 100) for d in range(10)}
    last_seen = {d: -1 for d in range(10)}
    for i in range(1, n):
        for d in range(10):
            skip = i - last_seen[d] if last_seen[d] != -1 else 100
            skip_cols[f'skip_{d}'][i] = skip
        for w in hc_windows:
            window_slice = history[max(0, i-w):i]
            for d in range(10):
                hot_count = np.sum(window_slice == d)
                stats_cols[f'hot{w}_{d}'][i] = hot_count
                if w < 50: stats_cols[f'cold{w}_{d}'][i] = len(window_slice) - hot_count
        last_seen[history[i]] = i
    for key, val in skip_cols.items(): df_feat[key] = val
    for key, val in stats_cols.items(): df_feat[key] = val
    return df_feat.fillna(-1)

class OptimizedEliminationSystemV4:
    def __init__(self, df, target_col, lotto_name):
        self.df = df.copy()
        self.target_col = target_col
        self.lotto_name = lotto_name
        n = len(self.df)
        if n >= 700:
            self.mode_name = "Mode 4 (700+ งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop = 100, 20, 10
            self.lags, self.rolls = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20]
            self.ai_weights = (1.0, 1.0, 1.0, 1.0)
        elif n >= 400:
            self.mode_name = "Mode 3 (400-699 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop = 100, 20, 10
            self.lags, self.rolls = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20]
            self.ai_weights = (1.0, 0.9, 0.8, 1.0)
        elif n >= 200:
            self.mode_name = "Mode 2 (200-399 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop = 80, 15, 8
            self.lags, self.rolls = [1, 2, 3, 5, 8], [3, 5, 10, 20]
            self.ai_weights = (1.0, 0.8, 0.6, 0.5)
        else:
            self.mode_name = "Mode 1 (100-199 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop = 60, 10, 5
            self.lags, self.rolls = [1, 2, 3, 5], [3, 5, 10]
            self.ai_weights = (1.0, 0.8, 0.5, 0.15)
        if n < 100: self.test_size = min(5, max(0, n - 30))
        self.models = {
            'rf': RandomForestClassifier(n_estimators=self.trees, random_state=42, max_depth=5, n_jobs=1),
            'et': ExtraTreesClassifier(n_estimators=self.trees, random_state=42, max_depth=5, n_jobs=1),
            'hgb': HistGradientBoostingClassifier(random_state=42, max_iter=50),
            'xgb': XGBClassifier(n_estimators=50, max_depth=3, tree_method="hist", verbosity=0, random_state=42, n_jobs=1)
        }
        self.model_weights_dict = {'rf': self.ai_weights[0], 'et': self.ai_weights[1], 'hgb': self.ai_weights[2], 'xgb': self.ai_weights[3]}
        self.df_feat = build_features_adaptive(self.df, self.target_col, tuple(self.lags), tuple(self.rolls))

    def precompute_markov_adaptive(self, df_hist):
        seq = df_hist[self.target_col].values
        n = len(seq)
        if n < 5: return np.ones(10)/10.0
        L1, L2, L3 = seq[-1], seq[-2], seq[-3] if n >= 6 else -1
        mc1, tot1 = np.zeros(10), 0
        for i in range(1, len(seq)-1):
            if seq[i] == L1:
                mc1[seq[i+1]] += 1
                tot1 += 1
        prob_o1 = mc1 / tot1 if tot1 > 0 else np.ones(10)/10.0
        if n < 200: return prob_o1
        mc2, tot2 = np.zeros(10), 0
        for i in range(2, len(seq)-1):
            if seq[i-1] == L2 and seq[i] == L1:
                mc2[seq[i+1]] += 1
                tot2 += 1
        prob_o2 = mc2 / tot2 if tot2 > 0 else prob_o1
        if n < 500: return (0.6 * prob_o2) + (0.4 * prob_o1)
        mc3, tot3 = np.zeros(10), 0
        for i in range(3, len(seq)-1):
            if seq[i-2] == L3 and seq[i-1] == L2 and seq[i] == L1:
                mc3[seq[i+1]] += 1
                tot3 += 1
        prob_o3 = mc3 / tot3 if tot3 > 0 else prob_o2
        return (0.5 * prob_o3) + (0.3 * prob_o2) + (0.2 * prob_o1)

    def calculate_freq_skip(self, df_hist, digit):
        col = self.target_col
        freq = (df_hist[col] == digit).sum() / max(len(df_hist), 1)
        matches = df_hist[df_hist[col] == digit]
        skip = len(df_hist) - matches.index[-1] - 1 if len(matches) > 0 else 100
        norm_freq = min(freq * 10, 1.0)
        norm_skip = max(1.0 - (skip / 30), 0.0)
        return (0.5 * norm_freq) + (0.5 * norm_skip)

    def run_backtest(self, X_train, y_train, df_hist_cut, test_size):
        cache_key = f"bt_{self.lotto_name}_{self.target_col}_{len(df_hist_cut)}_{test_size}_v4"
        global global_backtest_cache
        if cache_key in global_backtest_cache: return global_backtest_cache[cache_key]
        bt_train_X = X_train.iloc[:-test_size]
        bt_train_y = y_train.iloc[:-test_size]
        bt_test_X = X_train.iloc[-test_size:]
        bt_test_y = y_train.iloc[-test_size:].values
        
        le = LabelEncoder()
        bt_train_y_encoded = le.fit_transform(bt_train_y)
        
        ai_fails, stat_fails, day_fails = 0, 0, 0
        trained_models = {}
        for name, model in self.models.items():
            m = copy.deepcopy(model)
            m.fit(bt_train_X, bt_train_y_encoded)
            trained_models[name] = m
            
        ai_preds = np.zeros((test_size, 10))
        total_ai_weight = sum(self.model_weights_dict.values())
        for name, m in trained_models.items():
            preds = m.predict_proba(bt_test_X)
            full_preds = np.zeros((test_size, 10))
            for idx, c in enumerate(le.classes_):
                full_preds[:, int(c)] = preds[:, idx]
            ai_preds += full_preds * self.model_weights_dict[name]
        ai_preds /= total_ai_weight
        
        for i in range(test_size):
            if bt_test_y[i] in np.argsort(ai_preds[i])[:5]: ai_fails += 1
            curr_hist = df_hist_cut.iloc[:-(test_size - i)]
            mk = self.precompute_markov_adaptive(curr_hist)
            st_probs = np.zeros(10)
            for d in range(10): st_probs[d] = (0.5 * self.calculate_freq_skip(curr_hist, d)) + (0.5 * mk[d])
            if bt_test_y[i] in np.argsort(st_probs)[:5]: stat_fails += 1
            target_dow = df_hist_cut.iloc[-(test_size - i)]['date'].weekday()
            day_df = curr_hist[curr_hist['date'].dt.weekday == target_dow]
            day_probs = np.zeros(10)
            if len(day_df) > 0:
                counts = day_df[self.target_col].value_counts(normalize=True)
                for d in range(10): day_probs[d] = counts.get(d, 0.0)
            else: day_probs = np.ones(10)/10.0
            if bt_test_y[i] in np.argsort(day_probs)[:5]: day_fails += 1
            
        result = (ai_fails, stat_fails, day_fails)
        global_backtest_cache[cache_key] = result
        return result

    def analyze(self, target_dow):
        global global_model_cache
        df_work = self.df_feat
        df_hist = self.df
        data_size = len(df_hist)
        if data_size < 30: return None
        exclude = ['date', 'draw_num', 'hundred', 'ten', 'unit', 'bot_ten', 'bot_unit', self.target_col]
        feature_cols = [c for c in df_work.columns if c not in exclude]
        X, y = df_work[feature_cols], df_work[self.target_col]
        train_X, test_X = X.iloc[:-1], X.iloc[-1:]
        train_y = y.iloc[:-1]
        df_hist_cut = df_hist.iloc[:-1]

        if data_size < 200: w_ai, w_stat, w_day = 0.30, 0.50, 0.20
        elif data_size < 500: w_ai, w_stat, w_day = 0.40, 0.40, 0.20
        else: w_ai, w_stat, w_day = 0.50, 0.35, 0.15

        backtest_msg = ""
        if self.test_size > 0 and data_size > self.test_size + 30:
            ai_f, st_f, day_f = self.run_backtest(train_X, train_y, df_hist_cut, self.test_size)
            w_ai_adj = w_ai * (max(0.1, 1.0 - (ai_f / self.test_size))**2)
            w_st_adj = w_stat * (max(0.1, 1.0 - (st_f / self.test_size))**2)
            w_day_adj = w_day * (max(0.1, 1.0 - (day_f / self.test_size))**2)
            total_adj = w_ai_adj + w_st_adj + w_day_adj
            w_ai, w_stat, w_day = w_ai_adj/total_adj, w_st_adj/total_adj, w_day_adj/total_adj
            backtest_msg = f" (BT-Score: AI {int((1-ai_f/self.test_size)*100)}% | Stat {int((1-st_f/self.test_size)*100)}% | Day {int((1-day_f/self.test_size)*100)}%)"

        last_date = df_hist['date'].iloc[-1].strftime('%Y-%m-%d')
        cache_key = f"{self.lotto_name}_{self.target_col}_{last_date}_v4_encoded"
        ai_probs = np.zeros(10)
        
        le_main = LabelEncoder()
        train_y_encoded = le_main.fit_transform(train_y)

        if cache_key in global_model_cache:
            trained_models, cached_le = global_model_cache[cache_key]
            le_main = cached_le
        else:
            trained_models = {}
            for name, model in self.models.items():
                model.fit(train_X, train_y_encoded)
                trained_models[name] = model
            global_model_cache[cache_key] = (trained_models, le_main)

        total_ai_weight = sum(self.model_weights_dict.values())
        for name, model in trained_models.items():
            preds = model.predict_proba(test_X)[0]
            model_probs = np.zeros(10)
            for idx, c in enumerate(le_main.classes_): model_probs[int(c)] = preds[idx]
            ai_probs += model_probs * self.model_weights_dict[name]
        ai_probs /= total_ai_weight
        ai_probs /= (ai_probs.sum() + 1e-9)

        stat_probs = np.zeros(10)
        markov_scores = self.precompute_markov_adaptive(df_hist_cut)
        for d in range(10): stat_probs[d] = (0.5 * self.calculate_freq_skip(df_hist_cut, d)) + (0.5 * markov_scores[d])
        stat_probs /= (stat_probs.sum() + 1e-9)

        day_probs = np.zeros(10)
        day_df = df_hist_cut[df_hist_cut['date'].dt.weekday == target_dow]
        if len(day_df) > 0:
            counts = day_df[self.target_col].value_counts(normalize=True)
            for d in range(10): day_probs[d] = counts.get(d, 0.0)
        else: day_probs = np.ones(10)/10.0

        final_probs = (w_ai * ai_probs) + (w_stat * stat_probs) + (w_day * day_probs)
        final_probs /= (final_probs.sum() + 1e-9)

        return {'ai': ai_probs, 'stat': stat_probs, 'day': day_probs, 'final': final_probs, 'w_ai': w_ai, 'w_stat': w_stat, 'w_day': w_day, 'bt_msg': backtest_msg}
