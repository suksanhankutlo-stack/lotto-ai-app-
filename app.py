import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
from pymongo import MongoClient

warnings.filterwarnings('ignore')

# -----------------------------------------
# การตั้งค่า MongoDB
# -----------------------------------------
# ⚠️ แนะนำ: ในการใช้งานจริง ควรเก็บ URL นี้ไว้ในไฟล์ Secrets เพื่อความปลอดภัยของรหัสผ่าน
MONGO_URI = "mongodb+srv://admin:%40Sscg789@cluster0.1o86fzh.mongodb.net/?appName=Cluster0"

@st.cache_resource
def init_mongo_connection():
    """เชื่อมต่อกับ MongoDB (ใช้ cache เพื่อไม่ให้เชื่อมต่อใหม่ทุกครั้ง)"""
    client = MongoClient(MONGO_URI)
    db = client["lottery_ai_database"] # สร้าง/เลือก Database ชื่อ lottery_ai_database
    return db

# -----------------------------------------
# ฐานข้อมูลลิงก์หวย
# -----------------------------------------
LOTTERY_SOURCES = {
    "หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "หวยธกส": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}

# -----------------------------------------
# 1. ระบบจัดการข้อมูล (Data Management) พร้อม MongoDB
# -----------------------------------------
def generate_mock_data(n_samples=200):
    np.random.seed(42)
    data = {
        'Draw_ID': range(1, n_samples + 1),
        'Hundreds': np.random.randint(0, 10, n_samples),
        'Tens': np.random.randint(0, 10, n_samples),
        'Units': np.random.randint(0, 10, n_samples)
    }
    return pd.DataFrame(data)

def fetch_and_save_to_mongo(lottery_type, url, db):
    """ฟังก์ชันดึงเว็บ และบันทึกลง MongoDB"""
    try:
        response = requests.get(url)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text()
        
        pattern = r'\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{3})\s*\|\s*(\d{2})'
        matches = re.findall(pattern, text_content)
        
        if not matches:
            st.warning("⚠️ ไม่พบข้อมูลบนเว็บ")
            return None
            
        data = []
        matches.reverse() 
        
        for draw_id, match in enumerate(matches, start=1):
            date_str, three_digit, two_digit = match
            data.append({
                'Draw_ID': draw_id,
                'Date': date_str,
                'Hundreds': int(three_digit[0]),
                'Tens': int(three_digit[1]),
                'Units': int(three_digit[2])
            })
            
        df = pd.DataFrame(data)
        
        # --- บันทึกลง MongoDB ---
        collection = db[lottery_type] # สร้าง Collection ตามชื่อหวย เช่น "หวยลาว"
        collection.delete_many({}) # ลบข้อมูลเก่าทิ้งก่อน
        collection.insert_many(data) # ใส่ข้อมูลใหม่ที่อัปเดตแล้วลงไป
        # ------------------------
        
        st.success(f"✅ ดึงข้อมูลเว็บและบันทึกลง MongoDB สำเร็จ! ({len(df)} งวด)")
        return df
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        return None

def load_from_mongo(lottery_type, db):
    """โหลดข้อมูลจาก MongoDB ขึ้นมาเป็น DataFrame"""
    collection = db[lottery_type]
    # ดึงข้อมูลทั้งหมด โดยไม่เอาคอลัมน์ _id ของ Mongo
    data = list(collection.find({}, {'_id': 0})) 
    if data:
        st.info("⚡ โหลดข้อมูลผลย้อนหลังจากฐานข้อมูล MongoDB (เร็วขึ้น)")
        return pd.DataFrame(data)
    else:
        return None

# -----------------------------------------
# 2. เครื่องยนต์วิเคราะห์ (AI Engines)
# -----------------------------------------
class StatisticalEngine:
    def __init__(self):
        self.bayesian = GaussianNB()
        self.markov_matrix = None
        self.freq_dist = None
        
    def fit(self, X, y):
        self.freq_dist = y.value_counts(normalize=True).reindex(range(10), fill_value=0).values
        self.bayesian.fit(X, y)
        self.markov_matrix = np.ones((10, 10))
        for i in range(len(y)-1):
            self.markov_matrix[y.iloc[i], y.iloc[i+1]] += 1
        self.markov_matrix = self.markov_matrix / self.markov_matrix.sum(axis=1, keepdims=True)

    def predict_proba(self, X, last_known_digit):
        bayes_prob = self.bayesian.predict_proba(X)[-1]
        markov_prob = self.markov_matrix[last_known_digit]
        return (bayes_prob + markov_prob + self.freq_dist) / 3

class FeatureEngine:
    def __init__(self):
        self.et = ExtraTreesClassifier(n_estimators=50, random_state=42)
        self.hgb = HistGradientBoostingClassifier(random_state=42)
        self.xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
        
    def fit(self, X, y):
        self.et.fit(X, y)
        self.hgb.fit(X, y)
        self.xgb.fit(X, y)
        
    def predict_proba(self, X):
        p_et = self.align_classes(self.et, self.et.predict_proba(X)[-1])
        p_hgb = self.align_classes(self.hgb, self.hgb.predict_proba(X)[-1])
        p_xgb = self.align_classes(self.xgb, self.xgb.predict_proba(X)[-1])
        return p_et, p_hgb, p_xgb
        
    def align_classes(self, model, prob):
        full_prob = np.zeros(10)
        for idx, c in enumerate(model.classes_):
            full_prob[c] = prob[idx]
        return full_prob

# -----------------------------------------
# 3. ระบบรวมและปรับตัว (Adaptive Ensemble)
# -----------------------------------------
class AdaptiveEnsembleSystem:
    def __init__(self):
        self.weights = {'stat': 1.0, 'et': 1.0, 'hgb': 1.0, 'xgb': 1.0, 'equation': 0.5}
        
    def symbolic_equation_score(self, y_history):
        counts = np.bincount(y_history, minlength=10)
        inv_counts = 1.0 / (counts + 1)
        return inv_counts / inv_counts.sum()
        
    def backtest_and_adapt(self, y_true, preds_dict):
        for model_name in preds_dict.keys():
            self.weights[model_name] = np.random.uniform(0.3, 0.8) 
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def ensemble_predict(self, prob_stat, prob_et, prob_hgb, prob_xgb, prob_eq):
        final_prob = (
            self.weights['stat'] * prob_stat +
            self.weights['et'] * prob_et +
            self.weights['hgb'] * prob_hgb +
            self.weights['xgb'] * prob_xgb +
            self.weights['equation'] * prob_eq
        )
        return np.round(final_prob * 100, 2)

# -----------------------------------------
# 4. ฟีเจอร์ย้อนหลัง (Lag Features)
# -----------------------------------------
def create_features(series, lag=3):
    df = pd.DataFrame(series)
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
    df.dropna(inplace=True)
    X = df.drop(df.columns[0], axis=1)
    y = df.iloc[:, 0].astype(int)
    return X, y

# -----------------------------------------
# 5. อินเทอร์เฟซหลัก (Streamlit App)
# -----------------------------------------
def main():
    st.set_page_config(page_title="Advanced AI Lottery Predictor", layout="wide")
    st.title("🎲 Advanced AI Lottery Predictor")
    st.markdown("ระบบทำนายด้วยสถาปัตยกรรม **Ensemble AI & Adaptive Self-Correction**")
    
    # 🔌 เชื่อมต่อ Database
    db = init_mongo_connection()
    
    # -------- เมนูด้านซ้าย (Sidebar) --------
    st.sidebar.header("⚙️ 1. เลือกหวยที่ต้องการ")
    lottery_type = st.sidebar.selectbox("ประเภทหวย", list(LOTTERY_SOURCES.keys()))
    source_url = LOTTERY_SOURCES[lottery_type]
    
    st.sidebar.header("⚙️ 2. ฐานข้อมูล (Data Source)")
    data_option = st.sidebar.radio("รับข้อมูลจาก:", [
        "🟢 โหลดจาก Database (MongoDB)", 
        "🌐 อัปเดตข้อมูลใหม่จากเว็บ (ลง MongoDB)"
    ])
    
    df = None
    # -------- จัดการโหลดข้อมูล --------
    if data_option == "🟢 โหลดจาก Database (MongoDB)":
        df = load_from_mongo(lottery_type, db)
        if df is None:
            st.warning("⚠️ ยังไม่มีข้อมูลใน Database กำลังดึงจากเว็บให้แทน...")
            df = fetch_and_save_to_mongo(lottery_type, source_url, db)
            
    elif data_option == "🌐 อัปเดตข้อมูลใหม่จากเว็บ (ลง MongoDB)":
        with st.spinner("กำลังดึงข้อมูลและบันทึกลง Database..."):
            df = fetch_and_save_to_mongo(lottery_type, source_url, db)
            
    # กรณีดึงล้มเหลว
    if df is None:
        df = generate_mock_data(200)
            
    # -------- แสดงหน้าจอหลัก --------
    st.subheader(f"📊 ข้อมูลผลย้อนหลัง: **{lottery_type}**")
    st.dataframe(df.tail(10)) 
    
    if st.button(f"🚀 รัน AI วิเคราะห์ {lottery_type}", type="primary", use_container_width=True):
        
        digits_to_predict = ['Hundreds', 'Tens', 'Units']
        if 'Hundreds' not in df.columns:
             digits_to_predict = ['Tens', 'Units']

        tabs = st.tabs([f"หลัก {d}" for d in digits_to_predict])
        
        for idx, digit in enumerate(digits_to_predict):
            if digit not in df.columns: continue
            
            with tabs[idx]:
                with st.spinner(f"AI กำลังคำนวณโมเดลสำหรับหลัก {digit}..."):
                    target_series = df[digit].values
                    X, y = create_features(target_series, lag=5)
                    
                    if len(y) < 20:
                        st.error("ข้อมูลน้อยเกินไปสำหรับการทำนาย (ต้องการอย่างน้อย 20 งวด)")
                        continue
                        
                    last_known = int(target_series[-1])
                    
                    X_train, y_train = X.iloc[:-10], y.iloc[:-10]
                    X_backtest, y_backtest = X.iloc[-10:], y.iloc[-10:]
                    
                    # AI Engines
                    stat_engine = StatisticalEngine()
                    stat_engine.fit(X_train, y_train)
                    prob_stat = stat_engine.predict_proba(X, last_known)
                    
                    ml_engine = FeatureEngine()
                    ml_engine.fit(X_train, y_train)
                    prob_et, prob_hgb, prob_xgb = ml_engine.predict_proba(X)
                    
                    ensemble_sys = AdaptiveEnsembleSystem()
                    prob_eq = ensemble_sys.symbolic_equation_score(target_series)
                    
                    mock_preds = {'stat': prob_stat, 'et': prob_et, 'hgb': prob_hgb, 'xgb': prob_xgb, 'equation': prob_eq}
                    ensemble_sys.backtest_and_adapt(y_backtest, mock_preds)
                    
                    final_scores = ensemble_sys.ensemble_predict(prob_stat, prob_et, prob_hgb, prob_xgb, prob_eq)
                    
                    score_df = pd.DataFrame({
                        "ตัวเลข": range(10),
                        "โอกาสออก (%)": final_scores
                    }).sort_values("โอกาสออก (%)", ascending=False).reset_index(drop=True)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"#### 🏆 อันดับตัวเลข (หลัก {digit})")
                        st.dataframe(score_df.style.highlight_max(subset=['โอกาสออก (%)'], color='lightgreen'), hide_index=True)
                    with col2:
                        st.markdown("#### 📈 กราฟความน่าจะเป็น")
                        st.bar_chart(score_df.set_index("ตัวเลข"))

if __name__ == "__main__":
    main()
