import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------
# ข้อมูลลิงก์หวยที่อัปเดต
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
# 1. จัดการข้อมูล (Mock / Web Scraper Placeholder)
# -----------------------------------------
def generate_mock_data(n_samples=200):
    """จำลองผลหวยย้อนหลัง (หลักร้อย, หลักสิบ, หลักหน่วย)"""
    np.random.seed(42)
    data = {
        'Draw_ID': range(1, n_samples + 1),
        'Hundreds': np.random.randint(0, 10, n_samples),
        'Tens': np.random.randint(0, 10, n_samples),
        'Units': np.random.randint(0, 10, n_samples)
    }
    return pd.DataFrame(data)

def scrape_data_from_url(url):
    """
    (พื้นที่สำหรับเขียนโค้ด Web Scraping ในอนาคต)
    ใช้ BeautifulSoup หรือ requests ดึงตารางจาก Blogspot
    """
    # ตอนนี้ return เป็นข้อมูลจำลองแทนไปก่อน
    return generate_mock_data(200)

# -----------------------------------------
# 2. Engines (อิงตามสถาปัตยกรรมเดิม)
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
# 3. Adaptive & Ensemble System
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
# 4. Streamlit UI & Main Logic
# -----------------------------------------
def create_features(series, lag=3):
    df = pd.DataFrame(series)
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
    df.dropna(inplace=True)
    X = df.drop(df.columns[0], axis=1)
    y = df.iloc[:, 0].astype(int)
    return X, y

def main():
    st.set_page_config(page_title="Advanced AI Lottery Predictor", layout="wide")
    st.title("🎲 Advanced AI Lottery Predictor")
    st.markdown("ระบบทำนายด้วยสถาปัตยกรรม **Ensemble AI & Adaptive Self-Correction**")
    
    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ การตั้งค่าประเภทหวย")
    
    # เมนูเลือกประเภทหวยจาก Dictionary ที่ให้มา
    lottery_type = st.sidebar.selectbox("เลือกประเภทหวย", list(LOTTERY_SOURCES.keys()))
    
    # แสดงลิงก์ไปยัง Blogspot
    source_url = LOTTERY_SOURCES[lottery_type]
    st.sidebar.markdown(f"**แหล่งข้อมูลอ้างอิง:**\n[🔗 ไปที่เว็บ {lottery_type}]({source_url})")
    st.sidebar.markdown("---")
    
    st.sidebar.header("⚙️ การรับข้อมูล (Data Source)")
    data_option = st.sidebar.radio("เลือกแหล่งข้อมูล", ["ใช้ข้อมูลจำลอง (Mock Data)", "ดึงข้อมูลจาก URL (ทดลอง)", "อัปโหลด CSV"])
    
    # ตรวจสอบตัวเลือกการโหลดข้อมูล
    if data_option == "ใช้ข้อมูลจำลอง (Mock Data)":
        df = generate_mock_data(200)
    elif data_option == "ดึงข้อมูลจาก URL (ทดลอง)":
        st.sidebar.info("ฟีเจอร์กำลังพัฒนา (Web Scraping): ปัจจุบันจะใช้ Mock Data แทนไปก่อนจนกว่าจะมีการเขียนโค้ด BeautifulSoup ดึงตารางจากหน้าเว็บ")
        df = scrape_data_from_url(source_url)
    else:
        uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ผลย้อนหลัง (CSV)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("กรุณาอัปโหลดไฟล์ CSV หรือเปลี่ยนไปใช้ข้อมูลจำลอง")
            return
            
    # ---------------- Main Page ----------------
    st.subheader(f"📊 ข้อมูลผลย้อนหลัง: **{lottery_type}**")
    st.dataframe(df.tail(5))
    
    if st.button(f"🚀 รัน AI วิเคราะห์ {lottery_type}", type="primary"):
        digits_to_predict = ['Hundreds', 'Tens', 'Units']
        
        # ถ้าระบบเจอแค่เลข 2 ตัว (กรณีหวยประเภทอื่น)
        if 'Hundreds' not in df.columns:
             digits_to_predict = ['Tens', 'Units']

        with st.spinner(f"AI กำลังวิเคราะห์ผลย้อนหลังของ {lottery_type}..."):
            for digit in digits_to_predict:
                if digit not in df.columns: continue
                
                st.markdown(f"### วิเคราะห์ความน่าจะเป็นของหลัก: **{digit}**")
                
                target_series = df[digit].values
                X, y = create_features(target_series, lag=5)
                last_known = int(target_series[-1])
                
                X_train, y_train = X.iloc[:-10], y.iloc[:-10]
                X_backtest, y_backtest = X.iloc[-10:], y.iloc[-10:]
                
                # 1. Statistical Engine
                stat_engine = StatisticalEngine()
                stat_engine.fit(X_train, y_train)
                prob_stat = stat_engine.predict_proba(X, last_known)
                
                # 2. Feature Engine (ML)
                ml_engine = FeatureEngine()
                ml_engine.fit(X_train, y_train)
                prob_et, prob_hgb, prob_xgb = ml_engine.predict_proba(X)
                
                # 3. Ensemble & Equation Engine
                ensemble_sys = AdaptiveEnsembleSystem()
                prob_eq = ensemble_sys.symbolic_equation_score(target_series)
                
                # 4. Backtest & Self-Correction
                mock_preds = {'stat': prob_stat, 'et': prob_et, 'hgb': prob_hgb, 'xgb': prob_xgb, 'equation': prob_eq}
                ensemble_sys.backtest_and_adapt(y_backtest, mock_preds)
                
                # 5. สรุปคะแนน
                final_scores = ensemble_sys.ensemble_predict(prob_stat, prob_et, prob_hgb, prob_xgb, prob_eq)
                
                score_df = pd.DataFrame({
                    "ตัวเลข (0-9)": range(10),
                    "โอกาสออก (%)": final_scores
                }).sort_values("โอกาสออก (%)", ascending=False).reset_index(drop=True)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(score_df, hide_index=True)
                with col2:
                    st.bar_chart(score_df.set_index("ตัวเลข (0-9)"))
                    
            st.success(f"✅ ประมวลผล {lottery_type} เสร็จสิ้น!")

if __name__ == "__main__":
    main()
