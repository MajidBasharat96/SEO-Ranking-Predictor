# SEO-Ranking-Predictor

How to Run This (Locally)

Create project structure:
mkdir seo-ranking-predictor
cd seo-ranking-predictor
mkdir data src models app notebooks

Install dependencies:
pip install pandas numpy scikit-learn


Run:
python src/preprocessing.py

-----------------------------------------

How to Run (Full System)
1. Train Model
python main.py

2. Run API
uvicorn app.api:app --reload

3. Run Dashboard
streamlit run app/dashboard.py
