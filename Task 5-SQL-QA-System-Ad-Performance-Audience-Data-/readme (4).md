# 🧠 Task 5 – SQL QA System

### 🎯 Objective
Enable querying of structured ad performance and audience data using natural language.

### 🛠 Tools Used
- SQLite (Database)
- LangChain `SQLDatabaseChain`
- Gemini Pro (LLM)

### 📝 Database Schema
- **campaigns**: id, client, platform, spend, impressions, clicks, conversions  
- **audiences**: id, campaign_id, age_group, region, engagement_score  
- **influencers**: id, name, niche, followers, avg_engagement

### 📈 Implementation Steps
1. Created SQLite DB `advision.db` with ≥30 sample rows  
2. Integrated LangChain with Gemini Pro using `SQLDatabaseChain`  
3. Translated natural language questions into SQL queries  
4. Executed SQL queries and returned natural language answers

### 💬 Example Queries
- “Show top 3 campaigns by conversion rate among 18–25 year olds last month.”
- “Which platform had the highest number of impressions?”
- “Total spend by Apple.”
- “List influencers with engagement above 5.”

### 📁 Deliverables
- `sql_qa_system.ipynb` – Jupyter Notebook implementation  
- `advision.db` – SQLite Database  
- `README.md` – Documentation  
- `explanation_video.mp4` – Short walkthrough (3–6 min)
