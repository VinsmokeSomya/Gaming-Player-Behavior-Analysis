# 🚀 How to Run - Player Retention Analytics Dashboard

> **Get your analytics dashboard running in 5 minutes! This guide will take you from zero to analyzing 40,034 real players.**

---

## ⚡ **Super Quick Start (2 Commands)**

```bash
# 1. Setup everything
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && docker-compose up -d postgres

# 2. Run dashboard  
python app.py
```

**🎉 Open: http://127.0.0.1:8050**

---

## 📋 **Prerequisites Check**

Before we start, make sure you have:

| Requirement | Check | Status |
|-------------|-------|--------|
| **Python 3.9+** | `python --version` | ✅ You have 3.13 |
| **Docker** | `docker --version` | ✅ You have 28.4.0 |
| **Git** | `git --version` | ✅ Available |
| **4GB+ RAM** | Task Manager | ✅ Recommended |

---

## 🎯 **Step-by-Step Guide**

### 🔧 **Step 1: Environment Setup**

```bash
# Navigate to your project directory
cd "Gaming-Player Behavior Analysis"

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# For macOS/Linux users:
# source .venv/bin/activate
```

**✅ Expected Output:**
```
(.venv) PS C:\Users\YourName\Desktop\Gaming-Player Behavior Analysis>
```

---

### 📦 **Step 2: Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt
```

**✅ Expected Output:**
```
Successfully installed dash-3.2.0 pandas-2.3.3 plotly-6.3.0 ... (and many more)
```

**⏱️ Time:** ~2-3 minutes

---

### 🐳 **Step 3: Start Database**

```bash
# Start PostgreSQL database in background
docker-compose up -d postgres
```

**✅ Expected Output:**
```
[+] Running 2/2
 ✔ Network gaming-playerbehavioranalysis_analytics_network  Created
 ✔ Container player_analytics_db                            Started
```

**🔍 Verify Database:**
```bash
# Check if database is running
docker-compose ps
```

**✅ Should show:**
```
NAME                  IMAGE         COMMAND                  SERVICE    CREATED          STATUS                    PORTS
player_analytics_db   postgres:15   "docker-entrypoint.s…"   postgres   30 seconds ago   Up 29 seconds (healthy)   0.0.0.0:5432->5432/tcp
```

---

### 🎮 **Step 4: Launch Dashboard**

```bash
# Start the analytics dashboard
python app.py
```

**✅ Expected Output:**
```
🎮 Starting Player Retention Analytics Dashboard...
📊 Dashboard will be available at: http://127.0.0.1:8050
🔧 Press Ctrl+C to stop the server
Loading real gaming data...
✅ Loaded real data: 40,034 players, 1,411,443 events
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: on
```

**⏱️ Loading Time:** ~10-30 seconds (loading 40,034 real players)

---

### 🌐 **Step 5: Access Your Dashboard**

1. **Open your web browser**
2. **Navigate to:** `http://127.0.0.1:8050`
3. **🎉 You should see the dashboard loading!**

---

## 🎊 **Success! What You'll See**

### 📊 **Dashboard Tabs Available:**

| Tab | Description | What You'll Analyze |
|-----|-------------|-------------------|
| 📊 **Overview** | Key metrics dashboard | Total players, DAU, conversion rates |
| 🔥 **Cohort Analysis** | Retention heatmaps | How player cohorts retain over time |
| 📈 **Engagement Timeline** | Activity charts | Daily/weekly/monthly active users |
| ⚠️ **Churn Analysis** | Risk prediction | Players at risk of leaving |
| 🎯 **Funnel Analysis** | Drop-off visualization | Where players quit in progression |

### 📈 **Real Data You're Analyzing:**
- **40,034 authentic players** with complete profiles
- **1,411,443 real gaming events** (sessions, purchases, levels)
- **$1,780,829 in actual revenue** data
- **6,694,387 total gaming sessions**
- **182,236 level completions**

---

## 🛑 **Stopping the Dashboard**

When you're done analyzing:

```bash
# 1. Stop the dashboard (in the terminal where it's running)
Ctrl+C

# 2. Stop the database
docker-compose stop postgres

# 3. Deactivate virtual environment
deactivate
```

---

## 🔧 **Troubleshooting**

### ❌ **Problem: Dashboard won't start**

**Solution 1: Check port 8050**
```bash
# Kill any process using port 8050
netstat -ano | findstr :8050
taskkill /F /PID <process_id>
```

**Solution 2: Use different port**
```bash
# Edit app.py and change the port
python app.py --port 8051
```

---

### ❌ **Problem: Database connection failed**

**Solution: Restart database**
```bash
# Check database status
docker-compose ps

# Restart if needed
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

---

### ❌ **Problem: Slow loading (taking more than 1 minute)**

**Solution: Switch to sample data temporarily**
```python
# Edit app.py line 66:
data_loader = DataLoader(data_dir="data/sample")  # Use 1,000 players instead
```

Then restart: `python app.py`

---

### ❌ **Problem: Charts not loading**

**Solutions:**
1. **Clear browser cache** (Ctrl+F5)
2. **Try different browser** (Chrome, Firefox, Edge)
3. **Check console** (F12 → Console tab for errors)
4. **Restart dashboard** (Ctrl+C, then `python app.py`)

---

## 🎯 **Next Steps**

### 🔍 **Explore Your Data**
1. **Click around the charts** - They're interactive!
2. **Try the date filters** - Analyze different time periods
3. **Use cross-filtering** - Click one chart to filter others
4. **Export data** - Download insights for further analysis

### 📊 **Key Things to Try**
- **Cohort Heatmap**: Click cells to drill down into specific player groups
- **Churn Analysis**: Identify your highest-risk players
- **Funnel Analysis**: Find where players are dropping off
- **Engagement Timeline**: Spot trends and anomalies

### 🚀 **Advanced Usage**
- Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for advanced configuration
- Explore the source code in `src/` directory
- Run tests with `pytest tests/`

---

## 📞 **Need Help?**

### 🆘 **Quick Help**
- **Dashboard not loading?** → Check [Troubleshooting](#-troubleshooting) above
- **Want more features?** → See [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Found a bug?** → Create an issue on GitHub

### 📚 **Documentation**
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Comprehensive setup guide
- **[README.md](README.md)** - Project overview and features
- **Source Code** - Explore `src/` directory for implementation details

---

## 🎉 **Congratulations!**

**You're now running a production-ready analytics dashboard analyzing 40,034 real players!**

### 🎮 **What You've Accomplished:**
✅ Set up a complete analytics environment  
✅ Loaded 1.4M+ real gaming events  
✅ Connected to a PostgreSQL database  
✅ Launched interactive visualizations  
✅ Ready to make data-driven decisions  

### 🚀 **Ready to Analyze:**
- **Player retention patterns** across different cohorts
- **Churn risk predictions** for proactive player retention
- **Engagement trends** and seasonal patterns
- **Revenue optimization** opportunities
- **Level progression** and difficulty balancing

---

<div align="center">

## 🎊 **Happy Analyzing!**

**Your dashboard is ready at: [http://127.0.0.1:8050](http://127.0.0.1:8050)**

*Transform your player data into actionable insights* 🎮📊

**[📚 Advanced Setup](SETUP_GUIDE.md)** | **[🏠 Back to README](README.md)** | **[🐛 Report Issues](../../issues)**

</div>