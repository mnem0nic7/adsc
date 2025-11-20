# SpaceX Falcon 9 Data Wrangling Process
## Processing Launch Data for Machine Learning Classification

---

## 🎯 **Objective**
Transform raw SpaceX launch data into a clean, structured format with binary classification labels (successful/unsuccessful landing) for supervised machine learning models.

---

## 📊 **Complete Data Processing Flowchart**

```
┌─────────────────────────────────────────────────────────────────────┐
│              START: Raw Data from API/Web Scraping                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA LOADING & INITIAL INSPECTION                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  • Load dataset: df = pd.read_csv(dataset_part_1.csv)               │
│  • Display first 10 rows: df.head(10)                               │
│  • Check data shape and structure                                   │
│  • Identify columns and data types                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: DATA QUALITY ASSESSMENT                                    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  • Calculate missing values: df.isnull().sum()                      │
│  • Calculate percentage missing: df.isnull().sum()/len(df)*100      │
│  • Identify data types: df.dtypes                                   │
│  • Classify columns:                                                │
│    ├─ Numerical: FlightNumber, PayloadMass, etc.                   │
│    └─ Categorical: LaunchSite, Orbit, Outcome, etc.                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: EXPLORATORY DATA ANALYSIS (EDA)                            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TASK 1: Analyze Launch Sites                               │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Method: df['LaunchSite'].value_counts()                  │   │
│  │  • Sites identified:                                        │   │
│  │    - CCAFS SLC 40 (Cape Canaveral)                         │   │
│  │    - VAFB SLC 4E (Vandenberg)                              │   │
│  │    - KSC LC 39A (Kennedy Space Center)                     │   │
│  │  • Understand launch frequency per site                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TASK 2: Analyze Orbit Types                                │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Method: df['Orbit'].value_counts()                       │   │
│  │  • Orbit types found:                                       │   │
│  │    - LEO (Low Earth Orbit)                                  │   │
│  │    - GTO (Geostationary Transfer Orbit)                    │   │
│  │    - ISS (International Space Station)                     │   │
│  │    - SSO (Sun-Synchronous Orbit)                           │   │
│  │    - MEO, HEO, GEO, PO, VLEO, ES-L1                        │   │
│  │  • Count occurrences of each orbit type                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TASK 3: Analyze Landing Outcomes                           │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Method: landing_outcomes = df['Outcome'].value_counts()  │   │
│  │  • Outcome types discovered:                                │   │
│  │    ✅ True Ocean - Successfully landed in ocean            │   │
│  │    ❌ False Ocean - Unsuccessful ocean landing             │   │
│  │    ✅ True RTLS - Successfully landed on ground pad        │   │
│  │    ❌ False RTLS - Unsuccessful ground pad landing         │   │
│  │    ✅ True ASDS - Successfully landed on drone ship        │   │
│  │    ❌ False ASDS - Unsuccessful drone ship landing         │   │
│  │    ❌ None ASDS - No landing attempt (failure)             │   │
│  │    ❌ None None - No landing attempt (failure)             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: FEATURE ENGINEERING - CREATE CLASSIFICATION LABELS         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TASK 4: Define Bad Outcomes                                │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Create set of unsuccessful outcomes:                     │   │
│  │    bad_outcomes = {                                         │   │
│  │      'False Ocean',                                         │   │
│  │      'False RTLS',                                          │   │
│  │      'False ASDS',                                          │   │
│  │      'None ASDS',                                           │   │
│  │      'None None'                                            │   │
│  │    }                                                        │   │
│  │  • Index identification: landing_outcomes.keys()[[1,3,5,6,7]]│  │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Binary Classification Logic                                 │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │                                                              │   │
│  │  IF Outcome in bad_outcomes:                                │   │
│  │      landing_class = 0  (Unsuccessful Landing)              │   │
│  │  ELSE:                                                       │   │
│  │      landing_class = 1  (Successful Landing)                │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  • Implementation:                                                   │
│    landing_class = [0 if outcome in bad_outcomes                    │
│                     else 1 for outcome in df['Outcome']]            │
│                                                                       │
│  • Add to dataframe: df['Class'] = landing_class                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: VALIDATION & METRICS                                       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  • Calculate success rate: success_rate = df["Class"].mean()        │
│  • Verify classification distribution                               │
│  • Check for class imbalance                                        │
│  • Preview transformed data: df.head()                              │
│  • Validate 'Class' column: df[['Class']].head(8)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: DATA EXPORT                                                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  • Export cleaned data: df.to_csv("dataset_part_2.csv")             │
│  • Remove index: index=False                                        │
│  • Ready for next analysis stage (EDA, ML modeling)                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              END: Clean Data Ready for Machine Learning              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 **Key Data Wrangling Steps**

### **1. Missing Value Analysis**
```python
┌──────────────────────────────────────────────────────────┐
│  Calculate Missing Data Percentage                        │
│  ─────────────────────────────────────────────────────   │
│  missing_pct = df.isnull().sum() / len(df) * 100        │
│                                                           │
│  Purpose:                                                 │
│  • Identify data quality issues                          │
│  • Determine if imputation is needed                     │
│  • Decide on column retention                            │
└──────────────────────────────────────────────────────────┘
```

### **2. Data Type Classification**
```python
┌──────────────────────────────────────────────────────────┐
│  Identify Column Types                                    │
│  ─────────────────────────────────────────────────────   │
│  df.dtypes                                               │
│                                                           │
│  Numerical Columns:                                       │
│    • FlightNumber (int)                                  │
│    • PayloadMass (float)                                 │
│    • Block (int)                                         │
│                                                           │
│  Categorical Columns:                                     │
│    • LaunchSite (object)                                 │
│    • Orbit (object)                                      │
│    • Outcome (object)                                    │
│    • BoosterVersion (object)                             │
└──────────────────────────────────────────────────────────┘
```

### **3. Categorical Analysis**
```python
┌──────────────────────────────────────────────────────────┐
│  Value Counts for Categorical Features                   │
│  ─────────────────────────────────────────────────────   │
│                                                           │
│  Launch Sites:                                            │
│  • df['LaunchSite'].value_counts()                       │
│  • Distribution across facilities                        │
│                                                           │
│  Orbit Types:                                             │
│  • df['Orbit'].value_counts()                            │
│  • Frequency of orbital destinations                     │
│                                                           │
│  Landing Outcomes:                                        │
│  • df['Outcome'].value_counts()                          │
│  • Success/failure patterns                              │
└──────────────────────────────────────────────────────────┘
```

### **4. Label Creation Logic**
```python
┌──────────────────────────────────────────────────────────┐
│  Binary Classification Label Creation                     │
│  ─────────────────────────────────────────────────────   │
│                                                           │
│  Step 1: Define failure outcomes                         │
│  bad_outcomes = set(landing_outcomes.keys()[[1,3,5,6,7]])│
│                                                           │
│  Step 2: Apply conditional logic                         │
│  landing_class = [                                       │
│      0 if outcome in bad_outcomes else 1                 │
│      for outcome in df['Outcome']                        │
│  ]                                                       │
│                                                           │
│  Step 3: Add to DataFrame                                │
│  df['Class'] = landing_class                             │
│                                                           │
│  Result:                                                  │
│  • Class = 0 → Unsuccessful landing                      │
│  • Class = 1 → Successful landing                        │
└──────────────────────────────────────────────────────────┘
```

---

## 📈 **Data Transformation Summary**

```
BEFORE WRANGLING              AFTER WRANGLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Outcome Column                Class Column (Target Variable)
─────────────────            ─────────────────────────────────

✅ True Ocean       ────►    1 (Successful)
❌ False Ocean      ────►    0 (Unsuccessful)
✅ True RTLS        ────►    1 (Successful)
❌ False RTLS       ────►    0 (Unsuccessful)
✅ True ASDS        ────►    1 (Successful)
❌ False ASDS       ────►    0 (Unsuccessful)
❌ None ASDS        ────►    0 (Unsuccessful)
❌ None None        ────►    0 (Unsuccessful)

Multiple categorical        Single binary classification
outcome strings             label for supervised learning
```

---

## 🎯 **Key Outcomes & Metrics**

### **Success Rate Calculation**
```python
success_rate = df["Class"].mean()
# Returns proportion of successful landings (value between 0 and 1)
# Example: 0.67 = 67% success rate
```

### **Data Quality Metrics**
- ✅ Missing value percentages calculated for each column
- ✅ Data types validated (numerical vs categorical)
- ✅ Class distribution balanced/imbalanced identified
- ✅ Total records processed and validated

---

## 🔄 **Data Flow Pipeline**

```
Raw Data          Missing Value      Type              Categorical
Collection   →    Analysis      →    Classification →  Analysis
(API/Web)         (% null)           (dtypes)          (value_counts)
                                                              │
                                                              ▼
                                                        Pattern
                                                        Recognition
                                                              │
                                                              ▼
Export CSV   ←    Add Class      ←    Create Binary  ←  Define Bad
dataset_part_2    Column              Labels             Outcomes
```

---

## 💡 **Data Processing Techniques Applied**

### **1. List Comprehension**
```python
# Efficient way to create landing_class
landing_class = [0 if outcome in bad_outcomes 
                 else 1 for outcome in df['Outcome']]
```

### **2. Set Operations**
```python
# Fast lookup for bad outcomes
bad_outcomes = set(landing_outcomes.keys()[[1,3,5,6,7]])
# O(1) lookup time for membership testing
```

### **3. Pandas Methods**
```python
# Value counting for categorical analysis
df['LaunchSite'].value_counts()

# Missing value analysis
df.isnull().sum() / len(df) * 100

# Type checking
df.dtypes

# Statistical summary
df["Class"].mean()  # Success rate
```

---

## 📊 **Orbit Type Reference**

| Orbit | Full Name | Description | Altitude |
|-------|-----------|-------------|----------|
| LEO | Low Earth Orbit | Earth-centered orbit | < 2,000 km |
| VLEO | Very Low Earth Orbit | Below standard LEO | < 450 km |
| GTO | Geostationary Transfer Orbit | Transfer to GEO | Variable |
| SSO/SO | Sun-Synchronous Orbit | Nearly polar orbit | ~600-800 km |
| ISS | International Space Station | Modular space station | ~408 km |
| MEO | Medium Earth Orbit | Between LEO and GEO | 2,000-35,786 km |
| HEO | Highly Elliptical Orbit | High eccentricity orbit | Variable |
| GEO | Geostationary Orbit | Stationary above equator | 35,786 km |
| PO | Polar Orbit | Passes over both poles | Variable |
| ES-L1 | Earth-Sun Lagrange Point 1 | Gravitational equilibrium | ~1.5M km |

---

## 🛠️ **Tools & Libraries Used**

```python
import pandas as pd        # Data manipulation and analysis
import numpy as np         # Numerical operations and arrays
```

**Pandas Operations:**
- `read_csv()` - Load data
- `head()` - Preview data
- `isnull()` - Detect missing values
- `dtypes` - Check data types
- `value_counts()` - Count categorical values
- `to_csv()` - Export data

**NumPy Operations:**
- Array operations for numerical analysis

---

## 📋 **Data Validation Checklist**

✅ **Data Loading**
- [x] CSV file loaded successfully
- [x] All columns present
- [x] Data types correct

✅ **Quality Assessment**
- [x] Missing values identified
- [x] Percentage of missing data calculated
- [x] Data types classified

✅ **Exploratory Analysis**
- [x] Launch sites counted
- [x] Orbit types analyzed
- [x] Landing outcomes categorized

✅ **Feature Engineering**
- [x] Bad outcomes defined
- [x] Binary labels created
- [x] Class column added to DataFrame

✅ **Validation**
- [x] Success rate calculated
- [x] Data distribution verified
- [x] Sample records reviewed

✅ **Export**
- [x] Clean data exported to CSV
- [x] Ready for next analysis stage

---

## 🔗 **GitHub Repository Reference**

### **Data Wrangling Notebook**
**Repository:** `adsc`  
**Owner:** `mnem0nic7`  
**Branch:** `main`

**Direct Links to Data Wrangling Notebooks:**

1. **Data Wrangling Notebook:**  
   📓 [03-labs-jupyter-spacex-Data wrangling-v2.ipynb](https://github.com/mnem0nic7/adsc/blob/main/03-labs-jupyter-spacex-Data%20wrangling-v2.ipynb)
   - Missing value analysis
   - Launch site frequency analysis
   - Orbit type distribution
   - Landing outcome categorization
   - Binary classification label creation

2. **API Data Collection (Input Source):**  
   📓 [01-jupyter-labs-spacex-data-collection-api-v2.ipynb](https://github.com/mnem0nic7/adsc/blob/main/01-jupyter-labs-spacex-data-collection-api-v2.ipynb)
   - Raw data collection from SpaceX API
   - Initial data structuring

3. **Web Scraping (Input Source):**  
   📓 [02-jupyter-labs-webscraping.ipynb](https://github.com/mnem0nic7/adsc/blob/main/02-jupyter-labs-webscraping.ipynb)
   - Wikipedia data extraction
   - HTML table parsing

**Full Repository:**  
🔗 https://github.com/mnem0nic7/adsc

**Clone Command:**
```bash
git clone https://github.com/mnem0nic7/adsc.git
```

---

## 🎓 **Skills Demonstrated**

- ✅ Data quality assessment and validation
- ✅ Missing value analysis and handling
- ✅ Categorical data analysis and encoding
- ✅ Feature engineering for machine learning
- ✅ Binary classification label creation
- ✅ Exploratory data analysis (EDA)
- ✅ Pandas DataFrame manipulation
- ✅ Data transformation and cleaning
- ✅ Statistical analysis and metrics calculation
- ✅ Data export and pipeline preparation

---

## 📝 **Wrangling Summary**

**Input:** Raw SpaceX launch records with multiple categorical landing outcomes

**Processing:**
1. Load and inspect data structure
2. Assess data quality (missing values, types)
3. Analyze categorical distributions (sites, orbits, outcomes)
4. Define success/failure criteria
5. Create binary classification labels
6. Validate and calculate success metrics

**Output:** Clean dataset with binary 'Class' column (0=failure, 1=success) ready for machine learning model training

---

## 🎯 **Purpose for Peer Review**

This data wrangling process demonstrates:
- **Systematic approach** to data quality assessment
- **Domain understanding** of SpaceX landing outcomes
- **Feature engineering** skills for classification problems
- **Reproducible pipeline** for data transformation
- **Clear documentation** of decision-making process

The notebooks are available in the GitHub repository for code review, validation, and collaborative improvement.

---

*This data wrangling process transforms raw SpaceX launch data into a clean, labeled dataset suitable for supervised machine learning classification models to predict landing success.*
