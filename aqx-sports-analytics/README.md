# 🏀 DeepPitch: NBA Finals Prediction & Analytics

[![Hackathon](https://img.shields.io/badge/Hackathon-AQX_Sports_Analytics_Data_Bowl_2.0-blue.svg)](https://devpost.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

**DeepPitch** is a predictive analytics tool designed for the *AQX Sports Analytics Data Bowl 2.0* hackathon. It uses the history of NBA Finals and player/team statistics to predict whether a finals series will end in a **Sweep** (a crushing 4-0 victory) using a Machine Learning model.

## Practical Application (Use Case)
In professional sports, anticipating the dynamics of a playoff series is crucial for:
- **Coaches and analysts:** Adapting physical strategy (should key players be rested if the probability of a Sweep is high?).
- **Broadcasters and media:** Anticipating audience and series duration for television programming.
- **Bookmakers:** Refining odds based on historical dynamics.

---

## How it Works (Architecture)
The project relies on a complete pipeline from raw data to the user interface:
1. **Cleaning (Cleaner):** Handling specific encodings (`latin1`), removing empty rows, and intelligent imputation of missing values (Median for numbers, "Unknown" for text).
2. **Feature Engineering:** Transforming text data into numerical data usable by the algorithm via `LabelEncoder`.
3. **Modeling (Intelligence):** Training a **Random Forest Classifier** model that identifies the 3 most determining variables (features) for predicting a Sweep during the NBA finals.
4. **Interface (Dashboard):** An interactive web application using **Streamlit** allowing the user to modify parameters in real-time and get an instant prediction.

---

## Installation and Prerequisites

To test this project on your machine, please follow these steps in your terminal:

### 1. Clone the project and prepare the environment
It is highly recommended to use a virtual environment (venv) to avoid dependency conflicts.

```bash
# Navigate to the project folder
cd aqx-sports-analytics

# Create a virtual environment (MacOS/Linux)
python3 -m venv venv

# Activate the environment
source venv/bin/activate
# (On Windows, use: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Guide (How to test)

The project is designed modularly. You can either run the complete data pipeline or directly launch the web interface (pre-trained models are already included).

### Option A: Launch the interactive Dashboard directly
This is the recommended method to visualize the final project.

```bash
streamlit run src/app.py
```
Your browser will automatically open to `http://localhost:8501`. Use the sidebar to adjust the statistics and click on "Predict the Outcome".

### Option B: Re-generate the data and re-train the model
If you want to see how the code processes the raw data:

**1. Cleaning and Feature Engineering:**
```bash
python -m src.data_processing.features
```
Generates the `data/processed/nba_processed.csv` file from the raw data.

**2. Model Training:**
```bash
python -m src.model.train
```
Trains the Random Forest, displays the accuracy in the terminal, and saves the new weights in `src/model/weights/`.

---

## 📂 Project Structure

```text
aqx-sports-analytics/
├── data/
│   ├── raw/                 # Raw data (NBA Finals and MVP.csv)
│   └── processed/           # Cleaned and encoded data (nba_processed.csv)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # Initial pandas exploration
├── src/
│   ├── data_processing/
│   │   ├── cleaner.py       # Imputation and null handling
│   │   └── features.py      # Categorical encoding
│   ├── model/
│   │   ├── train.py         # Random Forest training script
│   │   └── weights/         # Model weights (.pkl)
│   ├── app.py               # Streamlit web interface
│   └── config.py            
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🧑‍💻 Technologies Used

* **Python 3**
* **Pandas & NumPy** *(Data manipulation)*
* **Scikit-Learn** *(Machine Learning / Random Forest)*
* **Streamlit** *(Web Interface)*
