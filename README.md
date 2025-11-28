# faculty_dashboard_NITJ
Analysing the publications of NITJ Faculties
# 📚 Research Publications Analytics Dashboard

A comprehensive end-to-end solution for scraping, analyzing, and visualizing academic research output. This project automates the collection of faculty publication data using Google Scholar and presents actionable insights through an interactive Streamlit dashboard.

## 🚀 Project Overview

This project consists of two main modules:
1.  **Data Collection (`b.py`):** A Python script that utilizes the SerpApi to scrape real-time publication data from Google Scholar profiles.
2.  **Analytics Dashboard (`a.py`):** An interactive web application built with Streamlit that processes the raw data to generate insights on research domains, citation trends, and faculty productivity.

## ✨ Features

### 1. Data Collection
* **Automated Scraping:** Fetches titles, publication years, venue/journal names, authors, and citation counts.
* **Pagination Handling:** Automatically iterates through all pages of an author's Google Scholar profile to ensure complete data retrieval.
* **Data Export:** Saves raw data into structured formats (Excel/CSV) for processing.

### 2. Dashboard & Analytics
* **Hierarchical Views:** Drill-down capabilities from **Institution Overview** → **Department Level** → **Individual Faculty Profile**.
* **Automated Classification:** Uses Natural Language Processing (keyword matching) to categorize papers into domains like *Machine Learning, IoT, Cloud Computing*, etc.
* **Smart Imputation:** Automatically classifies publications as "Conference" or "Journal" based on venue naming conventions.
* **Key Metrics:** Visualizes Total Citations, H-Index, Publication Productivity over years, and Research Interest distribution.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Data Collection:** `google-search-results` (SerpApi), `pandas`
* **Web Framework:** `streamlit`
* **Visualization:** `plotly.express`, `plotly.graph_objects`
* **Data Processing:** `pandas`, `collections`, `re`

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-project-folder>
    ```

2.  **Install required dependencies:**
    Ensure you have `Python` installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration & Usage

### Step 1: Data Collection (`b.py`)
This script scrapes data for a specific author.

1.  Open `b.py`.
2.  Replace the `API_KEY` with your valid SerpApi key.
3.  Set the `AUTHOR_ID` variable to the Google Scholar ID of the target researcher.
4.  Run the script:
    ```bash
    python b.py
    ```
    *Output:* This will generate a file named `data.xlsx`.
    *(Note: Ensure you convert this to CSV format or update the dashboard to read the Excel file if necessary).*

### Step 2: Launch Dashboard (`a.py`)
This script runs the visualization interface.

1.  Ensure your dataset (e.g., `SSG DATA - Sheet1 (3).csv`) is in the root directory.
2.  Run the Streamlit app:
    ```bash
    streamlit run a.py
    ```
3.  The dashboard will open in your default web browser (usually at `http://localhost:8501`).

## 📂 Project Structure

├── a.py # Main Streamlit Dashboard application ├── b.py # Data Scraping script (SerpApi) ├── SSG DATA - Sheet1 (3).csv # Dataset used by the dashboard ├── requirements.txt # Python dependencies └── README.md # Project documentation


## 🔄 Methodology

The project follows a 4-step data pipeline:

1.  **Collection:** Web scraping via Google Scholar to gather raw publication metadata.
2.  **Cleaning:** Sanitizing missing values and deriving missing columns (e.g., inferring `Publication Type` from journal names).
3.  **Extraction:** Applying keyword-based algorithms to classify research papers into specific technological domains.
4.  **Visualization:** Rendering interactive Plotly charts within the Streamlit UI for user-friendly analysis.

## 🤝 Contribution

Feel free to fork this project and submit pull requests. You can also open issues for bugs or feature suggestions.

## 📄 License

[MIT License](LICENSE)
