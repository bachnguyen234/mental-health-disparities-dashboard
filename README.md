# Mental Health Disparities Dashboard

An interactive data dashboard exploring **mental health disparities, treatment access, and barriers to care in the United States** using data from the 2023 National Survey on Drug Use and Health (NSDUH).

The dashboard allows users to examine patterns in mental health treatment and unmet needs across demographic and socioeconomic groups, including age, gender, race, income, and metropolitan status.

## Dashboard

The interactive dashboard provides several perspectives on mental health disparities:

- **Access** — Explore patterns in mental health treatment and help-seeking
- **Barriers** — Examine reported barriers to receiving mental health care
- **Gap** — Investigate unmet mental health needs
- **Region** — Compare patterns across metropolitan and nonmetropolitan areas
- **Summary Statistics** — Review key descriptive statistics

Users can filter the dashboard by:

- Gender
- Age group
- Race/ethnicity
- Income
- Region type

## Data

The analysis uses a subset of the **2023 National Survey on Drug Use and Health (NSDUH)** collected by the Substance Abuse and Mental Health Services Administration (SAMHSA).

The dataset includes variables related to:

- Demographics
- Mental health treatment
- Help-seeking behavior
- Barriers to care
- Mental health conditions

## Methods

The project includes:

- Data cleaning and recoding
- Descriptive statistical analysis
- Demographic group comparisons
- Interactive data visualization
- Dashboard-based exploratory analysis

## Technology

- Python
- pandas
- NumPy
- Plotly
- Dash
- Dash Bootstrap Components
- Jupyter Notebook

## Repository Structure

```text
├── assets/                         # Dashboard styling and assets
├── data/                           # Dashboard data
├── health_dashboard.py             # Main Dash application
├── health_interactive_dashboard.ipynb
├── export_dashboard.py             # Dashboard export utility
├── dash_practice.ipynb             # Development notebook
├── dashboard_snapshot.html         # Static dashboard snapshot
└── README.md

```
## Running the Dashboard

Install the required Python packages:
```python
pip install pandas numpy plotly dash dash-bootstrap-components
```
Then run:
```python
python health_dashboard.py
```
The Dash application will launch locally in your browser.

## Course Context

This project was completed as the final project for PSYC 270: Health Psychology at Denison University.
