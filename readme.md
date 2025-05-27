# Kovai.co Task: Public Transport Passenger Forecasting

This project focuses on forecasting daily passenger counts for various public transport services using Facebook's Prophet time series forecasting model. The services analyzed include:

- Local Route
- Light Rail
- Peak Service
- Rapid Route
- School Service

##  Repository Contents

- `data.csv` – Cleaned dataset containing historical passenger counts.
- `forecasting.py` – Python script implementing the forecasting pipeline.
- `forecasts/` – Directory containing forecast plots and result CSV files.
- `Task Report.pdf` – Two-page report detailing key insights and technical methodology.

##  Key Insights

- **Local Route** and **Peak Service** exhibit the highest average passenger counts, indicating heavy usage.
- **School Service** shows distinct spikes during school days and drops sharply on weekends and holidays.
- A strong weekly seasonality is observed, with higher ridership on weekdays across all services.
- **Rapid Route** demonstrates a gradual upward trend over the past year, likely due to route expansions or population growth.
- Positive correlation exists between **Local Route** and **Peak Service** passenger counts, suggesting overlapping demand during peak hours.

##  Technical Overview

- **Model Used**: [Facebook Prophet](https://facebook.github.io/prophet/)
- **Seasonality**: Daily, Weekly, Yearly (if data spans over a year), and Monthly (custom seasonality added for data over 30 days).
- **Seasonality Mode**: Multiplicative
- **Forecast Horizon**: 7 days
- **Confidence Interval**: 95%

##  Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/lingesh0/Kovai.co_task.git
   cd Kovai.co_task
   ```

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure that you have Python 3.7 or higher installed.*

4. **Run the Forecasting Script**:
   ```bash
   python forecasting.py
   ```

   The script will generate forecast plots and CSV files in the `forecasts/` directory.

##  Output Files

- **Forecast Plots**: Visual representations of the forecasted passenger counts for each service.
- **Components Plots**: Decomposition of the time series into trend, weekly, and yearly components.
- **CSV Files**:
  - `7day_passenger_forecast.csv`: Contains the forecasted passenger counts for the next 7 days.
  - `forecast_summary.csv`: Summary statistics comparing historical and forecasted data.

##  Report

Refer to `Task Report.pdf` for a detailed analysis, including:

- Key insights derived from the data.
- Technical explanation of the forecasting methodology.
- Model parameters and configurations.

