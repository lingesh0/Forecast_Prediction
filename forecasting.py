import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress Prophet warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
CONFIG = {
    'csv_path': r"D:\kovai.co_task'\data.csv",
    'forecast_periods': 7,
    'service_types': [
        "Local Route",
        "Light Rail",
        "Peak Service",
        "Rapid Route",
        "School"
    ],
    'output_dir': Path("forecasts"),
    'figure_size': (12, 8),
    'dpi': 300
}

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

def load_and_prepare_data(path):
    """Load and prepare data for forecasting."""
    try:
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Check for valid service types in the data
        available_services = [col for col in CONFIG['service_types'] if col in df.columns]
        missing_services = [col for col in CONFIG['service_types'] if col not in df.columns]

        if missing_services:
            logger.warning(f"Missing service types in data: {missing_services}")

        # Reshape from wide to long format
        df_long = df.melt(id_vars=['Date'], value_vars=available_services,
                          var_name='Service Type', value_name='Passenger Count')

        # Clean data
        initial_rows = len(df_long)
        df_long.dropna(subset=['Passenger Count'], inplace=True)
        df_long = df_long[df_long['Passenger Count'] >= 0]  # Remove negative values

        logger.info(f"Data loaded: {len(df_long)} rows (removed {initial_rows - len(df_long)} invalid rows)")
        logger.info(f"Date range: {df_long['Date'].min()} to {df_long['Date'].max()}")

        return df_long, available_services

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def validate_data_for_forecasting(df_service, service_type, min_data_points=10):
    """Validate if there's enough data for reliable forecasting."""
    if len(df_service) < min_data_points:
        logger.warning(f"Insufficient data for {service_type}: {len(df_service)} points (minimum: {min_data_points})")
        return False

    # Check for data variability
    if df_service['y'].std() == 0:
        logger.warning(f"No variability in {service_type} data - all values are the same")
        return False

    return True

def forecast_service(df, service_type, periods=7, output_dir=None):
    """Generate forecast for a specific service type."""
    try:
        logger.info(f"Forecasting for {service_type}")

        # Filter and prepare data
        df_service = df[df['Service Type'] == service_type].copy()
        df_agg = df_service.groupby('Date')['Passenger Count'].sum().reset_index()
        df_agg.rename(columns={'Date': 'ds', 'Passenger Count': 'y'}, inplace=True)
        df_agg = df_agg.sort_values('ds').reset_index(drop=True)

        # Validate data
        if not validate_data_for_forecasting(df_agg, service_type):
            return None

        # Configure Prophet model with appropriate seasonality
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True if len(df_agg) > 365 else False,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )

        # Add custom seasonalities if we have enough data
        if len(df_agg) > 30:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Fit model
        model.fit(df_agg)

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Create and save plot
        if output_dir:
            # Main forecast plot
            fig1 = model.plot(forecast, figsize=CONFIG['figure_size'])
            plt.title(f"Passenger Count Forecast - {service_type}", fontsize=14, fontweight='bold')
            plt.xlabel("Date")
            plt.ylabel("Passenger Count")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = output_dir / f"forecast_{service_type.replace(' ', '_').lower()}.png"
            plt.savefig(plot_path, dpi=CONFIG['dpi'], bbox_inches='tight')
            plt.close()

            # Components plot (separate figure)
            fig2 = model.plot_components(forecast, figsize=CONFIG['figure_size'])
            plt.tight_layout()

            components_path = output_dir / f"components_{service_type.replace(' ', '_').lower()}.png"
            plt.savefig(components_path, dpi=CONFIG['dpi'], bbox_inches='tight')
            plt.close()

            logger.info(f"Forecast plot saved: {plot_path}")
            logger.info(f"Components plot saved: {components_path}")

        # Prepare forecast output
        future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
        future_forecast['Service Type'] = service_type
        future_forecast['yhat'] = future_forecast['yhat'].round(0)
        future_forecast['yhat_lower'] = future_forecast['yhat_lower'].round(0)
        future_forecast['yhat_upper'] = future_forecast['yhat_upper'].round(0)

        # Ensure no negative predictions
        future_forecast['yhat'] = future_forecast['yhat'].clip(lower=0)
        future_forecast['yhat_lower'] = future_forecast['yhat_lower'].clip(lower=0)
        future_forecast['yhat_upper'] = future_forecast['yhat_upper'].clip(lower=0)

        return future_forecast

    except Exception as e:
        logger.error(f"Error forecasting {service_type}: {e}")
        return None

def generate_summary_statistics(df_long, forecasts_df):
    """Generate summary statistics for the forecasts."""
    summary = []

    for service in df_long['Service Type'].unique():
        historical_data = df_long[df_long['Service Type'] == service]['Passenger Count']
        # Use renamed column here
        forecast_data = forecasts_df[forecasts_df['Service Type'] == service]['Predicted_Count']

        summary.append({
            'Service Type': service,
            'Historical Avg': historical_data.mean(),
            'Historical Std': historical_data.std(),
            'Forecast Avg': forecast_data.mean() if len(forecast_data) > 0 else 0,
            'Forecast Trend': 'Increasing' if len(forecast_data) > 1 and forecast_data.iloc[-1] > forecast_data.iloc[0] else 'Stable/Decreasing'
        })

    return pd.DataFrame(summary)

def main():
    """Main execution function."""
    try:
        # Setup
        setup_output_directory(CONFIG['output_dir'])

        # Load and prepare data
        df_long, available_services = load_and_prepare_data(CONFIG['csv_path'])

        if df_long.empty:
            logger.error("No valid data found. Exiting.")
            return

        # Generate forecasts
        all_forecasts = []
        successful_forecasts = 0

        for service in available_services:
            forecast_df = forecast_service(
                df_long,
                service,
                periods=CONFIG['forecast_periods'],
                output_dir=CONFIG['output_dir']
            )

            if forecast_df is not None:
                all_forecasts.append(forecast_df)
                successful_forecasts += 1

        if not all_forecasts:
            logger.error("No successful forecasts generated. Exiting.")
            return

        # Combine and save results
        final_forecast = pd.concat(all_forecasts, ignore_index=True)
        final_forecast = final_forecast[['Service Type', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        final_forecast.rename(columns={
            'ds': 'Date',
            'yhat': 'Predicted_Count',
            'yhat_lower': 'Lower_Bound',
            'yhat_upper': 'Upper_Bound'
        }, inplace=True)

        # Save main forecast
        output_path = CONFIG['output_dir'] / f"{CONFIG['forecast_periods']}day_passenger_forecast.csv"
        final_forecast.to_csv(output_path, index=False)

        # Generate and save summary
        summary_df = generate_summary_statistics(df_long, final_forecast)
        summary_path = CONFIG['output_dir'] / "forecast_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Display results
        logger.info(f"Forecasting completed successfully!")
        logger.info(f"Generated forecasts for {successful_forecasts}/{len(available_services)} service types")
        logger.info(f"Main forecast saved to: {output_path}")
        logger.info(f"Summary saved to: {summary_path}")

        print("\n" + "=" * 50)
        print("FORECAST SUMMARY")
        print("=" * 50)
        print(summary_df.to_string(index=False))

        print(f"\n\nDetailed {CONFIG['forecast_periods']}-Day Forecast:")
        print("-" * 50)
        print(final_forecast.to_string(index=False))

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
