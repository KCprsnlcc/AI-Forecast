# Forecasting and Transactions Analysis

This project is designed to analyze transactions and generate forecasts using machine learning models.

## Project Structure

```
.gitignore
forecasts.csv               # Contains forecasted data
main.py                     # Main script to run the analysis
transactions.csv            # Contains transaction data
exports/                    # Directory for exported results
    forecast_comparison.png # Visualization of forecast comparisons
models/                     # Directory for saved machine learning models
    arima_model.pkl         # ARIMA model
    metadata.pkl            # Metadata for models
    ml_models.pkl           # General ML models
    prophet_model.pkl       # Prophet model
```

## Requirements

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- fbprophet (or `prophet` for newer versions)

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your transaction data in `transactions.csv`.
2. Run the main script:

```bash
python main.py
```

3. The forecasts will be saved in `forecasts.csv`, and visualizations will be exported to the `exports/` directory.

## Models

The project uses the following models for forecasting:

- **ARIMA**: Stored in `models/arima_model.pkl`
- **Prophet**: Stored in `models/prophet_model.pkl`
- **Other ML Models**: Stored in `models/ml_models.pkl`

## Outputs

- **Forecasts**: Saved in `forecasts.csv`.
- **Visualizations**: Exported as images in the `exports/` directory.