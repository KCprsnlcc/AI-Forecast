# Forecasting and Transactions Analysis

This project is designed to analyze transactions and generate forecasts using machine learning models. It includes both a command-line interface and a graphical user interface (GUI).

## Project Structure

```
.gitignore
forecasts.csv               # Contains forecasted data
main.py                     # Main script to run the analysis
gui.py                      # Graphical user interface
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
- PySide6 (for the GUI)

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

1. Place your transaction data in `transactions.csv`.
2. Run the main script:

```bash
python main.py
```

3. The forecasts will be saved in `forecasts.csv`, and visualizations will be exported to the `exports/` directory.

### Graphical User Interface (GUI)

The project includes a user-friendly GUI that provides:
- Dashboard with financial summary and charts
- Transaction management
- Forecast generation and visualization
- Data analysis tools

To launch the GUI:

```bash
python gui.py
```

## Models

The project uses the following models for forecasting:

- **ARIMA**: Stored in `models/arima_model.pkl`
- **Prophet**: Stored in `models/prophet_model.pkl`
- **Other ML Models**: Stored in `models/ml_models.pkl`

## Outputs

- **Forecasts**: Saved in `forecasts.csv`.
- **Visualizations**: Exported as images in the `exports/` directory.

## GUI Features

- **Dashboard**: View financial summary, quick actions, and charts
- **Transactions**: Add and manage income and expense transactions
- **Forecasts**: Generate AI-powered forecasts for different timeframes
- **Analysis**: Analyze income, expenses, and savings with visualizations

## Full Documentation

For complete details about the project, please refer to the [Full Documentation](DOCUMENTATION.md).