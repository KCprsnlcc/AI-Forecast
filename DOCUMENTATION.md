# AI Finance Tracker - User Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Core Features](#core-features)
   - [Transaction Management](#transaction-management)
   - [Viewing Transactions](#viewing-transactions)
   - [AI Forecasting](#ai-forecasting)
   - [Forecasts and Accuracy](#forecasts-and-accuracy)
   - [Data Analysis](#data-analysis)
   - [Timeframe Predictions](#timeframe-predictions)
   - [Exporting Data](#exporting-data)
5. [Advanced Features](#advanced-features)
   - [Model Retraining](#model-retraining)
   - [AI Insights](#ai-insights)
   - [Visualization](#visualization)
6. [Technical Reference](#technical-reference)
   - [File Structure](#file-structure)
   - [Forecasting Models](#forecasting-models)
   - [Data Format](#data-format)
7. [Troubleshooting](#troubleshooting)
8. [FAQs](#faqs)

## Introduction

Enhanced AI Finance Tracker is a powerful command-line application designed to help you manage your personal finances with the aid of artificial intelligence. The application allows you to track income and expenses, generate financial forecasts using multiple AI models, analyze spending patterns, and receive personalized insights about your financial health.

Key features include:
- Transaction tracking for income and expenses
- Multiple AI forecasting models (ARIMA, Prophet, and category-based ML)
- Financial data analysis with visualizations
- Predictions across different timeframes (3 months, 6 months, 1 year)
- Exporting capabilities for forecasts and analysis
- Personalized financial insights and recommendations

## Installation

### Prerequisites
- Python 3.7 or higher
- Required Python packages:
  - pandas
  - numpy
  - statsmodels
  - prophet
  - scikit-learn
  - joblib
  - matplotlib
  - tabulate
  - colorama

### Setup Instructions

1. Clone or download the application files to your local machine.

2. Install required packages:
   ```
   pip install pandas numpy statsmodels prophet scikit-learn joblib matplotlib tabulate colorama
   ```

3. Run the application:
   ```
   python finance_tracker.py
   ```

## Getting Started

### First-Time Setup

When you first launch the application, it will create the necessary directories and files:
- `transactions.csv`: Stores all your financial transactions
- `forecasts.csv`: Stores forecast history and accuracy
- `exports/`: Directory for exported files and charts
- `models/`: Directory for saved ML models

### Main Menu

The application presents a menu with these options:
1. Add Transaction
2. View Transactions
3. Generate AI Forecasts
4. View Forecasts
5. View Model Accuracy
6. Data Analysis
7. Timeframe Predictions
8. Export Predictions
9. Force Retrain Models
0. Exit

## Core Features

### Transaction Management

#### Adding Transactions
To add a new transaction:
1. Select option `1` from the main menu
2. Enter the transaction type (`income` or `expense`)
3. Enter the amount (must be a positive number)
4. Enter a category (e.g., "Housing", "Food", "Salary")
5. Enter the date in YYYY-MM-DD format (optional, defaults to today)

Transaction types:
- **Income**: Money received (salary, freelance work, gifts, etc.)
- **Expense**: Money spent (rent, groceries, utilities, etc.)

Example:
```
Enter type (income/expense): income
Enter amount: 1500
Enter category (e.g., Housing, Food): Salary
Enter date (YYYY-MM-DD) [optional]: 2025-05-01
✅ Transaction saved.
```

### Viewing Transactions

Select option `2` from the main menu to view your recent transactions.

The transaction view displays:
- Date
- Transaction type (color-coded: green for income, red for expenses)
- Amount
- Category

Additionally, you'll see a summary showing:
- Total Income
- Total Expenses
- Balance (Income - Expenses)

### AI Forecasting

Select option `3` from the main menu to generate AI forecasts. The application uses three different forecasting models:

1. **ARIMA (AutoRegressive Integrated Moving Average)**: A statistical model that uses time series data to predict future values
2. **Prophet**: Facebook's time series forecasting model designed to handle seasonality
3. **Category-based ML**: Machine learning models that predict expenses by category

When you generate forecasts, the application will:
1. Check if there's enough data to make predictions
2. Train or load the necessary models
3. Generate predictions for the next 30 days
4. Display the results of all three models
5. Show AI-generated insights based on your financial history
6. Create visualization charts of the forecasts

The forecast results are automatically saved to enable accuracy tracking over time.

### Forecasts and Accuracy

#### Viewing Forecasts
Select option `4` from the main menu to view your latest forecasts.

The forecast view displays:
- Forecast date
- Model type
- Category (if applicable)
- Predicted amount
- Actual amount (if the date has passed)
- Accuracy percentage (if the date has passed)

#### Viewing Model Accuracy
Select option `5` from the main menu to check how accurate your forecasting models have been.

You can filter the accuracy data by timeframe:
- 3 Months
- 6 Months
- 1 Year
- All

The accuracy view displays:
- Average accuracy by model type
- For category-based models, accuracy by category
- Visualization of model accuracy comparison

### Data Analysis

Select option `6` from the main menu for in-depth analysis of your financial data.

Data analysis options:
- **All**: Overall financial summary
- **Income**: Income patterns and breakdown
- **Expenses**: Expense distribution and category analysis
- **Savings**: Savings trends and savings rate analysis

Each analysis type provides:
- Relevant statistics
- Trend information
- Visualizations (saved to the exports directory)

#### All Data Analysis
Displays:
- Total Income
- Total Expenses
- Net Savings or Deficit
- Monthly averages (if enough data available)

#### Income Analysis
Displays:
- Total Income
- Average Income Entry
- Largest/Smallest Income
- Income by Category (if categories exist)

#### Expense Analysis
Displays:
- Total Expenses
- Average Expense
- Largest/Smallest Expense
- Expenses by Category (with percentages)
- Pie chart of expense distribution

#### Savings Analysis
Displays:
- Total Savings
- Average Monthly Savings
- Overall Savings Rate
- Savings Trend
- Chart showing monthly savings and savings rate

### Timeframe Predictions

Select option `7` from the main menu to generate predictions for specific timeframes.

Timeframe options:
- 3 Months
- 6 Months
- 1 Year

For each timeframe, the application:
1. Uses Prophet to make longer-term predictions
2. Forecasts both expenses and income (if sufficient data)
3. Calculates projected savings or deficit
4. Creates visualizations of the forecast

### Exporting Data

Select option `8` from the main menu to export prediction data.

Export options:
- 3 Months
- 6 Months
- 1 Year

Exports are saved as CSV files in the `exports/` directory with timestamps in the filename.

## Advanced Features

### Model Retraining

Select option `9` from the main menu to force retrain all models.

Models are automatically retrained when:
- New data is available since the last training
- It's been more than 7 days since the last training
- Model files don't exist yet

Manual retraining might be useful when:
- You've added significant new data
- You want to ensure models reflect recent transactions
- You're experiencing unexpected forecast results

### AI Insights

AI insights are automatically generated when you select option `3` (Generate AI Forecasts). These insights include:

- **Savings Trends**: How your savings are changing month-to-month
- **Highest Expense Categories**: Your biggest spending categories
- **Unusual Spending Patterns**: Notifications of spending that's higher or lower than usual
- **Emergency Fund Recommendations**: Suggestions for building an emergency fund based on your income

### Visualization

The application generates various charts to help visualize your financial data:

- **Forecast Comparison**: Comparison of different forecasting models
- **Accuracy Chart**: Visualization of model accuracy
- **Expense Distribution**: Pie chart showing expense breakdown by category
- **Savings Trend**: Bar chart showing monthly savings and line chart showing savings rate
- **Timeframe Forecast**: Line chart showing projected income and expenses

All charts are saved to the `exports/` directory.

## Technical Reference

### File Structure

```
.
├── finance_tracker.py       # Main application file
├── transactions.csv         # Transaction data
├── forecasts.csv            # Forecast history and accuracy data
├── exports/                 # Directory for exported files and charts
│   ├── accuracy_chart.png
│   ├── expense_distribution.png
│   ├── forecast_comparison.png
│   ├── savings_trend.png
│   └── financial_predictions_*.csv
└── models/                  # Directory for saved ML models
    ├── arima_model.pkl
    ├── prophet_model.pkl
    ├── ml_models.pkl
    └── metadata.pkl
```

### Forecasting Models

#### ARIMA Model
- Uses AutoRegressive Integrated Moving Average algorithm
- Analyzes time series patterns to predict future values
- Best for short-term predictions (days to weeks)
- Requires at least 30 data points for reliable predictions

#### Prophet Model
- Developed by Facebook for time series forecasting
- Handles seasonality and holiday effects well
- Better for mid-term predictions (weeks to months)
- Requires at least 10 data points for reliable predictions

#### Category-Based ML Model
- Uses Linear Regression to predict expenses by category
- Considers day of week, day of month, month, and day of year
- Good for identifying category-specific spending patterns
- Requires at least 5 data points per category

### Data Format

#### Transactions CSV Structure
- `date`: Transaction date (YYYY-MM-DD)
- `type`: Transaction type (income/expense)
- `amount`: Transaction amount (positive number)
- `category`: Transaction category (string)

#### Forecasts CSV Structure
- `date_generated`: Date the forecast was created
- `forecast_date`: Date the forecast is for
- `model_type`: Type of forecasting model (arima/prophet/category_ml)
- `category`: Category (for category-based models) or 'all'
- `predicted_amount`: Predicted amount
- `actual_amount`: Actual amount (populated after the date passes)
- `accuracy`: Forecast accuracy percentage (populated after the date passes)

## Troubleshooting

### Common Issues

#### "Insufficient data" errors
- **Solution**: Add more transactions. Most models require at least 10-30 data points.

#### Low accuracy warnings
- **Solution**: This is normal when you're just starting. Accuracy improves as you add more data.

#### Model retraining takes a long time
- **Solution**: This is normal for larger datasets. Prophet model training can be especially time-consuming.

#### Charts not generating
- **Solution**: Ensure matplotlib is correctly installed. Some environments may require additional dependencies.

#### Error parsing dates
- **Solution**: Ensure dates are in YYYY-MM-DD format.

### Recovery Steps

If the application encounters issues:

1. Check that all required packages are installed
2. Verify that the CSV files are not corrupted
3. If necessary, back up your transaction data and restart with clean files
4. Force retrain models (option 9) to ensure models are up-to-date

## FAQs

**Q: How much data do I need before I can get good forecasts?**
A: For basic forecasting, at least 10-15 transactions spread over different days. For more accurate predictions, aim for 1-2 months of regular data entry.

**Q: How are forecast accuracies calculated?**
A: The application uses a modified Mean Absolute Percentage Error (MAPE) formula: `Accuracy = 100 * (1 - min(1, abs(actual - predicted) / actual))`. This gives a percentage from 0-100%, where 100% is perfect accuracy.

**Q: Can I edit or delete transactions?**
A: The current version doesn't support direct editing or deletion through the interface. For advanced users, you can manually edit the transactions.csv file when the application is not running.

**Q: What does "Force Retrain Models" do?**
A: It rebuilds all prediction models from scratch using your current transaction data. This is useful if you've added a lot of transactions or if the models are behaving unexpectedly.

**Q: How far ahead can the application predict?**
A: While the standard forecast is 30 days, the timeframe predictions can forecast up to 1 year ahead. However, longer predictions naturally become less accurate.

**Q: What's the difference between the forecasting models?**
A: ARIMA is good for short-term predictions with clear patterns, Prophet handles seasonality better, and category-based ML gives you insights into specific spending categories.