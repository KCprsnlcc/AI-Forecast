import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

# File paths
CSV_FILE = "transactions.csv"
FORECASTS_FILE = "forecasts.csv"
EXPORTS_DIR = "exports"
MODELS_DIR = "models"

# Ensure directories exist
for directory in [MODELS_DIR, EXPORTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Model file paths
ARIMA_MODEL_PATH = os.path.join(MODELS_DIR, "arima_model.pkl")
PROPHET_MODEL_PATH = os.path.join(MODELS_DIR, "prophet_model.pkl")
ML_MODELS_PATH = os.path.join(MODELS_DIR, "ml_models.pkl")

# --------------------- Data Handling --------------------- #

def load_data():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["date", "type", "amount", "category"])
    return pd.read_csv(CSV_FILE, parse_dates=["date"])

def save_data(df):
    df.to_csv(CSV_FILE, index=False)

def load_forecasts():
    if not os.path.exists(FORECASTS_FILE):
        return pd.DataFrame(columns=["date_generated", "forecast_date", "model_type", 
                                    "category", "predicted_amount", "actual_amount", "accuracy"])
    try:
        return pd.read_csv(FORECASTS_FILE, parse_dates=["date_generated", "forecast_date"])
    except pd.errors.EmptyDataError:
        print(f"{Fore.YELLOW}Warning: Forecasts file exists but is empty. Creating new dataframe.{Style.RESET_ALL}")
        return pd.DataFrame(columns=["date_generated", "forecast_date", "model_type", 
                                   "category", "predicted_amount", "actual_amount", "accuracy"])
def save_forecasts(df):
    df.to_csv(FORECASTS_FILE, index=False)

def export_predictions(timeframe="3 Months"):
    """Export predictions to CSV file with timestamp"""
    df = load_data()
    forecasts_df = load_forecasts()
    
    if df.empty or forecasts_df.empty:
        print(f"{Fore.RED}No data available for export.{Style.RESET_ALL}")
        return
    
    # Calculate time range based on selected timeframe
    today = datetime.today()
    if timeframe == "3 Months":
        end_date = today + timedelta(days=90)
    elif timeframe == "6 Months":
        end_date = today + timedelta(days=180)
    elif timeframe == "1 Year":
        end_date = today + timedelta(days=365)
    else:
        end_date = today + timedelta(days=90)  # Default to 3 months
    
    # Filter forecasts for the selected timeframe
    filtered_forecasts = forecasts_df[
        (forecasts_df['forecast_date'] >= today) & 
        (forecasts_df['forecast_date'] <= end_date)
    ]
    
    if filtered_forecasts.empty:
        print(f"{Fore.RED}No predictions available for the selected timeframe.{Style.RESET_ALL}")
        return
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(EXPORTS_DIR, f"financial_predictions_{timeframe.replace(' ', '_')}_{timestamp}.csv")
    
    # Export to CSV
    filtered_forecasts.to_csv(filename, index=False)
    print(f"{Fore.GREEN}✅ Predictions exported to {filename}{Style.RESET_ALL}")

# --------------------- UI --------------------- #

def add_transaction():
    t_type = input(f"{Fore.CYAN}Enter type (income/expense): {Style.RESET_ALL}").lower()
    
    # Validate transaction type
    while t_type not in ['income', 'expense']:
        print(f"{Fore.RED}Invalid type. Please enter 'income' or 'expense'.{Style.RESET_ALL}")
        t_type = input(f"{Fore.CYAN}Enter type (income/expense): {Style.RESET_ALL}").lower()
    
    # Get and validate amount
    amount_input = input(f"{Fore.CYAN}Enter amount: {Style.RESET_ALL}")
    try:
        amount = float(amount_input)
        if amount <= 0:
            raise ValueError("Amount must be positive")
    except ValueError:
        print(f"{Fore.RED}Invalid amount. Please enter a positive number.{Style.RESET_ALL}")
        return
    
    category = input(f"{Fore.CYAN}Enter category (e.g., Housing, Food): {Style.RESET_ALL}")
    date_input = input(f"{Fore.CYAN}Enter date (YYYY-MM-DD) [optional]: {Style.RESET_ALL}") or str(datetime.today().date())
    
    # Validate date format
    try:
        date = pd.to_datetime(date_input)
    except:
        print(f"{Fore.RED}Invalid date format. Please use YYYY-MM-DD.{Style.RESET_ALL}")
        return

    df = load_data()
    new_row = {"date": date, "type": t_type, "amount": amount, "category": category}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df)
    print(f"{Fore.GREEN}✅ Transaction saved.{Style.RESET_ALL}")
    
    # Update accuracy of past predictions now that we have new actual data
    update_forecast_accuracy()

def view_transactions(limit=20):
    """Display the most recent transactions with colored output"""
    df = load_data()
    if df.empty:
        print(f"{Fore.YELLOW}No transactions yet.{Style.RESET_ALL}")
        return
    
    # Sort by date (newest first) and limit the number of rows
    df = df.sort_values('date', ascending=False).head(limit)
    
    # Format the DataFrame for display
    formatted_df = df.copy()
    formatted_df['type'] = formatted_df['type'].apply(
        lambda x: f"{Fore.GREEN}{x}{Style.RESET_ALL}" if x == 'income' else f"{Fore.RED}{x}{Style.RESET_ALL}"
    )
    formatted_df['amount'] = formatted_df['amount'].apply(
        lambda x: f"${x:.2f}"
    )
    
    print(f"\n{Fore.CYAN}=== Recent Transactions ({min(limit, len(df))}) ==={Style.RESET_ALL}")
    print(tabulate(formatted_df, headers="keys", tablefmt="grid", showindex=False))
    
    # Show summary statistics
    income_total = df[df['type'] == 'income']['amount'].sum()
    expense_total = df[df['type'] == 'expense']['amount'].sum()
    balance = income_total - expense_total
    
    print(f"\n{Fore.CYAN}=== Summary ==={Style.RESET_ALL}")
    print(f"Total Income: {Fore.GREEN}${income_total:.2f}{Style.RESET_ALL}")
    print(f"Total Expenses: {Fore.RED}${expense_total:.2f}{Style.RESET_ALL}")
    
    if balance >= 0:
        print(f"Balance: {Fore.GREEN}${balance:.2f}{Style.RESET_ALL}")
    else:
        print(f"Balance: {Fore.RED}${balance:.2f}{Style.RESET_ALL}")

def view_forecasts(limit=30):
    """Display forecasts with improved formatting"""
    df = load_forecasts()
    if df.empty:
        print(f"{Fore.YELLOW}No forecasts yet.{Style.RESET_ALL}")
        return
    
    # Get the most recent forecasts
    latest_date = df['date_generated'].max()
    latest_forecasts = df[df['date_generated'] == latest_date].sort_values('forecast_date')
    
    # Limit to first 'limit' rows
    latest_forecasts = latest_forecasts.head(limit)
    
    # Format for display
    formatted_df = latest_forecasts.copy()
    formatted_df['predicted_amount'] = formatted_df['predicted_amount'].apply(lambda x: f"${x:.2f}")
    formatted_df['actual_amount'] = formatted_df['actual_amount'].apply(
        lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A"
    )
    formatted_df['accuracy'] = formatted_df['accuracy'].apply(
        lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A"
    )
    
    print(f"\n{Fore.CYAN}=== Latest Forecasts ==={Style.RESET_ALL}")
    print(f"Generated on: {latest_date}")
    print(tabulate(formatted_df[['forecast_date', 'model_type', 'category', 'predicted_amount', 'actual_amount', 'accuracy']], 
                   headers=['Date', 'Model', 'Category', 'Predicted', 'Actual', 'Accuracy'], 
                   tablefmt="grid", showindex=False))

def view_accuracy(timeframe=None):
    """Display model accuracy with optional timeframe filter"""
    df = load_forecasts()
    if df.empty or df['accuracy'].isna().all():
        print(f"{Fore.YELLOW}No accuracy data available yet.{Style.RESET_ALL}")
        return
    
    # Filter for records with accuracy data
    accuracy_df = df[df['accuracy'].notna()]
    
    # Apply timeframe filter if specified
    if timeframe:
        today = datetime.today()
        if timeframe == "3 Months":
            cutoff_date = today - timedelta(days=90)
        elif timeframe == "6 Months":
            cutoff_date = today - timedelta(days=180)
        elif timeframe == "1 Year":
            cutoff_date = today - timedelta(days=365)
        else:
            cutoff_date = today - timedelta(days=90)  # Default
        
        accuracy_df = accuracy_df[accuracy_df['forecast_date'] >= cutoff_date]
    
    if accuracy_df.empty:
        print(f"{Fore.YELLOW}No accuracy data available for the selected timeframe.{Style.RESET_ALL}")
        return
    
    # Calculate average accuracy by model type
    by_model = accuracy_df.groupby('model_type')['accuracy'].mean().reset_index()
    by_model.columns = ['Model Type', 'Average Accuracy (%)']
    
    # For category-based models, also show by category
    cat_models = accuracy_df[accuracy_df['model_type'] == 'category_ml']
    
    print(f"\n{Fore.CYAN}=== Model Accuracy {f'({timeframe})' if timeframe else ''} ==={Style.RESET_ALL}")
    print(tabulate(by_model, headers="keys", tablefmt="grid", showindex=False))
    
    if not cat_models.empty:
        by_category = cat_models.groupby('category')['accuracy'].mean().reset_index()
        by_category.columns = ['Category', 'Average Accuracy (%)']
        
        print(f"\n{Fore.CYAN}=== Category ML Accuracy ==={Style.RESET_ALL}")
        print(tabulate(by_category, headers="keys", tablefmt="grid", showindex=False))
    
    # Plot accuracy chart if matplotlib is available
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(by_model['Model Type'], by_model['Average Accuracy (%)'], color='skyblue')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(EXPORTS_DIR, "accuracy_chart.png")
        plt.savefig(chart_path)
        plt.close()
        
        print(f"\n{Fore.GREEN}✅ Accuracy chart saved to {chart_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Could not generate chart: {e}{Style.RESET_ALL}")

# --------------------- Model Management --------------------- #

def check_model_freshness(df):
    """Check if models need retraining based on new data"""
    if not os.path.exists(ARIMA_MODEL_PATH):
        return True  # No models exist yet
    
    # Get the last transaction date from when models were trained
    try:
        model_metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
        last_train_date = model_metadata.get("last_train_date")
        last_data_point = model_metadata.get("last_data_point")
        
        # If there's new data since last training or it's been more than 7 days, retrain
        latest_data = df['date'].max()
        if last_data_point is None or latest_data > last_data_point or \
           (datetime.now() - last_train_date).days > 7:
            return True
    except:
        return True  # If metadata doesn't exist or is corrupt, retrain
    
    return False

def save_model_metadata(df):
    """Save metadata about when models were trained and on what data"""
    metadata = {
        "last_train_date": datetime.now(),
        "last_data_point": df['date'].max() if not df.empty else None,
        "transaction_count": len(df)
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))

# --------------------- ARIMA --------------------- #

def train_arima_model(df):
    """Train and save ARIMA model"""
    df = df[df['type'] == 'expense'].copy()
    if df.empty or len(df) < 30:
        return None, "Insufficient expense data for ARIMA."
    
    # Prepare daily data
    daily_expenses = df.groupby('date')['amount'].sum().resample('D').sum().fillna(0)
    
    # Train model
    model = ARIMA(daily_expenses, order=(1, 1, 1))
    fit = model.fit()
    
    # Save model
    joblib.dump(fit, ARIMA_MODEL_PATH)
    
    return fit, None

def forecast_arima(df, force_retrain=False):
    error_msg = None
    
    # Check if model exists and if we need to retrain
    if force_retrain or check_model_freshness(df):
        model, error_msg = train_arima_model(df)
    else:
        try:
            model = joblib.load(ARIMA_MODEL_PATH)
        except:
            model, error_msg = train_arima_model(df)
    
    if error_msg:
        return error_msg, None
    
    # Generate forecast for next 30 days
    forecast = model.forecast(steps=30)
    total_forecast = forecast.sum()
    
    # Log predictions to forecast history
    log_forecast_predictions('arima', forecast, None)
    
    return f"\n{Fore.CYAN}💹 ARIMA Forecast (Next 30 Days Total): {Fore.GREEN}${total_forecast:.2f}{Style.RESET_ALL}", forecast

# --------------------- Prophet --------------------- #

def train_prophet_model(df):
    """Train and save Prophet model"""
    df = df[df['type'] == 'expense'].copy()
    if df.empty or len(df) < 10:
        return None, "Insufficient data for Prophet."
    
    # Prepare data for Prophet
    prophet_df = df.groupby('date')['amount'].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Train model
    model = Prophet()
    model.fit(prophet_df)
    
    # Save model
    with open(PROPHET_MODEL_PATH, 'wb') as f:
        joblib.dump(model, f)
    
    return model, None

def forecast_prophet(df, force_retrain=False):
    error_msg = None
    
    # Check if model exists and if we need to retrain
    if force_retrain or check_model_freshness(df):
        model, error_msg = train_prophet_model(df)
    else:
        try:
            with open(PROPHET_MODEL_PATH, 'rb') as f:
                model = joblib.load(f)
        except:
            model, error_msg = train_prophet_model(df)
    
    if error_msg:
        return error_msg, None
    
    # Generate forecast
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast_tail = forecast.tail(30)
    total_forecast = forecast_tail['yhat'].sum()
    
    # Log predictions to forecast history
    log_forecast_predictions('prophet', None, forecast_tail)
    
    return f"\n{Fore.CYAN}🔮 Prophet Forecast (Next 30 Days Total): {Fore.GREEN}${total_forecast:.2f}{Style.RESET_ALL}", forecast_tail

# --------------------- Category-Based ML --------------------- #

def train_category_ml_models(df):
    """Train and save category-based ML models"""
    df = df[df['type'] == 'expense'].copy()
    if df.empty or len(df) < 10:
        return None, "Insufficient data for category ML."
    
    df['day'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    models = {}
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        if len(cat_df) < 5:
            continue
        
        X = cat_df[['day', 'month', 'day_of_month', 'day_of_week']]
        y = cat_df['amount']
        model = LinearRegression().fit(X, y)
        models[cat] = model
    
    if not models:
        return None, "Not enough category data to make ML predictions."
    
    # Save models dictionary
    joblib.dump(models, ML_MODELS_PATH)
    
    return models, None

def forecast_category_ml(df, force_retrain=False):
    error_msg = None
    
    # Check if model exists and if we need to retrain
    if force_retrain or check_model_freshness(df):
        models, error_msg = train_category_ml_models(df)
    else:
        try:
            models = joblib.load(ML_MODELS_PATH)
        except:
            models, error_msg = train_category_ml_models(df)
    
    if error_msg:
        return error_msg, None
    
    # Generate predictions for each category
    results = []
    today = datetime.today()
    
    # Create 30 days of predictions
    forecasts = {}
    for day_offset in range(1, 31):
        future_date = today + timedelta(days=day_offset)
        future_day = future_date.timetuple().tm_yday
        future_month = future_date.month
        future_day_of_month = future_date.day
        future_day_of_week = future_date.weekday()
        
        for cat, model in models.items():
            if cat not in forecasts:
                forecasts[cat] = []
                
            future_X = pd.DataFrame({
                'day': [future_day],
                'month': [future_month],
                'day_of_month': [future_day_of_month],
                'day_of_week': [future_day_of_week]
            })
            pred = model.predict(future_X)[0]
            
            # Ensure prediction is not negative
            pred = max(0, pred)
            
            forecasts[cat].append((future_date, pred))
    
    # Format results for display
    for cat, preds in forecasts.items():
        total_pred = sum(p[1] for p in preds)
        results.append((cat, total_pred))
    
    # Log predictions to forecast history
    log_category_forecast_predictions(forecasts)
    
    return f"\n{Fore.CYAN}🧠 Category-Based ML Predictions (30 days ahead):{Style.RESET_ALL}\n" + "\n".join(
        [f"{cat}: {Fore.GREEN}${pred:.2f}{Style.RESET_ALL}" for cat, pred in results]
    ), forecasts

# --------------------- Forecast Logging --------------------- #

def log_forecast_predictions(model_type, arima_forecast=None, prophet_forecast=None):
    """Log forecast predictions to the forecasts CSV file"""
    forecasts_df = load_forecasts()
    today = datetime.today()
    new_rows = []
    
    if model_type == 'arima' and arima_forecast is not None:
        for i, pred_value in enumerate(arima_forecast):
            forecast_date = today + timedelta(days=i+1)
            new_row = {
                "date_generated": today,
                "forecast_date": forecast_date,
                "model_type": 'arima',
                "category": 'all',  # ARIMA predicts total expenses
                "predicted_amount": pred_value,
                "actual_amount": np.nan,  # Will be filled later
                "accuracy": np.nan  # Will be calculated later
            }
            new_rows.append(new_row)
    
    if model_type == 'prophet' and prophet_forecast is not None:
        for i, row in enumerate(prophet_forecast.itertuples()):
            forecast_date = row.ds
            new_row = {
                "date_generated": today,
                "forecast_date": forecast_date,
                "model_type": 'prophet',
                "category": 'all',  # Prophet predicts total expenses
                "predicted_amount": row.yhat,
                "actual_amount": np.nan,  # Will be filled later
                "accuracy": np.nan  # Will be calculated later
            }
            new_rows.append(new_row)
    
    # Add new forecasts to the dataframe
    if new_rows:
        forecasts_df = pd.concat([forecasts_df, pd.DataFrame(new_rows)], ignore_index=True)
        save_forecasts(forecasts_df)

def log_category_forecast_predictions(category_forecasts):
    """Log category-based ML forecast predictions"""
    forecasts_df = load_forecasts()
    today = datetime.today()
    new_rows = []
    
    for category, predictions in category_forecasts.items():
        for forecast_date, pred_value in predictions:
            new_row = {
                "date_generated": today,
                "forecast_date": forecast_date,
                "model_type": 'category_ml',
                "category": category,
                "predicted_amount": pred_value,
                "actual_amount": np.nan,  # Will be filled later
                "accuracy": np.nan  # Will be calculated later
            }
            new_rows.append(new_row)
    
    # Add new forecasts to the dataframe
    if new_rows:
        forecasts_df = pd.concat([forecasts_df, pd.DataFrame(new_rows)], ignore_index=True)
        save_forecasts(forecasts_df)

def update_forecast_accuracy():
    """Update forecast accuracy by comparing predictions with actual values"""
    transactions_df = load_data()
    forecasts_df = load_forecasts()
    
    if transactions_df.empty or forecasts_df.empty:
        return
    
    # Only look at transactions up to yesterday (to ensure full day data)
    yesterday = datetime.today() - timedelta(days=1)
    
    # Calculate daily totals for all expenses
    daily_totals = transactions_df[transactions_df['type'] == 'expense'].copy()
    daily_totals = daily_totals.groupby(['date', 'category'])['amount'].sum().reset_index()
    
    # Calculate daily totals without category
    all_daily_totals = daily_totals.groupby('date')['amount'].sum().reset_index()
    
    # Update each forecast with actual values
    updated = False
    
    for i, row in forecasts_df.iterrows():
        if not pd.isna(row['actual_amount']):
            continue  # Already updated
            
        forecast_date = row['forecast_date'].date()
        
        # Check if this date has passed and we have data for it
        if forecast_date <= yesterday.date():
            if row['category'] == 'all':
                # Find matching date in all_daily_totals
                matching = all_daily_totals[all_daily_totals['date'].dt.date == forecast_date]
                if not matching.empty:
                    actual_amount = matching['amount'].values[0]
                    forecasts_df.at[i, 'actual_amount'] = actual_amount
                    
                    # Calculate accuracy
                    predicted = row['predicted_amount']
                    if predicted > 0 and actual_amount > 0:
                        # Use MAPE formula: 100 * (1 - abs(actual - predicted) / actual)
                        accuracy = 100 * (1 - min(1, abs(actual_amount - predicted) / actual_amount))
                        forecasts_df.at[i, 'accuracy'] = max(0, accuracy)  # Ensure non-negative
                        updated = True
            else:
                # Find matching date and category
                matching = daily_totals[(daily_totals['date'].dt.date == forecast_date) & 
                                       (daily_totals['category'] == row['category'])]
                if not matching.empty:
                    actual_amount = matching['amount'].values[0]
                    forecasts_df.at[i, 'actual_amount'] = actual_amount
                    
                    # Calculate accuracy
                    predicted = row['predicted_amount']
                    if predicted > 0 and actual_amount > 0:
                        # Use MAPE formula: 100 * (1 - abs(actual - predicted) / actual)
                        accuracy = 100 * (1 - min(1, abs(actual_amount - predicted) / actual_amount))
                        forecasts_df.at[i, 'accuracy'] = max(0, accuracy)  # Ensure non-negative
                        updated = True
    
    if updated:
        save_forecasts(forecasts_df)

# --------------------- AI Insights --------------------- #

def generate_ai_insights(df):
    """Generate AI insights based on historical data and predictions"""
    if df.empty:
        return []
    
    insights = []
    
    # Calculate savings trend
    if 'income' in df['type'].values and 'expense' in df['type'].values:
        # Group by month
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby(['month', 'type'])['amount'].sum().reset_index()
        
        # Calculate monthly savings
        monthly_pivot = monthly.pivot(index='month', columns='type', values='amount').fillna(0)
        if 'income' in monthly_pivot.columns and 'expense' in monthly_pivot.columns:
            monthly_pivot['savings'] = monthly_pivot['income'] - monthly_pivot['expense']
            
            # Calculate savings trend
            if len(monthly_pivot) >= 3:
                recent_savings = monthly_pivot['savings'].tail(3)
                if recent_savings.iloc[-1] > recent_savings.iloc[-2]:
                    savings_trend = "increasing"
                    percent_change = ((recent_savings.iloc[-1] - recent_savings.iloc[-2]) / recent_savings.iloc[-2] * 100) \
                                    if recent_savings.iloc[-2] > 0 else 0
                    insights.append(
                        f"Your savings are {savings_trend}, up by {percent_change:.1f}% compared to last month."
                    )
                elif recent_savings.iloc[-1] < recent_savings.iloc[-2]:
                    savings_trend = "decreasing"
                    percent_change = ((recent_savings.iloc[-2] - recent_savings.iloc[-1]) / recent_savings.iloc[-2] * 100) \
                                    if recent_savings.iloc[-2] > 0 else 0
                    insights.append(
                        f"Your savings are {savings_trend}, down by {percent_change:.1f}% compared to last month."
                    )
    
    # Identify highest expense category
    expenses = df[df['type'] == 'expense']
    if not expenses.empty:
        by_category = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
        if not by_category.empty:
            top_category = by_category.index[0]
            top_amount = by_category.iloc[0]
            insights.append(
                f"Your highest expense category is {top_category} (${top_amount:.2f} total)."
            )
    
    # Check for unusual spending patterns
    if not expenses.empty and len(expenses) > 10:
        recent = expenses[expenses['date'] >= expenses['date'].max() - timedelta(days=7)]
        if not recent.empty:
            avg_weekly = expenses['amount'].sum() / (len(expenses['date'].unique()) / 7)
            recent_total = recent['amount'].sum()
            
            if recent_total > avg_weekly * 1.5:
                insights.append(
                    f"Your recent spending (${recent_total:.2f}) is {recent_total/avg_weekly:.1f}x higher than your weekly average."
                )
            elif recent_total < avg_weekly * 0.5 and len(recent) >= 3:
                insights.append(
                    f"Your recent spending (${recent_total:.2f}) is {avg_weekly/recent_total:.1f}x lower than your weekly average. Great job!"
                )
    
    # Recommend emergency fund if not enough savings
    if 'income' in df['type'].values:
        monthly_income = df[df['type'] == 'income']['amount'].sum() / (len(df['date'].unique()) / 30)
        recommended_emergency = monthly_income * 3  # 3 months of expenses
        
        total_savings = 0
        if 'expense' in df['type'].values:
            total_income = df[df['type'] == 'income']['amount'].sum()
            total_expenses = df[df['type'] == 'expense']['amount'].sum()
            total_savings = total_income - total_expenses
        
        if total_savings < recommended_emergency:
            amount_needed = recommended_emergency - total_savings
            monthly_contribution = amount_needed / 6  # Achieve in 6 months
            insights.append(
                f"Consider adding ${monthly_contribution:.2f}/month to your emergency fund to reach the recommended " +
                f"3-month safety net (${recommended_emergency:.2f})."
            )
    
    return insights

def generate_ai_forecasts(force_retrain=False, display_insights=True):
    """Generate forecasts using all three models and show insights"""
    df = load_data()
    if df.empty:
        print(f"{Fore.YELLOW}No data to analyze.{Style.RESET_ALL}")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Generate forecasts using all three models
    arima_result, arima_forecast = forecast_arima(df, force_retrain)
    prophet_result, prophet_forecast = forecast_prophet(df, force_retrain)
    category_result, category_forecast = forecast_category_ml(df, force_retrain)
    
    # If we trained models, save the metadata
    if force_retrain:
        save_model_metadata(df)
    
    print(arima_result)
    print(prophet_result)
    print(category_result)
    
    # Generate and display insights
    if display_insights:
        insights = generate_ai_insights(df)
        if insights:
            print(f"\n{Fore.CYAN}=== AI Insights ==={Style.RESET_ALL}")
            for i, insight in enumerate(insights, 1):
                print(f"{i}. {insight}")
        else:
            print(f"\n{Fore.YELLOW}Not enough data for AI insights yet.{Style.RESET_ALL}")

    # Return forecasts for potential further processing
    return {
        'arima': arima_forecast,
        'prophet': prophet_forecast,
        'category': category_forecast
    }

def plot_forecasts(forecasts=None):
    """Plot forecasts using matplotlib if available"""
    if forecasts is None:
        # Generate forecasts if not provided
        forecasts = generate_ai_forecasts(display_insights=False)
    
    if forecasts is None:
        print(f"{Fore.YELLOW}No forecast data available to plot.{Style.RESET_ALL}")
        return
    
    try:
        # Create a new figure
        plt.figure(figsize=(12, 6))
        
        # Plot ARIMA forecast if available
        if 'arima' in forecasts and forecasts['arima'] is not None:
            dates = [datetime.today() + timedelta(days=i+1) for i in range(len(forecasts['arima']))]
            plt.plot(dates, forecasts['arima'], label='ARIMA', marker='o', linestyle='-', alpha=0.7)
        
        # Plot Prophet forecast if available
        if 'prophet' in forecasts and forecasts['prophet'] is not None:
            plt.plot(forecasts['prophet']['ds'], forecasts['prophet']['yhat'], 
                     label='Prophet', marker='x', linestyle='-', alpha=0.7)
        
        # Format the plot
        plt.title('Expense Forecasts Comparison')
        plt.xlabel('Date')
        plt.ylabel('Amount ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        chart_path = os.path.join(EXPORTS_DIR, "forecast_comparison.png")
        plt.savefig(chart_path)
        plt.close()
        
        print(f"\n{Fore.GREEN}✅ Forecast comparison chart saved to {chart_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Could not generate forecast chart: {e}{Style.RESET_ALL}")

# --------------------- Timeframe-Based Analysis --------------------- #

def generate_timeframe_prediction(timeframe="3 Months"):
    """Generate predictions for specific timeframe"""
    df = load_data()
    if df.empty:
        print(f"{Fore.YELLOW}No data to analyze.{Style.RESET_ALL}")
        return
    
    # Calculate steps based on timeframe
    if timeframe == "3 Months":
        steps = 90
    elif timeframe == "6 Months":
        steps = 180
    elif timeframe == "1 Year":
        steps = 365
    else:
        steps = 90  # Default
    
    # Generate longer-term forecast
    try:
        # Use Prophet for longer time horizons
        df_expense = df[df['type'] == 'expense'].copy()
        if df_expense.empty or len(df_expense) < 10:
            print(f"{Fore.YELLOW}Insufficient data for {timeframe} prediction.{Style.RESET_ALL}")
            return
        
        # Prepare data for Prophet
        prophet_df = df_expense.groupby('date')['amount'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Train model
        model = Prophet()
        model.fit(prophet_df)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        forecast_tail = forecast.tail(steps)
        total_forecast = forecast_tail['yhat'].sum()
        
        # Income forecast (if available)
        income_forecast = None
        income_total = None
        df_income = df[df['type'] == 'income'].copy()
        if not df_income.empty and len(df_income) >= 10:
            # Prepare income data for Prophet
            income_prophet_df = df_income.groupby('date')['amount'].sum().reset_index()
            income_prophet_df.columns = ['ds', 'y']
            
            # Train income model
            income_model = Prophet()
            income_model.fit(income_prophet_df)
            
            # Generate income forecast
            income_future = income_model.make_future_dataframe(periods=steps)
            income_forecast = income_model.predict(income_future)
            income_forecast_tail = income_forecast.tail(steps)
            income_total = income_forecast_tail['yhat'].sum()
        
        # Calculate savings if both forecasts are available
        savings_total = None
        if income_total is not None:
            savings_total = income_total - total_forecast
        
        # Display results
        print(f"\n{Fore.CYAN}=== {timeframe} Financial Prediction ==={Style.RESET_ALL}")
        print(f"Total Projected Expenses: {Fore.RED}${total_forecast:.2f}{Style.RESET_ALL}")
        
        if income_total is not None:
            print(f"Total Projected Income: {Fore.GREEN}${income_total:.2f}{Style.RESET_ALL}")
            
        if savings_total is not None:
            if savings_total >= 0:
                print(f"Projected Savings: {Fore.GREEN}${savings_total:.2f}{Style.RESET_ALL}")
            else:
                print(f"Projected Deficit: {Fore.RED}${abs(savings_total):.2f}{Style.RESET_ALL}")
        
        # Try to plot the forecast
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot expense forecast
            plt.plot(forecast_tail['ds'], forecast_tail['yhat'], 
                     label='Expenses', color='red', marker='', linestyle='-')
            plt.fill_between(forecast_tail['ds'], 
                            forecast_tail['yhat_lower'], 
                            forecast_tail['yhat_upper'], 
                            color='red', alpha=0.2)
            
            # Plot income forecast if available
            if income_forecast is not None:
                income_tail = income_forecast.tail(steps)
                plt.plot(income_tail['ds'], income_tail['yhat'], 
                        label='Income', color='green', marker='', linestyle='-')
                plt.fill_between(income_tail['ds'], 
                                income_tail['yhat_lower'], 
                                income_tail['yhat_upper'], 
                                color='green', alpha=0.2)
            
            # Format plot
            plt.title(f'{timeframe} Financial Forecast')
            plt.xlabel('Date')
            plt.ylabel('Amount ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(EXPORTS_DIR, f"{timeframe.replace(' ', '_')}_forecast.png")
            plt.savefig(chart_path)
            plt.close()
            
            print(f"\n{Fore.GREEN}✅ {timeframe} forecast chart saved to {chart_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not generate chart: {e}{Style.RESET_ALL}")
        
        return forecast_tail
        
    except Exception as e:
        print(f"{Fore.RED}Error generating {timeframe} prediction: {e}{Style.RESET_ALL}")
        return None

# --------------------- Data Type Analysis --------------------- #

def analyze_data_type(data_type="All"):
    """Analyze specific data type (All/Income/Expenses/Savings)"""
    df = load_data()
    if df.empty:
        print(f"{Fore.YELLOW}No data to analyze.{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.CYAN}=== {data_type} Analysis ==={Style.RESET_ALL}")
    
    if data_type == "All":
        # Show summary of all data
        income_df = df[df['type'] == 'income']
        expense_df = df[df['type'] == 'expense']
        
        income_total = income_df['amount'].sum() if not income_df.empty else 0
        expense_total = expense_df['amount'].sum() if not expense_df.empty else 0
        savings = income_total - expense_total
        
        print(f"Total Income: {Fore.GREEN}${income_total:.2f}{Style.RESET_ALL}")
        print(f"Total Expenses: {Fore.RED}${expense_total:.2f}{Style.RESET_ALL}")
        
        if savings >= 0:
            print(f"Net Savings: {Fore.GREEN}${savings:.2f}{Style.RESET_ALL}")
        else:
            print(f"Net Deficit: {Fore.RED}${abs(savings):.2f}{Style.RESET_ALL}")
        
        # Calculate monthly averages if enough data
        if len(df['date'].unique()) >= 30:
            months = (df['date'].max() - df['date'].min()).days / 30
            if months >= 1:
                monthly_income = income_total / months
                monthly_expenses = expense_total / months
                monthly_savings = monthly_income - monthly_expenses
                
                print(f"\nMonthly Averages:")
                print(f"Income: {Fore.GREEN}${monthly_income:.2f}{Style.RESET_ALL}")
                print(f"Expenses: {Fore.RED}${monthly_expenses:.2f}{Style.RESET_ALL}")
                
                if monthly_savings >= 0:
                    print(f"Savings: {Fore.GREEN}${monthly_savings:.2f}{Style.RESET_ALL}")
                else:
                    print(f"Deficit: {Fore.RED}${abs(monthly_savings):.2f}{Style.RESET_ALL}")
    
    elif data_type == "Income":
        # Focus on income analysis
        income_df = df[df['type'] == 'income']
        if income_df.empty:
            print(f"{Fore.YELLOW}No income data available.{Style.RESET_ALL}")
            return
        
        total = income_df['amount'].sum()
        avg = income_df['amount'].mean()
        max_amount = income_df['amount'].max()
        min_amount = income_df['amount'].min()
        
        print(f"Total Income: {Fore.GREEN}${total:.2f}{Style.RESET_ALL}")
        print(f"Average Income Entry: ${avg:.2f}")
        print(f"Largest Income: ${max_amount:.2f}")
        print(f"Smallest Income: ${min_amount:.2f}")
        
        # Show income by category if categories exist
        if 'category' in income_df.columns and not income_df['category'].isna().all():
            by_category = income_df.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            print(f"\nIncome by Category:")
            for cat, amount in by_category.items():
                print(f"{cat}: {Fore.GREEN}${amount:.2f}{Style.RESET_ALL}")
    
    elif data_type == "Expenses":
        # Focus on expense analysis
        expense_df = df[df['type'] == 'expense']
        if expense_df.empty:
            print(f"{Fore.YELLOW}No expense data available.{Style.RESET_ALL}")
            return
        
        total = expense_df['amount'].sum()
        avg = expense_df['amount'].mean()
        max_amount = expense_df['amount'].max()
        min_amount = expense_df['amount'].min()
        
        print(f"Total Expenses: {Fore.RED}${total:.2f}{Style.RESET_ALL}")
        print(f"Average Expense: ${avg:.2f}")
        print(f"Largest Expense: ${max_amount:.2f}")
        print(f"Smallest Expense: ${min_amount:.2f}")
        
        # Show expenses by category
        by_category = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        print(f"\nExpenses by Category:")
        for cat, amount in by_category.items():
            percent = (amount / total) * 100
            print(f"{cat}: {Fore.RED}${amount:.2f}{Style.RESET_ALL} ({percent:.1f}%)")
        
        # Try to plot expense distribution
        try:
            plt.figure(figsize=(10, 6))
            by_category_df = by_category.reset_index()
            
            # Use only top 6 categories for clarity, group others
            if len(by_category_df) > 6:
                top_cats = by_category_df.head(5)
                other_sum = by_category_df.iloc[5:]['amount'].sum()
                
                # Create final dataframe for plotting
                plot_df = pd.concat([
                    top_cats,
                    pd.DataFrame({'category': ['Other'], 'amount': [other_sum]})
                ])
            else:
                plot_df = by_category_df
            
            # Plot pie chart
            plt.pie(plot_df['amount'], labels=plot_df['category'], autopct='%1.1f%%', 
                   startangle=90, shadow=True)
            plt.axis('equal')
            plt.title('Expense Distribution by Category')
            
            # Save chart
            chart_path = os.path.join(EXPORTS_DIR, "expense_distribution.png")
            plt.savefig(chart_path)
            plt.close()
            
            print(f"\n{Fore.GREEN}✅ Expense distribution chart saved to {chart_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not generate chart: {e}{Style.RESET_ALL}")
    
    elif data_type == "Savings":
        # Focus on savings analysis (income - expenses)
        income_df = df[df['type'] == 'income']
        expense_df = df[df['type'] == 'expense']
        
        if income_df.empty or expense_df.empty:
            print(f"{Fore.YELLOW}Need both income and expense data for savings analysis.{Style.RESET_ALL}")
            return
        
        # Group by month for trend analysis
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby(['month', 'type'])['amount'].sum().reset_index()
        
        # Calculate monthly savings
        monthly_pivot = monthly.pivot(index='month', columns='type', values='amount').fillna(0)
        if 'income' in monthly_pivot.columns and 'expense' in monthly_pivot.columns:
            monthly_pivot['savings'] = monthly_pivot['income'] - monthly_pivot['expense']
            monthly_pivot['savings_rate'] = (monthly_pivot['savings'] / monthly_pivot['income'] * 100).round(1)
            
            # Convert period index back to datetime for plotting
            monthly_pivot = monthly_pivot.reset_index()
            monthly_pivot['month'] = monthly_pivot['month'].dt.to_timestamp()
            
            # Show savings statistics
            total_income = monthly_pivot['income'].sum()
            total_expenses = monthly_pivot['expense'].sum()
            total_savings = monthly_pivot['savings'].sum()
            avg_savings_rate = (total_savings / total_income * 100) if total_income > 0 else 0
            
            print(f"Total Savings: {Fore.GREEN if total_savings >= 0 else Fore.RED}${total_savings:.2f}{Style.RESET_ALL}")
            print(f"Average Monthly Savings: ${monthly_pivot['savings'].mean():.2f}")
            print(f"Overall Savings Rate: {avg_savings_rate:.1f}%")
            
            if len(monthly_pivot) >= 2:
                # Calculate trend
                last_month = monthly_pivot.iloc[-1]
                prev_month = monthly_pivot.iloc[-2]
                
                change = last_month['savings'] - prev_month['savings']
                percent_change = (change / abs(prev_month['savings']) * 100) if prev_month['savings'] != 0 else float('inf')
                
                if change >= 0:
                    print(f"\nSavings Trend: {Fore.GREEN}↑ Increased by ${change:.2f} ({abs(percent_change):.1f}%){Style.RESET_ALL}")
                else:
                    print(f"\nSavings Trend: {Fore.RED}↓ Decreased by ${abs(change):.2f} ({abs(percent_change):.1f}%){Style.RESET_ALL}")
            
            # Plot savings trend
            try:
                plt.figure(figsize=(12, 6))
                
                # Plot monthly savings
                plt.bar(monthly_pivot['month'], monthly_pivot['savings'], 
                       color=[('green' if x >= 0 else 'red') for x in monthly_pivot['savings']])
                
                # Add line for savings rate
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(monthly_pivot['month'], monthly_pivot['savings_rate'], 'b-', linewidth=2)
                ax2.set_ylabel('Savings Rate (%)', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
                
                # Format plot
                ax1.set_title('Monthly Savings and Savings Rate')
                ax1.set_xlabel('Month')
                ax1.set_ylabel('Savings Amount ($)')
                ax1.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save chart
                chart_path = os.path.join(EXPORTS_DIR, "savings_trend.png")
                plt.savefig(chart_path)
                plt.close()
                
                print(f"\n{Fore.GREEN}✅ Savings trend chart saved to {chart_path}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}Could not generate chart: {e}{Style.RESET_ALL}")

# --------------------- Main App --------------------- #

def main():
    while True:
        print(f"\n{Fore.CYAN}====== Enhanced AI Finance Tracker ======{Style.RESET_ALL}")
        print("1. Add Transaction")
        print("2. View Transactions")
        print("3. Generate AI Forecasts")
        print("4. View Forecasts")
        print("5. View Model Accuracy")
        print("6. Data Analysis")
        print("7. Timeframe Predictions")
        print("8. Export Predictions")
        print("9. Force Retrain Models")
        print("0. Exit")

        choice = input(f"\n{Fore.CYAN}Choose an option: {Style.RESET_ALL}")
        
        if choice == '1':
            add_transaction()
        elif choice == '2':
            view_transactions()
        elif choice == '3':
            generate_ai_forecasts()
            plot_forecasts()
        elif choice == '4':
            view_forecasts()
        elif choice == '5':
            timeframe = input(f"{Fore.CYAN}Select timeframe (3 Months/6 Months/1 Year/All): {Style.RESET_ALL}") or "All"
            view_accuracy(timeframe if timeframe != "All" else None)
        elif choice == '6':
            data_type = input(f"{Fore.CYAN}Select data type (All/Income/Expenses/Savings): {Style.RESET_ALL}") or "All"
            analyze_data_type(data_type)
        elif choice == '7':
            timeframe = input(f"{Fore.CYAN}Select timeframe (3 Months/6 Months/1 Year): {Style.RESET_ALL}") or "3 Months"
            generate_timeframe_prediction(timeframe)
        elif choice == '8':
            timeframe = input(f"{Fore.CYAN}Select timeframe to export (3 Months/6 Months/1 Year): {Style.RESET_ALL}") or "3 Months"
            export_predictions(timeframe)
        elif choice == '9':
            print(f"{Fore.YELLOW}Retraining all models...{Style.RESET_ALL}")
            generate_ai_forecasts(force_retrain=True)
            print(f"{Fore.GREEN}✅ Models retrained successfully.{Style.RESET_ALL}")
        elif choice == '0':
            print(f"{Fore.GREEN}Thank you for using AI Finance Tracker. Goodbye!{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()