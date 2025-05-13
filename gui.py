import sys
import os
import pandas as pd
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QPushButton, QTabWidget, QComboBox, QLineEdit, 
                              QDateEdit, QTableWidget, QTableWidgetItem, QMessageBox,
                              QGroupBox, QFormLayout, QSplitter, QFileDialog, QGridLayout)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QFont, QColor, QPixmap
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import functions from main.py
from main import (load_data, save_data, load_forecasts, save_forecasts, 
                 generate_ai_forecasts, generate_timeframe_prediction, 
                 analyze_data_type, update_forecast_accuracy, view_accuracy,
                 EXPORTS_DIR, export_predictions)

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)

class FinanceTrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Finance Tracker")
        self.setMinimumSize(1000, 700)
        
        # Create the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.dashboard_tab = QWidget()
        self.transactions_tab = QWidget()
        self.forecasts_tab = QWidget()
        self.analysis_tab = QWidget()
        
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.transactions_tab, "Transactions")
        self.tabs.addTab(self.forecasts_tab, "Forecasts")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        
        # Initialize tabs
        self.init_dashboard_tab()
        self.init_transactions_tab()
        self.init_forecasts_tab()
        self.init_analysis_tab()
        
        # Load initial data
        self.load_data_and_update_ui()
        
    def load_data_and_update_ui(self):
        """Load data and update UI components"""
        self.transactions_df = load_data()
        self.forecasts_df = load_forecasts()
        self.update_dashboard()
        self.update_transactions_table()
        self.update_forecasts_table()
    
    def init_dashboard_tab(self):
        """Initialize the dashboard tab"""
        layout = QVBoxLayout(self.dashboard_tab)
        
        # Summary section
        summary_group = QGroupBox("Financial Summary")
        summary_layout = QGridLayout()
        summary_group.setLayout(summary_layout)
        
        # Summary labels
        self.total_income_label = QLabel("Total Income: $0.00")
        self.total_expense_label = QLabel("Total Expenses: $0.00")
        self.balance_label = QLabel("Balance: $0.00")
        self.monthly_income_label = QLabel("Monthly Income: $0.00")
        self.monthly_expense_label = QLabel("Monthly Expenses: $0.00")
        self.monthly_savings_label = QLabel("Monthly Savings: $0.00")
        
        # Set font for summary labels
        font = QFont()
        font.setPointSize(12)
        self.total_income_label.setFont(font)
        self.total_expense_label.setFont(font)
        self.balance_label.setFont(font)
        
        # Add labels to summary layout
        summary_layout.addWidget(self.total_income_label, 0, 0)
        summary_layout.addWidget(self.total_expense_label, 0, 1)
        summary_layout.addWidget(self.balance_label, 0, 2)
        summary_layout.addWidget(self.monthly_income_label, 1, 0)
        summary_layout.addWidget(self.monthly_expense_label, 1, 1)
        summary_layout.addWidget(self.monthly_savings_label, 1, 2)
        
        # Quick actions section
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout()
        actions_group.setLayout(actions_layout)
        
        self.add_transaction_btn = QPushButton("Add Transaction")
        self.generate_forecast_btn = QPushButton("Generate Forecasts")
        self.export_btn = QPushButton("Export Data")
        
        actions_layout.addWidget(self.add_transaction_btn)
        actions_layout.addWidget(self.generate_forecast_btn)
        actions_layout.addWidget(self.export_btn)
        
        # Connect buttons
        self.add_transaction_btn.clicked.connect(self.show_add_transaction_dialog)
        self.generate_forecast_btn.clicked.connect(self.generate_forecasts)
        self.export_btn.clicked.connect(self.export_data)
        
        # Charts section
        charts_group = QGroupBox("Financial Charts")
        charts_layout = QHBoxLayout()
        charts_group.setLayout(charts_layout)
        
        self.expense_chart = MatplotlibCanvas(width=5, height=4)
        self.forecast_chart = MatplotlibCanvas(width=5, height=4)
        
        charts_layout.addWidget(self.expense_chart)
        charts_layout.addWidget(self.forecast_chart)
        
        # Add all sections to main layout
        layout.addWidget(summary_group)
        layout.addWidget(actions_group)
        layout.addWidget(charts_group)
    
    def init_transactions_tab(self):
        """Initialize the transactions tab"""
        layout = QVBoxLayout(self.transactions_tab)
        
        # Add transaction form
        form_group = QGroupBox("Add Transaction")
        form_layout = QFormLayout()
        form_group.setLayout(form_layout)
        
        # Transaction type dropdown
        self.transaction_type = QComboBox()
        self.transaction_type.addItems(["income", "expense"])
        form_layout.addRow("Type:", self.transaction_type)
        
        # Amount input
        self.amount_input = QLineEdit()
        form_layout.addRow("Amount ($):", self.amount_input)
        
        # Category input
        self.category_input = QLineEdit()
        form_layout.addRow("Category:", self.category_input)
        
        # Date input
        self.date_input = QDateEdit()
        self.date_input.setDate(QDate.currentDate())
        self.date_input.setCalendarPopup(True)
        form_layout.addRow("Date:", self.date_input)
        
        # Add button
        self.add_btn = QPushButton("Add Transaction")
        self.add_btn.clicked.connect(self.add_transaction)
        form_layout.addRow("", self.add_btn)
        
        # Transactions table
        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(4)
        self.transactions_table.setHorizontalHeaderLabels(["Date", "Type", "Category", "Amount"])
        self.transactions_table.horizontalHeader().setStretchLastSection(True)
        
        # Add components to layout
        layout.addWidget(form_group)
        layout.addWidget(QLabel("Recent Transactions:"))
        layout.addWidget(self.transactions_table)
    
    def init_forecasts_tab(self):
        """Initialize the forecasts tab"""
        layout = QVBoxLayout(self.forecasts_tab)
        
        # Controls section
        controls_layout = QHBoxLayout()
        
        self.forecast_timeframe = QComboBox()
        self.forecast_timeframe.addItems(["3 Months", "6 Months", "1 Year"])
        controls_layout.addWidget(QLabel("Timeframe:"))
        controls_layout.addWidget(self.forecast_timeframe)
        
        self.generate_btn = QPushButton("Generate Forecast")
        self.generate_btn.clicked.connect(self.generate_timeframe_forecast)
        controls_layout.addWidget(self.generate_btn)
        
        self.retrain_btn = QPushButton("Retrain Models")
        self.retrain_btn.clicked.connect(self.retrain_models)
        controls_layout.addWidget(self.retrain_btn)
        
        # Forecasts table
        self.forecasts_table = QTableWidget()
        self.forecasts_table.setColumnCount(6)
        self.forecasts_table.setHorizontalHeaderLabels(
            ["Forecast Date", "Model", "Category", "Predicted Amount", "Actual Amount", "Accuracy"]
        )
        self.forecasts_table.horizontalHeader().setStretchLastSection(True)
        
        # Forecast chart
        self.forecast_canvas = MatplotlibCanvas(width=10, height=6)
        
        # Add components to layout
        layout.addLayout(controls_layout)
        layout.addWidget(QLabel("Forecast Results:"))
        layout.addWidget(self.forecasts_table)
        layout.addWidget(self.forecast_canvas)
    
    def init_analysis_tab(self):
        """Initialize the analysis tab"""
        layout = QVBoxLayout(self.analysis_tab)
        
        # Controls section
        controls_layout = QHBoxLayout()
        
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(["All", "Income", "Expenses", "Savings"])
        controls_layout.addWidget(QLabel("Analysis Type:"))
        controls_layout.addWidget(self.analysis_type)
        
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self.analyze_btn)
        
        # Model accuracy section
        accuracy_group = QGroupBox("Model Accuracy")
        accuracy_layout = QVBoxLayout()
        accuracy_group.setLayout(accuracy_layout)
        
        accuracy_controls = QHBoxLayout()
        self.accuracy_timeframe = QComboBox()
        self.accuracy_timeframe.addItems(["All", "3 Months", "6 Months", "1 Year"])
        accuracy_controls.addWidget(QLabel("Timeframe:"))
        accuracy_controls.addWidget(self.accuracy_timeframe)
        
        self.accuracy_btn = QPushButton("View Accuracy")
        self.accuracy_btn.clicked.connect(self.show_accuracy)
        accuracy_controls.addWidget(self.accuracy_btn)
        
        self.accuracy_table = QTableWidget()
        self.accuracy_table.setColumnCount(2)
        self.accuracy_table.setHorizontalHeaderLabels(["Model Type", "Average Accuracy (%)"])
        
        accuracy_layout.addLayout(accuracy_controls)
        accuracy_layout.addWidget(self.accuracy_table)
        
        # Analysis chart
        self.analysis_canvas = MatplotlibCanvas(width=10, height=6)
        
        # Add components to layout
        layout.addLayout(controls_layout)
        layout.addWidget(accuracy_group)
        layout.addWidget(self.analysis_canvas)
    
    def update_dashboard(self):
        """Update dashboard with current data"""
        if self.transactions_df.empty:
            return
        
        # Calculate summary statistics
        income_df = self.transactions_df[self.transactions_df['type'] == 'income']
        expense_df = self.transactions_df[self.transactions_df['type'] == 'expense']
        
        income_total = income_df['amount'].sum() if not income_df.empty else 0
        expense_total = expense_df['amount'].sum() if not expense_df.empty else 0
        balance = income_total - expense_total
        
        # Update summary labels
        self.total_income_label.setText(f"Total Income: ${income_total:.2f}")
        self.total_expense_label.setText(f"Total Expenses: ${expense_total:.2f}")
        
        if balance >= 0:
            self.balance_label.setText(f"Balance: ${balance:.2f}")
            self.balance_label.setStyleSheet("color: green")
        else:
            self.balance_label.setText(f"Balance: -${abs(balance):.2f}")
            self.balance_label.setStyleSheet("color: red")
        
        # Calculate monthly averages if enough data
        if not self.transactions_df.empty and len(self.transactions_df['date'].unique()) >= 30:
            date_range = (self.transactions_df['date'].max() - self.transactions_df['date'].min()).days
            if date_range > 0:
                months = date_range / 30
                if months >= 1:
                    monthly_income = income_total / months
                    monthly_expenses = expense_total / months
                    monthly_savings = monthly_income - monthly_expenses
                    
                    self.monthly_income_label.setText(f"Monthly Income: ${monthly_income:.2f}")
                    self.monthly_expense_label.setText(f"Monthly Expenses: ${monthly_expenses:.2f}")
                    
                    if monthly_savings >= 0:
                        self.monthly_savings_label.setText(f"Monthly Savings: ${monthly_savings:.2f}")
                        self.monthly_savings_label.setStyleSheet("color: green")
                    else:
                        self.monthly_savings_label.setText(f"Monthly Deficit: -${abs(monthly_savings):.2f}")
                        self.monthly_savings_label.setStyleSheet("color: red")
        
        # Update expense chart
        if not expense_df.empty:
            self.update_expense_chart(expense_df)
        
        # Update forecast chart if forecasts exist
        if not self.forecasts_df.empty:
            self.update_forecast_chart()
    
    def update_expense_chart(self, expense_df):
        """Update the expense distribution chart"""
        # Group by category
        by_category = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Use only top 6 categories for clarity, group others
        if len(by_category) > 6:
            top_cats = by_category.head(5)
            other_sum = by_category.iloc[5:].sum()
            
            # Create final series for plotting
            plot_data = pd.concat([
                top_cats,
                pd.Series({'Other': other_sum})
            ])
        else:
            plot_data = by_category
        
        # Clear previous chart
        self.expense_chart.axes.clear()
        
        # Create pie chart
        self.expense_chart.axes.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', 
                                   startangle=90, shadow=True)
        self.expense_chart.axes.set_title('Expense Distribution')
        self.expense_chart.axes.axis('equal')
        
        # Refresh canvas
        self.expense_chart.draw()
    
    def update_forecast_chart(self):
        """Update the forecast chart"""
        # Get the most recent forecasts
        if self.forecasts_df.empty:
            return
            
        latest_date = self.forecasts_df['date_generated'].max()
        latest_forecasts = self.forecasts_df[self.forecasts_df['date_generated'] == latest_date]
        
        # Group by model type and date
        grouped = latest_forecasts.groupby(['model_type', 'forecast_date'])['predicted_amount'].sum().reset_index()
        
        # Clear previous chart
        self.forecast_chart.axes.clear()
        
        # Plot each model type
        for model_type, data in grouped.groupby('model_type'):
            self.forecast_chart.axes.plot(data['forecast_date'], data['predicted_amount'], 
                                         label=model_type, marker='o')
        
        # Format chart
        self.forecast_chart.axes.set_title('Expense Forecasts')
        self.forecast_chart.axes.set_xlabel('Date')
        self.forecast_chart.axes.set_ylabel('Amount ($)')
        self.forecast_chart.axes.legend()
        self.forecast_chart.axes.grid(True, alpha=0.3)
        for tick in self.forecast_chart.axes.get_xticklabels():
            tick.set_rotation(45)
        
        # Refresh canvas
        self.forecast_chart.draw()
    
    def update_transactions_table(self):
        """Update the transactions table with current data"""
        self.transactions_table.setRowCount(0)
        
        if self.transactions_df.empty:
            return
        
        # Sort by date (newest first)
        df = self.transactions_df.sort_values('date', ascending=False)
        
        # Add rows to table
        for i, row in enumerate(df.itertuples()):
            self.transactions_table.insertRow(i)
            
            # Format date
            date_item = QTableWidgetItem(str(row.date.date()))
            self.transactions_table.setItem(i, 0, date_item)
            
            # Format type with color
            type_item = QTableWidgetItem(row.type)
            if row.type == 'income':
                type_item.setForeground(QColor('green'))
            else:
                type_item.setForeground(QColor('red'))
            self.transactions_table.setItem(i, 1, type_item)
            
            # Category
            category_item = QTableWidgetItem(row.category)
            self.transactions_table.setItem(i, 2, category_item)
            
            # Format amount
            amount_item = QTableWidgetItem(f"${row.amount:.2f}")
            if row.type == 'income':
                amount_item.setForeground(QColor('green'))
            else:
                amount_item.setForeground(QColor('red'))
            self.transactions_table.setItem(i, 3, amount_item)
    
    def update_forecasts_table(self):
        """Update the forecasts table with current data"""
        self.forecasts_table.setRowCount(0)
        
        if self.forecasts_df.empty:
            return
        
        # Get the most recent forecasts
        latest_date = self.forecasts_df['date_generated'].max()
        latest_forecasts = self.forecasts_df[self.forecasts_df['date_generated'] == latest_date].sort_values('forecast_date')
        
        # Add rows to table
        for i, row in enumerate(latest_forecasts.itertuples()):
            self.forecasts_table.insertRow(i)
            
            # Format date
            date_item = QTableWidgetItem(str(row.forecast_date.date()))
            self.forecasts_table.setItem(i, 0, date_item)
            
            # Model type
            model_item = QTableWidgetItem(row.model_type)
            self.forecasts_table.setItem(i, 1, model_item)
            
            # Category
            category_item = QTableWidgetItem(row.category)
            self.forecasts_table.setItem(i, 2, category_item)
            
            # Predicted amount
            predicted_item = QTableWidgetItem(f"${row.predicted_amount:.2f}")
            self.forecasts_table.setItem(i, 3, predicted_item)
            
            # Actual amount (if available)
            if pd.isna(row.actual_amount):
                actual_item = QTableWidgetItem("N/A")
            else:
                actual_item = QTableWidgetItem(f"${row.actual_amount:.2f}")
            self.forecasts_table.setItem(i, 4, actual_item)
            
            # Accuracy (if available)
            if pd.isna(row.accuracy):
                accuracy_item = QTableWidgetItem("N/A")
            else:
                accuracy_item = QTableWidgetItem(f"{row.accuracy:.1f}%")
            self.forecasts_table.setItem(i, 5, accuracy_item)
    
    def show_add_transaction_dialog(self):
        """Show the add transaction form in the transactions tab"""
        self.tabs.setCurrentWidget(self.transactions_tab)
        self.amount_input.setFocus()
    
    def add_transaction(self):
        """Add a new transaction"""
        try:
            # Get values from form
            t_type = self.transaction_type.currentText()
            
            # Validate amount
            try:
                amount = float(self.amount_input.text())
                if amount <= 0:
                    QMessageBox.warning(self, "Invalid Input", "Amount must be positive.")
                    return
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for amount.")
                return
            
            # Get other fields
            category = self.category_input.text()
            if not category:
                QMessageBox.warning(self, "Invalid Input", "Please enter a category.")
                return
                
            date = self.date_input.date().toPython()
            
            # Create new transaction
            df = load_data()
            new_row = {"date": pd.Timestamp(date), "type": t_type, "amount": amount, "category": category}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_data(df)
            
            # Update forecast accuracy
            update_forecast_accuracy()
            
            # Clear form
            self.amount_input.clear()
            self.category_input.clear()
            self.date_input.setDate(QDate.currentDate())
            
            # Reload data and update UI
            self.load_data_and_update_ui()
            
            QMessageBox.information(self, "Success", "Transaction added successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def generate_forecasts(self):
        """Generate AI forecasts"""
        try:
            QMessageBox.information(self, "Generating Forecasts", 
                                  "Generating forecasts. This may take a moment...")
            
            # Generate forecasts
            forecasts = generate_ai_forecasts(display_insights=False)
            
            # Reload data and update UI
            self.load_data_and_update_ui()
            
            QMessageBox.information(self, "Success", "Forecasts generated successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def generate_timeframe_forecast(self):
        """Generate timeframe-specific forecast"""
        try:
            timeframe = self.forecast_timeframe.currentText()
            
            QMessageBox.information(self, "Generating Forecast", 
                                  f"Generating {timeframe} forecast. This may take a moment...")
            
            # Generate forecast
            forecast = generate_timeframe_prediction(timeframe)
            
            # Reload data and update UI
            self.load_data_and_update_ui()
            
            QMessageBox.information(self, "Success", f"{timeframe} forecast generated successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def retrain_models(self):
        """Force retrain all models"""
        try:
            QMessageBox.information(self, "Retraining Models", 
                                  "Retraining all models. This may take a moment...")
            
            # Retrain models
            generate_ai_forecasts(force_retrain=True, display_insights=False)
            
            # Reload data and update UI
            self.load_data_and_update_ui()
            
            QMessageBox.information(self, "Success", "Models retrained successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def run_analysis(self):
        """Run data analysis"""
        try:
            analysis_type = self.analysis_type.currentText()
            
            # Run analysis (this function prints to console, we need to modify it)
            analyze_data_type(analysis_type)
            
            # Look for the generated chart
            if analysis_type == "Expenses":
                chart_path = os.path.join(EXPORTS_DIR, "expense_distribution.png")
            elif analysis_type == "Savings":
                chart_path = os.path.join(EXPORTS_DIR, "savings_trend.png")
            else:
                chart_path = None
            
            # If chart exists, display it
            if chart_path and os.path.exists(chart_path):
                self.display_chart_from_file(chart_path)
            
            QMessageBox.information(self, "Success", f"{analysis_type} analysis completed.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def show_accuracy(self):
        """Show model accuracy"""
        try:
            timeframe = self.accuracy_timeframe.currentText()
            if timeframe == "All":
                timeframe = None
            
            # Get accuracy data
            # This is a bit tricky since view_accuracy prints to console
            # For now, we'll just run it and look for the exported chart
            view_accuracy(timeframe)
            
            # Look for the generated chart
            chart_path = os.path.join(EXPORTS_DIR, "accuracy_chart.png")
            
            # If chart exists, display it
            if os.path.exists(chart_path):
                self.display_chart_from_file(chart_path)
            
            # We should also update the accuracy table
            self.update_accuracy_table()
            
            QMessageBox.information(self, "Success", "Accuracy data updated.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def update_accuracy_table(self):
        """Update the accuracy table with current data"""
        # This is a simplified version since we don't have direct access to the accuracy data
        # In a real implementation, we would modify view_accuracy to return the data
        
        forecasts_df = load_forecasts()
        if forecasts_df.empty or forecasts_df['accuracy'].isna().all():
            return
        
        # Filter for records with accuracy data
        accuracy_df = forecasts_df[forecasts_df['accuracy'].notna()]
        
        # Calculate average accuracy by model type
        by_model = accuracy_df.groupby('model_type')['accuracy'].mean().reset_index()
        
        # Clear table
        self.accuracy_table.setRowCount(0)
        
        # Add rows to table
        for i, row in enumerate(by_model.itertuples()):
            self.accuracy_table.insertRow(i)
            
            # Model type
            model_item = QTableWidgetItem(row.model_type)
            self.accuracy_table.setItem(i, 0, model_item)
            
            # Accuracy
            accuracy_item = QTableWidgetItem(f"{row.accuracy:.1f}%")
            self.accuracy_table.setItem(i, 1, accuracy_item)
    
    def display_chart_from_file(self, chart_path):
        """Display a chart from a file in the analysis canvas"""
        # Clear previous chart
        self.analysis_canvas.axes.clear()
        
        # Load and display image
        img = matplotlib.image.imread(chart_path)
        self.analysis_canvas.axes.imshow(img)
        self.analysis_canvas.axes.axis('off')  # Hide axes
        
        # Refresh canvas
        self.analysis_canvas.draw()
    
    def export_data(self):
        """Export predictions to CSV"""
        try:
            timeframe = self.forecast_timeframe.currentText()
            export_predictions(timeframe)
            
            QMessageBox.information(self, "Success", f"Predictions exported successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = FinanceTrackerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 