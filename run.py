#!/usr/bin/env python3
import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import pandas
        import numpy
        import statsmodels
        import prophet
        import sklearn
        import tabulate
        import matplotlib
        import colorama
        import PySide6
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required dependencies:")
        print("pip install pandas numpy statsmodels prophet scikit-learn tabulate matplotlib colorama PySide6")
        return False

def main():
    """Main launcher function"""
    if not check_dependencies():
        return
    
    print("=== AI Finance Tracker ===")
    print("1. Launch GUI Version")
    print("2. Launch CLI Version")
    print("0. Exit")
    
    choice = input("Choose an option: ")
    
    if choice == '1':
        try:
            from gui import main as gui_main
            gui_main()
        except Exception as e:
            print(f"Error launching GUI: {e}")
    elif choice == '2':
        try:
            from main import main as cli_main
            cli_main()
        except Exception as e:
            print(f"Error launching CLI: {e}")
    elif choice == '0':
        print("Goodbye!")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 