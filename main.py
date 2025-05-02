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
