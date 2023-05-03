import datetime
import pandas as pd
from django.shortcuts import render

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import base64
import io
import matplotlib.pyplot as plt
from urllib.parse import quote


def decompose_time_series(df):
    df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.drop(columns=['Date'])
    df['Value'] = pd.to_numeric(df['Value'])
    decomposition = sm.tsa.seasonal_decompose(df['Value'], model='additive', period=12)
    return decomposition


def check_stationarity(df):
    df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.drop(columns=['Date'])
    df['Value'] = pd.to_numeric(df['Value'])
    result = adfuller(df['Value'])
    return result


def arima_forecast(df, p, d, q, steps):
    df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.drop(columns=['Date'])
    df['Value'] = pd.to_numeric(df['Value'])
    model = ARIMA(df['Value'], order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def simple_exponential_smoothing(df, alpha, steps):
    df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.drop(columns=['Date'])
    df['Value'] = pd.to_numeric(df['Value'])
    model = SimpleExpSmoothing(df['Value'])
    model_fit = model.fit(smoothing_level=alpha)
    forecast = model_fit.forecast(steps=steps)
    return forecast


def double_exponential_smoothing(df, alpha, beta, steps):
    df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.drop(columns=['Date'])
    df['Value'] = pd.to_numeric(df['Value'])
    model = ExponentialSmoothing(df['Value'], trend='add')
    model_fit = model.fit(smoothing_level=alpha, smoothing_trend=beta)
    forecast = model_fit.forecast(steps=steps)
    return forecast


def holt_winters_forecast(df, seasonal_periods, steps):
    df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.drop(columns=['Date'])
    df['Value'] = pd.to_numeric(df['Value'])
    model = ExponentialSmoothing(df['Value'], trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def forecast_to_html_table(forecast, column_name):
    forecast_df = forecast.to_frame()
    forecast_df.columns = [column_name]
    forecast_df.index.name = 'Date'
    html_table = forecast_df.to_html(
        classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')
    return html_table


def decomposition_to_html_table(decomposition):
    observed_df = decomposition.observed.to_frame()
    trend_df = decomposition.trend.to_frame()
    seasonal_df = decomposition.seasonal.to_frame()
    residual_df = decomposition.resid.to_frame()

    observed_df.columns = ['Observed']
    trend_df.columns = ['Trend']
    seasonal_df.columns = ['Seasonal']
    residual_df.columns = ['Residual']
    decomposition_df = pd.concat([observed_df, trend_df, seasonal_df, residual_df], axis=1)
    decomposition_df.index.name = 'Date'
    html_table = decomposition_df.to_html(
        classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')
    return html_table


def adfuller_to_html_table(adf_result):
    adf_statistic, p_value, lags_used, nobs, critical_values, icbest = adf_result
    result_df = pd.DataFrame(
        {
            "ADF Statistic": [adf_statistic],
            "p-value": [p_value],
            "Lags used": [lags_used],
            "Number of observations": [nobs],
        }
    )
    for key, value in critical_values.items():
        result_df[f"Critical Value ({key})"] = value
    html_table = result_df.to_html(
        classes='table table-striped table-hover table-bordered table-sm table-responsive text-center', index=False)
    return html_table


def plot_to_base64(forecast, title):
    plt.figure()
    plt.plot(forecast, label=title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return quote(base64.b64encode(buf.read()).decode('utf-8'))


def upload_file(request):
    if request.method == 'POST':
        file = request.FILES['file']
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            df = df.dropna()

            for index, row in df.iterrows():
                date_str = row['Date']
                value_str = row['Value']

                # Проверка соответствия формату даты
                try:
                    datetime.datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    df = df.drop(index)
                    continue

                # Проверка соответствия формату числа
                try:
                    float(value_str)
                except ValueError:
                    df = df.drop(index)

            # Пример использования функции arima_forecast()
            forecast = arima_forecast(df, p=5, d=1, q=0, steps=10)
            arima_image = plot_to_base64(forecast, 'ARIMA Forecast')
            forecast_table = forecast_to_html_table(forecast, column_name='ARIMA Forecast')
            table = decomposition_to_html_table(decompose_time_series(df))
            check = adfuller_to_html_table(check_stationarity(df))
            simple = forecast_to_html_table(simple_exponential_smoothing(df, alpha=0.2, steps=10), column_name='Simple')
            double = forecast_to_html_table(double_exponential_smoothing(df, alpha=0.2, beta=0.2, steps=10),
                                            column_name='Double')
            holt = forecast_to_html_table(holt_winters_forecast(df, seasonal_periods=12, steps=10),
                                          column_name='Holt-Winters')
            return render(request, 'success.html',
                          {'arima_image': arima_image, 'forecast_table': forecast_table, 'decompos': table,
                           'check': check, 'simple': simple,
                           'double': double, 'holt': holt})
        else:
            return render(request, 'index.html')
    else:
        return render(request, 'upload.html')


def index(request):
    return render(request, 'index.html')
