import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_idx = y_true != 0
    return np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100

def evaluate_model(X, y, model, normalize=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Train R2": r2_score(y_train, y_pred_train),
        "Train MAE": mean_absolute_error(y_train, y_pred_train),
        "Train MSE": mean_squared_error(y_train, y_pred_train),
        "Train MAPE": mean_absolute_percentage_error(y_train, y_pred_train),

        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Test R2": r2_score(y_test, y_pred_test),
        "Test MAE": mean_absolute_error(y_test, y_pred_test),
        "Test MSE": mean_squared_error(y_test, y_pred_test),
        "Test MAPE": mean_absolute_percentage_error(y_test, y_pred_test),
    }

    return results


df = pd.read_csv('regression-dataset-train.csv')

df['temp_dif'] = df['feels_like_temp'] - df['temperature']
df['temp_sum'] = df['feels_like_temp'] + df['temperature']
df['temp_mult'] = df['feels_like_temp'] * df['temperature']
df['y_m'] = (df['year'] + 1) * df['month']
df['y_h'] = (df['year'] + 1) * df['is_holiday']
df['y_t'] = (df['year'] + 1) * df['temperature']
df['y_t_s'] = (df['season_id'] + 1) * df['y_t']
df['y_ts'] = (df['year'] + 1) * df['temp_sum']
df['y_td'] = (df['year'] + 1) * df['temp_dif']
df['y_tm'] = (df['year'] + 1) * df['temp_mult']
df['y_ts_s'] = (df['season_id'] + 1) * df['y_ts']
df['weather_condition2'] = - df['weather_condition'] ** 2
df['wind_speed2'] = - df['wind_speed'] ** 2
df['w2w2'] = - df['wind_speed2'] * df['weather_condition2']

feature_sets = {
    "initial": ['feels_like_temp', 'temperature', 'humidity', 'wind_speed', 
                 'year', 'month', 'season_id', 'is_workingday', 
                 'weekday', 'is_holiday', 'weather_condition'],

    "full_set":  ['feels_like_temp', 'temperature', 'humidity', 'wind_speed', 
                 'year', 'month', 'season_id', 'is_workingday', 
                 'weekday', 'is_holiday', 'weather_condition',
                 'temp_dif','temp_mult', 'temp_sum', 'y_m', 'y_t', 'y_t_s', 
                 'y_ts', 'y_td', 'y_tm', 'weather_condition2',
                 'w2w2', 'wind_speed2'],

    "best_in_test":  ['feels_like_temp', 'temperature', 'humidity', 'wind_speed', 
                 'year', 'month', 'season_id', 'is_workingday', 
                 'weekday', 'is_holiday', 'weather_condition',
                 'temp_dif', 'temp_sum', 'temp_mult', 'y_ts']
}

models = {
    # "LinearRegression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, subsample=0.2, objective='reg:squarederror', random_state=42),
    "XGBoost2": xgb.XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=4, subsample=0.2, objective='reg:squarederror', random_state=42),
    "XGBoost3": xgb.XGBRegressor(n_estimators=500, learning_rate=0.019, max_depth=4, subsample=0.2, objective='reg:squarederror', random_state=42),
    "RandomForest": RandomForestRegressor(max_depth=9, min_samples_leaf=1, min_samples_split=3, n_estimators=150 ,random_state=42),
    "RandomForest2": RandomForestRegressor(max_depth=36, min_samples_leaf=1, min_samples_split=3, n_estimators=250 ,random_state=42)
}

y = df['total_users']

results = []

for name, features in feature_sets.items():
    X = df[features]
    for norm in [False, True]:
        for model_name, model in models.items():
            metrics = evaluate_model(X, y, model, normalize=norm)
            results.append({
                "Features": name,
                "Model": model_name,
                "Normalized": norm,
                **{k: round(v, 2) for k, v in metrics.items()}
            })


results_df = pd.DataFrame(results)
print(results_df.sort_values(by="Test RMSE"))
