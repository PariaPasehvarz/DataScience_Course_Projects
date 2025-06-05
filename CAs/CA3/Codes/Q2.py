import pandas as pd
import xgboost as xgb

df = pd.read_csv('./regression-dataset-train.csv')


df['temp_dif'] = df['feels_like_temp'] - df['temperature']
df['temp_sum'] = (df['feels_like_temp'] + df['temperature'])
df['temp_mult'] = df['feels_like_temp'] * df['temperature']
# df['y_m'] = (df['year'] + 1) * df['month']
# df['y_h'] = (df['year'] + 1) * df['is_holiday']
# df['y_t'] = (df['year'] + 1) * df['temperature']
# df['y_t_s'] = (df['season_id'] + 1) * df['y_t']
df['y_ts'] = (df['year'] + 1) * (df['feels_like_temp'] + df['temperature'])
# df['y_td'] = (df['year'] + 1) * df['temp_dif']
# df['y_tm'] = (df['year'] + 1) * df['temp_mult']
# df['y_ts_s'] = (df['season_id'] + 1) * df['y_ts']
# df['weather_condition2'] = - df['weather_condition'] ** 2
# df['wind_speed2'] = - df['wind_speed']
# df['w2w2'] = -  (df['wind_speed'] ** 2) * (df['weather_condition'] ** 2)

df_model = df.drop(['id', 'date'], axis=1)

X = df_model.drop(columns=['total_users'])
y = df_model['total_users']


xg_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.019, max_depth=4, subsample=0.2, colsample_bytree=0.7,  objective='reg:squarederror', random_state=2)
# xg_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=4, subsample=0.2, colsample_bytree=0.7,  objective='reg:squarederror', random_state=2)

xg_model.fit(X, y)

test_df = pd.read_csv('regression-dataset-test-unlabeled.csv')

ids = test_df['id']

test_df['temp_dif'] = test_df['feels_like_temp'] - test_df['temperature']
test_df['temp_sum'] = test_df['feels_like_temp'] + test_df['temperature']
test_df['temp_mult'] = test_df['feels_like_temp'] * test_df['temperature']
# test_df['y_m'] = (test_df['year'] + 1) * test_df['month']
# test_df['y_h'] = (test_df['year'] + 1) * test_df['is_holiday']
# test_df['y_t'] = (test_df['year'] + 1) * test_df['temperature']
# test_df['y_t_s'] = (test_df['season_id'] + 1) * test_df['y_t']
test_df['y_ts'] = (test_df['year'] + 1) * (test_df['feels_like_temp'] + test_df['temperature'])
# test_df['y_td'] = (test_df['year'] + 1) * test_df['temp_dif']
# test_df['y_tm'] = (test_df['year'] + 1) * test_df['temp_mult']
# test_df['y_ts_s'] = (test_df['season_id'] + 1) * test_df['y_ts']
# test_df['weather_condition2'] = - (test_df['weather_condition'] ** 2)
# test_df['wind_speed2'] = - test_df['wind_speed'] 
# test_df['w2w2'] = - (test_df['weather_condition'] ** 2) * (test_df['weather_condition'] ** 2)

X_test_unlabeled = test_df.drop(['id', 'date'], axis=1)

predictions = xg_model.predict(X_test_unlabeled)

# output_df = pd.DataFrame({'id': ids, 'label': predictions.round().astype(int)})
output_df = pd.DataFrame({'id': ids, 'label': predictions.astype(float)})

output_df.to_csv('regression_predictions.csv', index=False)
print("Predictions saved to 'regression_predictions.csv'")