import pathlib
import joblib
import xgboost
p = pathlib.Path(r'C:\Users\kumar\Downloads\New folder (2)\bike_rental_model.pkl')
print('path', p)
print('exists', p.exists())
print('size', p.stat().st_size)
print('xgboost', xgboost.__version__)
try:
    joblib.load(p)
    print('loaded ok')
except Exception as e:
    print(type(e).__name__, e)
