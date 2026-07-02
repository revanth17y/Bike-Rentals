import pathlib
import traceback
import sys

print('Python executable:', sys.executable)
print('Python version:', sys.version)

try:
    import pandas as pd
    import streamlit as st
    import joblib
    import xgboost
    print('pandas:', pd.__version__)
    print('streamlit:', st.__version__)
    print('joblib:', joblib.__version__)
    print('xgboost:', xgboost.__version__)
except Exception as exc:
    print('Import error:')
    traceback.print_exc()
    sys.exit(1)

for name in ['bike_rental_model.pkl', 'scaler.pkl']:
    path = pathlib.Path(name)
    print('\n===', name, '===')
    print('exists:', path.exists())
    if not path.exists():
        continue
    print('size:', path.stat().st_size)
    with path.open('rb') as f:
        header = f.read(64)
    print('header bytes:', header)
    try:
        obj = joblib.load(path)
        print('joblib.load succeeded:', type(obj).__name__)
        if name == 'bike_rental_model.pkl':
            if hasattr(obj, 'predict'):
                print('object appears to be a model with predict()')
            else:
                print('loaded object type:', type(obj))
    except Exception as exc:
        print('joblib.load failed:')
        traceback.print_exc()
        if name == 'bike_rental_model.pkl':
            try:
                import pickle
                with path.open('rb') as f:
                    data = pickle.load(f)
                print('pickle.load succeeded:', type(data).__name__)
            except Exception as exc2:
                print('pickle.load also failed:')
                traceback.print_exc()
