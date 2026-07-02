import importlib.util
import pathlib
import traceback

path = pathlib.Path(r"C:\Users\kumar\Downloads\New folder (2)\app (1) (1).py")
print('exists', path.exists())
spec = importlib.util.spec_from_file_location('app_bike', path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('import succeeded')
    print('model_error type', type(mod.model_error).__name__, bool(mod.model_error))
    print('scaler_error type', type(mod.scaler_error).__name__, bool(mod.scaler_error))
    print('model type', type(mod.model).__name__)
    print('scaler type', type(mod.scaler).__name__)
except Exception:
    traceback.print_exc()
