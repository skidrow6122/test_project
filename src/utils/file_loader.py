import os
import pandas as pd


def load_csv(file_name: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # absolute path
    csv_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', file_name))  # data path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ could not find file: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="euc-kr")
        print("✅ data load success")
        return df
    except Exception as e:
        print(f"❌ data load fail: {e}")
        return None