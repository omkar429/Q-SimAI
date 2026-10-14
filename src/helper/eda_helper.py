import pandas as pd

# Data Load Function
def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except NotADirectoryError:
        raise("Directory Not Found")

