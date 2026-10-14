import pandas as pd
import numpy as np
import sys
from sklearn.metrics import log_loss



# Data Load Function
def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except NotADirectoryError:
        raise("Directory Not Found")


# Data Summary Function
def data_summary(data: pd.DataFrame) -> None:
    try:
        data_shape = data.shape
        null_values = data.isnull().mean()
        duplicated_data = data.duplicated().mean()
        data_datatype = data.info()
        qids = pd.Series(data['qid1'].tolist() + data['qid2'].tolist())

        # print all the values
        print("_" * 100)
        print(f"Data Shape: {data_shape}")
        print("_" * 100)
        print(f"Null value In Columns: {null_values * 100}")
        print("_" * 100)
        print(f" Duplicated Data: {duplicated_data * 100}%")
        print("_" * 100)
        print(f" Columns: {data.columns}")
        print("_" * 100)
        print("Data Types:",end="")
        print("_" * 100)
        print('Total number of questions in the data: {}'.format(len(np.unique(qids))))
        print("_" * 100)
        print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))
        print("_" * 100)
        print(f"Unique Qus in Qid1:{len(data.qid1.unique())}")
        print("_" * 100)
        print(f"Unique Qus in Qid1:{len(data.qid2.unique())}")
        print("_" * 100)
        print(data['is_duplicate'].value_counts())
        
        p = data['is_duplicate'].mean() # Our predicted probability
        print('Predicted score:', log_loss(data['is_duplicate'], np.zeros_like(data['is_duplicate']) + p))
        print("-" * 100)
    except Exception as e:
        raise RuntimeError(
            "Something went wrong in the data summary function"
        ) from e



# Main Function run all the function 
def main() -> None:
    data_path = sys.argv[1]
    data = load_data(data_path)
    data_summary(data=data)

if __name__ == "__main__":
    main()