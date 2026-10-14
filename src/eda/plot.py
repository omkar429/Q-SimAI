import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import sys

"""Create Here Data Plot and extract data inside"""

# Data load Function
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df

# Save plot function 
def save_plot(plot_path: str) -> None:
    path = pathlib.Path(plot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, format='jpg', bbox_inches="tight")
    plt.close()

def log(data: pd.DataFrame, save_path: str) -> None:
    qids = pd.Series(data['qid1'].tolist() + data['qid2'].tolist())
    plt.figure(figsize=(14,5))
    plt.hist(qids.value_counts(), bins=25)
    plt.yscale('log')
    plt.title('Log-Histogram of question appearance counts')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    save_plot(plot_path=save_path)

# Main function that run all the function 
def main() -> None:
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    data = load_data(data_path=data_path)
    log(data=data, save_path=save_path)

if __name__ == "__main__":
    main()