from preprocessing import load_and_preprocess
from modeling import train_and_evaluate
from evaluation import show_results

def main():
    df = load_and_preprocess("insurance.csv")
    results_df = train_and_evaluate(df)
    show_results(results_df)

if __name__ == "__main__":
    main()