import pandas as pd

def main():

    as1_df = pd.read_csv('./Generated Data/AS1_samples.csv')
    as1_df.describe().to_csv('./Generated Data/AS1_described.csv',index=True)

if __name__ == "__main__":
    main()


