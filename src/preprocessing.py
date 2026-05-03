import pandas as pd
import numpy as np
import sys
import os
import yaml


# load paramerters from yaml file
param= yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocessing(input, output):
    # read the data
    df = pd.read_csv(input)
    
    # drop the null values
    df.dropna(inplace=True)
    
    # drop the duplicates
    df.drop_duplicates(inplace=True)
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # save the preprocessed data
    df.to_csv(output, index=False)
    
    print(f"Preprocessing completed. Preprocessed data saved at {output}")
    
if __name__ == "__main__":
    preprocessing(param["input"], param["output"])
    
    