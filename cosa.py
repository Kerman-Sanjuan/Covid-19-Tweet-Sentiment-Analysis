import pandas as pd
import numpy as np

df = pd.read_csv("csv/best_attr.csv")
df_ = pd.DataFrame(columns=df.columns)
df_.to_csv("headers.csv",index=False,encoding='utf-8')
