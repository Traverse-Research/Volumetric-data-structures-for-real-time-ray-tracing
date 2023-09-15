import pandas as pd
import numpy as np
large_width = 400
np.set_printoptions(linewidth=large_width)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

df = pd.read_csv("captures_summary.txt")
print(df.columns)
columns = df[["k", "iterations", "variance threshold", "RMSE", "ratio"]]
print(columns)
columns_array = columns.to_numpy(copy=True)
print(columns_array)
svd = np.linalg.svd(columns_array)
print("U:\n"+ str(svd.U))
print("S:\n"+ str(svd.S))
print("Vh:\n"+ str(svd.Vh))