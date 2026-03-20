import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("data.csv")
    df.columns = ["token", "allocated"]
    
    # # Plot dot graph
    # plt.scatter(df["token"], df["allocated"], marker="o", color="blue")
    # plt.xlabel("Token")
    # plt.ylabel("Allocated")
    # plt.title("Dot Graph")
    # plt.grid(True)
    # plt.show()
    
    # Plot line graph
    plt.plot(df["token"], df["allocated"], color="red")
    plt.xlabel("Token")
    plt.ylabel("Allocated")
    plt.title("Line Graph")
    plt.grid(True)
    plt.savefig("no_cache.png")

if __name__ == "__main__":
    main()
