import matplotlib.pyplot as plt
import seaborn as sns

def plot_barplot(data, x, y, title, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()