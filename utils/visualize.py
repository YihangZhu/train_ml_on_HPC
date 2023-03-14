import matplotlib.pyplot as plt


def plot_errors(train_acc, valid_acc, file):
    """
    Function for plotting training and validation losses
    """
    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_acc, color='blue', label='train')
    ax.plot(valid_acc, color='red', label='test')
    ax.set(title="Errors over epochs",
           xlabel='epoch',
           ylabel='err (%)')
    ax.legend()
    # change the plot style to default
    plt.style.use('default')
    plt.savefig(file)
    plt.close()
