# Visualizes the searched values from the hyperparameter search and saves them as images.
# importr matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd
import pickle
from training import TrainLogger, TrainConfig

def get_results_df(results):

    # plot lr logarithmicly and loss using seaborn
    lrs = [param_set[0] for param_set, log in results]
    batch_sizes = [param_set[1] for param_set, log in results]
    losses = [log.get_best_loss() for param_set, log in results]
    trial = range(len(losses))

    df = pd.DataFrame({"lr": lrs, "loss": losses, "batch_size": batch_sizes, "trial": trial})

    return df

def plot_paramsearch(results_pickle_name, output_filename):
    with open(results_pickle_name, 'rb') as f:
        results = pickle.load(f)

    df = get_results_df(results)

    # sns.scatterplot(data=df, x="lr", y="loss")
    # plt.xscale("log")
    # plt.show()


    # plot epoch and lrs with the loss as hue
    # hue palette should be green for low loss and red for high loss
    palette = sns.color_palette("RdYlGn", len(df["loss"]))
    palette = palette[::-1] # reverse the palette
    sns.scatterplot(data=df, x="batch_size", y="lr", hue="loss", palette=palette)
    # since loss is a float, create a simple legend instead of per value
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='loss',
                              markerfacecolor=palette[-1], markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='loss',
                              markerfacecolor=palette[int(len(palette)/2)], markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='loss',
                              markerfacecolor=palette[0], markersize=10)]
    lowest_loss = df["loss"].min()
    highest_loss = df["loss"].max()
    med_loss = (lowest_loss + highest_loss)/2
    plt.legend(handles=legend_elements, labels=[f"high loss ({highest_loss:.3f})",
                                                f"medium loss ({med_loss:.3f})",
                                                f"low loss ({lowest_loss:.3f})",
                                                ])

    plt.xlabel("Batch size")
    plt.ylabel("Learning rate")
    plt.title("Loss for different tested parameters")

    plt.yscale("log")

    plt.savefig(output_filename)
    plt.show()
    plt.cla()

# plot_paramsearch("outputs/paramsearch small dataset results/gridsearch_supervised_20230822_123450.pickle",
#                  output_filename="outputs/paramsearch small dataset results/grid_search.png")
#
# plot_paramsearch("outputs/paramsearch small dataset results/randomsearch_supervised_20230822_124548.pickle",
#                  output_filename="outputs/paramsearch small dataset results/random_search.png")
#
# plot_paramsearch("outputs/paramsearch small dataset results/hyperopt_supervised_20230822_142959_all.pickle",
#                  output_filename="outputs/paramsearch small dataset results/hyperopt_search.png")

plot_paramsearch("outputs/hyperopt/hyperopt_rot-only_20230822_212642_all.pickle", output_filename="outputs/optimized runs/rot_search.png")