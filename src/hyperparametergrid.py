from matplotlib import pyplot
from trainhelper import generate_rbf_values
from numpy import log10, floor, arange
import pandas

def plot_grid():
    zs = [[0.809280841, 0.809280841, 0.809280841, 0.809280841, 0.809280841],
          [0.809280841, 0.809280841, 0.809280841, 0.809280841, 0.809280841],
          [0.809280841, 0.809280841, 0.809280841, 0.809280841, 0.808898169],
          [0.809280841, 0.809280841, 0.809280841, 0.809280841, 0.792783575],
          [0.809280841, 0.809280841, 0.809280841, 0.804444709, 0.765587156],
          [0.809280841, 0.809280841, 0.802522213, 0.800295934, 0.759659282],
          [0.809280841, 0.809280841, 0.803681803, 0.802409522, 0.79783853]]

    df_correlation = pandas.DataFrame(zs)

    rbf_param_values = generate_rbf_values(5, 7)
    cs = [str(round(c, 2-int(floor(log10(abs(c))))-1)) for c in rbf_param_values['cs']]
    gammas = [str(round(gamma, 2-int(floor(log10(abs(gamma))))-1))
              for gamma in rbf_param_values['gammas']]

    fig, axis = pyplot.subplots()
    fig.subplots_adjust(bottom=0.25, left=0.25)

    heatmap = axis.pcolor(df_correlation, cmap='gray')
    pyplot.colorbar(heatmap)
    pyplot.xlabel('C')
    pyplot.ylabel('Gamma')
    axis.set_xticks(arange(df_correlation.shape[1]) + 0.5, minor=False)
    axis.set_yticks(arange(df_correlation.shape[0]) + 0.5, minor=False)

    axis.set_xticklabels(cs, rotation=90)
    axis.set_yticklabels(gammas)

    pyplot.show()


plot_grid()
