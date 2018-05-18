from matplotlib import pyplot
from trainhelper import generate_rbf_values
from numpy import log10, floor, arange
import pandas

def plot_grid():
    zs = [[0.791918106,0.791918106,0.791918106,0.794826855,0.792935253],
          [0.791918106,0.791918106,0.791918106,0.798178678,0.785651818],
          [0.791918106,0.791918106,0.791918106,0.79369041,0.766947142],
          [0.791918106,0.791918106,0.791918106,0.792344372,0.767818774],
          [0.791918106,0.791918106,0.791918106,0.784658575,0.776861219],
          [0.791918106,0.791918106,0.787881048,0.778809308,0.780153575],
          [0.791918106,0.791918106,0.789564832,0.785312651,0.784740528]]


    df_correlation = pandas.DataFrame(zs)

    rbf_param_values = generate_rbf_values(5, 7)
    cs = [str(round(c, 2-int(floor(log10(abs(c))))-1)) for c in rbf_param_values['cs']]
    gammas = [str(round(gamma, 2-int(floor(log10(abs(gamma))))-1))
              for gamma in rbf_param_values['gammas']]
    
    fig, axis = pyplot.subplots()
    fig.subplots_adjust(bottom=0.25, left=0.25)

    heatmap = axis.pcolor(df_correlation, cmap='gray')
    cbar = pyplot.colorbar(heatmap)
    cbar.ax.set_ylabel('F1 Score\n', rotation=90, labelpad=10)
    pyplot.xlabel('C')
    pyplot.ylabel('Gamma')
    axis.set_xticks(arange(df_correlation.shape[1]) + 0.5, minor=False)
    axis.set_yticks(arange(df_correlation.shape[0]) + 0.5, minor=False)

    axis.set_xticklabels(cs, rotation=90)
    axis.set_yticklabels(gammas)

    pyplot.show()


plot_grid()
