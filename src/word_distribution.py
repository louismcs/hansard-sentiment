import csv

from matplotlib import pyplot
def plot_distribution():
    with open('Data/Iraq/Evaluation/freqoffset2.csv', 'r') as f:
        reader = csv.reader(f)
        xs = []
        ys = []
        for line in reader:
            xs.append(int(line[0]))
            ys.append(float(line[1]))

    ax = pyplot.gca()
    #ax.set_xscale('log')
    pyplot.axhline(0, ls='--', color='black', lw=0.1)
    pyplot.xlabel('Word Frequency')
    pyplot.ylabel('Probability Offset')
    pyplot.scatter(xs, ys, s=1, c='gray')
    pyplot.show()
plot_distribution()
