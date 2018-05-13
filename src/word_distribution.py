import csv

from matplotlib import pyplot
def plot_distribution():
    with open('Data/Iraq/Evaluation/ngramoffset.csv', 'r') as f:
        reader = csv.reader(f)
        xs = []
        ys = []
        for line in reader:
            xs.append(int(line[0]))
            ys.append(float(line[1]))

    ax = pyplot.gca()
    ax.set_xscale('log')
    pyplot.xlabel('Word Frequency')
    pyplot.ylabel('Probability Offset')
    pyplot.scatter(xs, ys, s=1, c='gray', alpha=0.5)
    pyplot.show()
plot_distribution()
