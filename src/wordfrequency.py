import csv

from matplotlib import pyplot
def plot_distribution():
    with open('Data/Iraq/Evaluation/ngramranks.csv', 'r') as f:
        reader = csv.reader(f)
        freqs = []
        ranks = []
        for i, line in enumerate(reader):
            freqs.append(int(line[0]))
            ranks.append(i + 1)

    ax = pyplot.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    pyplot.xlabel('Rank')
    pyplot.ylabel('Frequency')
    pyplot.scatter(ranks, freqs, s=1, c='gray', alpha=0.5)
    pyplot.show()
plot_distribution()
