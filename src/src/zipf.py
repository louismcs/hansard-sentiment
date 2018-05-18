import csv

from matplotlib import pyplot
def plot_zipf():
    with open('Data/Iraq/Evaluation/words.csv', 'r') as f:
        reader = csv.reader(f)
        freqs = []
        ranks = []
        for line in reader:
            freqs.append(int(line[0]))
            ranks.append(int(line[3]))

    ax = pyplot.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    pyplot.xlabel('Rank')
    pyplot.ylabel('Frequency')
    pyplot.scatter(ranks, freqs, s=1, c='gray', alpha=0.5)
    pyplot.show()


plot_zipf()
