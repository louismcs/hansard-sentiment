import csv
from scipy import stats
from numpy import array


def compute_t():
    with open('Data/Iraq/Evaluation/votes.csv', 'r') as f:
        reader = csv.reader(f)
        p_speeches = []
        n_speeches = []
        for line in reader:
            if line[1] == '1':
                p_speeches.append(int(line[0]))
            else:
                n_speeches.append(int(line[0]))

    ts = stats.ttest_ind(array(p_speeches), array(n_speeches))
    print(ts)


compute_t()
