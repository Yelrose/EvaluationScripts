from scipy.stats.stats import spearmanr
import sys
import numpy as np


def get_sim_function(string):
    return eval(string)
def main():

    if len(sys.argv) < 4:
        print '''
            Usage: python ws_eval.py <representation_name> <representation_path> <task_path> [lambda_expression]

            We are providing some code here:
                import numpy as np
            The default similarity measure is cosine distance
            Example:
            If you want to use euclid distance
                python ws_eval.py skipgram vectors.txt wordsim353 'lambda x,y:np.sqrt((x - y).T.dot(x - y))'
        '''
        return


    representation = create_representation(sys.argv[2])
    data,tot,found = read_test_set(sys.argv[3],representation)
    sim = lambda x,y: x.dot(y) / np.linalg.norm(x) / np.linalg.norm(y)
    if len(sys.argv) >= 5:
        sim= get_sim_function(sys.argv[4])
    correlation = evaluate(representation, data,sim)
    print sys.argv[1], sys.argv[2], '\t%0.3f' % correlation,tot,found

def create_representation(path):
    representation = {}
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            representation[line[0]] = np.array([float(wd)  for wd in line[1:]])
    return representation



def read_test_set(path,representation):
    test = []
    vocabs = {}
    not_found = 0
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            if x not in vocabs: vocabs[x] = 0
            if y not in vocabs: vocabs[y] = 0
            if x not in representation:
                if vocabs[x] == 0:
                    not_found += 1
                    vocabs[x] += 1
            if y not in representation:
                if vocabs[y] == 0:
                    not_found += 1
                    vocabs[y] += 1
            test.append(((x, y), sim))
    return test,len(vocabs),len(vocabs) - not_found


def evaluate(representation, data,sim):
    results = []
    for (x, y), sim in data:
        results.append((sim(representation[x], representation[y]), sim))
    actual, expected = zip(*results)
    return spearmanr(actual, expected)[0]


if __name__ == '__main__':
    main()
