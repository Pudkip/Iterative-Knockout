import operator
import math
import copy
import itertools

def Mean_Log_Loss(predictions, labels, limit=10):
    lim = 10 ** -limit
    labels = labels.tolist()
    cost = list(map(operator.sub, labels, predictions))
    cost_adj = []
    for i in range(len(cost)):
        cost_adj.append(-math.log10(abs(cost[i] + lim)))
    accuracy = sum(cost_adj) / len(cost_adj)
    return accuracy

def Single_Iterative_Knockout(features_knockout, model, labels, baseline):
    inp = copy.copy(features_knockout)
    accuracies = []
    for i in range(features_knockout.shape[1]):
        for j in range(features_knockout.shape[0]):
            features_knockout[j][i] = [0]
        predictions = model.predict(features_knockout)
        predictions = predictions.reshape(features_knockout.shape[0], ).tolist()
        accuracy = Mean_Log_Loss(predictions=predictions, labels = labels)
        accuracies.append(accuracy)
        features_knockout = copy.copy(inp)
    accuracies[:] = [abs(x - baseline) for x in accuracies]

    return accuracies

def N_Gram_Iterative_Knockout(features_knockout, model, labels, baseline, gram_size=2):
    inp = copy.copy(features_knockout)
    accuracies = []
    index = []
    for i in range(features_knockout.shape[1]-gram_size):
        for j in range(features_knockout.shape[0]):
                features_knockout[j][i:i+gram_size] = [0]
        index.append(str(i)+":"+str(i+gram_size-1))
        predictions = model.predict(features_knockout)
        predictions = predictions.reshape(features_knockout.shape[0], ).tolist()
        accuracy = Mean_Log_Loss(predictions=predictions, labels = labels)
        accuracies.append(accuracy)
        features_knockout = copy.copy(inp)
    accuracies[:] = [abs(x - baseline) for x in accuracies]

    return accuracies, index

def High_Order_Iterative_Knockout(features_knockout, model, labels, baseline,):
    inp = copy.copy(features_knockout)
    accuracies = []
    iter_list = list(range(features_knockout.shape[1]))
    combinations = []
    for k in range(features_knockout.shape[1]):
        combinations.append(list(itertools.combinations(iter_list, k)))
    combinations = list(itertools.chain.from_iterable(combinations[1:]))
    combinations = [list(item) for item in combinations]
    for i in range(len(combinations)):
        for j in range(features_knockout.shape[0]):
                features_knockout[j][combinations[i]] = [0]
        predictions = model.predict(features_knockout)
        predictions = predictions.reshape(features_knockout.shape[0], ).tolist()
        accuracy = Mean_Log_Loss(predictions=predictions, labels = labels)
        accuracies.append(accuracy)
        features_knockout = copy.copy(inp)
    accuracies[:] = [abs(x - baseline) for x in accuracies]

    return accuracies, combinations
