import numpy as np
from random import randint
from keras.utils import to_categorical
np.set_printoptions(threshold=np.nan)

def Create_Features(sample_size, sequence_length, feature_length):
    features = []
    for i in range(sample_size):
        _ = []
        for j in range(sequence_length):
            _.append(randint(0,feature_length))
        features.append(_)

    features = np.array(features)

    features = to_categorical(features)
    features = features.reshape([sample_size,sequence_length,feature_length])
    print("features shape: ", features.shape)
    return features

def Create_Labels(sample_size):
    phenotype = []
    for _ in range(sample_size):
        phenotype.append(randint(0,1))
    phenotype = np.array(phenotype)
    print("phenotype shape: ", phenotype.shape)
    return phenotype

def Sample_Data(sample_size,sequence_length, target):
    features = []
    for i in range(sample_size):
        _ = []
        for j in range(sequence_length):
            if j == target and i + randint(0,1)%2 ==0:
                _.append(1)
            else:
                _.append(randint(0, 3))
        features.append(_)

    phenotype = []
    for x in range(sample_size):
        if features[x][target] == 1:
            phenotype.append(1)
        else:
            phenotype.append(0)

    features = np.array(features)

    features = to_categorical(features)
    features = features.reshape([sample_size,sequence_length,4])
    phenotype = np.array(phenotype)
    return features, phenotype