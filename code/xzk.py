from data_loader import DataGenerator
import numpy as np

if __name__ == '__main__':
    data_generate = DataGenerator(train=True, batch_size=64)
    for i in data_generate.__iter__():
        t = i[0][0]
        mask = np.greater(np.expand_dims(t, 2), 0)
        print(t)
