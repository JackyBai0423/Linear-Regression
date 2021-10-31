import numpy as np
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_col_mean(dataset, col):
    sum_mean = 0
    for i in dataset:
        sum_mean += float(i[col])
    return sum_mean / len(dataset)


def get_dataset(filename):
    """


    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = open(filename, "r", encoding="utf-8")
    result = []

    for i in dataset:
        i = i.strip()
        result.append(i.split(','))
    result.pop(0)
    for i in result:
        i.pop(0)
    result = np.array(result)
    return result


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    print(len(dataset))
    sum_deviation = 0
    mean = get_col_mean(dataset, col)
    print("{:.2f}".format(mean))
    for i in dataset:
        sum_deviation += pow(float(i[col]) - mean, 2)

    print("{:.2f}".format(np.sqrt(1 / (len(dataset) - 1) * sum_deviation)))
    pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    sum_deviate = 0
    for i in dataset:
        temp = betas[0]
        for j in range(0, len(cols)):
            temp += betas[j + 1] * float(i[cols[j]])
        sum_deviate += pow(temp - float(i[0]), 2)
    mse = sum_deviate / len(dataset)
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    gradients = [0] * len(betas)
    result = []
    for i in dataset:
        temp = betas[0]
        for j in range(0, len(cols)):
            temp += betas[j + 1] * float(i[cols[j]])
        for j in range(0, len(betas)):
            if j == 0:
                gradients[0] += (temp - float(i[0]))
            else:
                gradients[j] += (temp - float(i[0])) * float(i[cols[j - 1]])
    for i in range(0, len(gradients)):
        result.append((2 / len(dataset)) * gradients[i])
    return result


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """

    for i in range(1, T + 1):
        gradients = gradient_descent(dataset, cols, betas)
        for j in range(0, len(betas)):
            betas[j] -= eta * gradients[j]
        print(i, '{:.2f}'.format(regression(dataset, cols, betas)), end=' ')
        for j in betas:
            print('{:.2f}'.format(j), end=' ')
        if i == 1:
            print('# order: T, mse, beta0', end='')
            for k in cols:
                print(', beta' + str(k), end='')
        print()
    pass


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    cols.insert(0, 0)
    Xs = np.zeros((len(dataset), len(cols)))
    Ys = np.zeros((len(dataset), 1))
    mse = 0
    for i in range(0, len(dataset)):
        Ys[i][0] = float(dataset[i][0])
        for j in range(0, len(cols)):
            Xs[i][j] = float(dataset[i][cols[j]])
        Xs[i][0] = 1
    Xs_trans = np.transpose(Xs)
    result = np.dot(np.linalg.inv(np.dot(Xs_trans, Xs)), np.dot(Xs_trans, Ys))
    betas = []
    for i in result:
        betas.append(i[0])
    mse = regression(dataset, cols[1:], betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)[1:]
    features.insert(0, 1)
    result = np.dot(betas, np.transpose(features))
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    result_linear = []
    result_quadratic = []
    for i in X:
        zi = np.random.normal(0, sigma)
        i = list(i)
        i.insert(0, 1)
        result_linear.append([np.dot(betas, i) + zi, i[1]])
        temp = alphas[0]
        for j in range(1, len(betas)):
            temp += alphas[j] * pow(i[j], 2)
        result_quadratic.append([temp + zi, i[1]])

    return np.array(result_linear), np.array(result_quadratic)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')
    Xs = []
    sigmas = []
    for i in range(0, 1000):
        Xs.append([np.random.randint(-100, 100)])
    for i in range(-4, 6):
        sigmas.append(pow(10, i))
    plt.xlabel('Standard Deviation of Error Term')
    plt.ylabel('MSE of Trained Model')
    betas = np.array([0, 2])
    alphas = np.array([0, 1])
    linear = []
    quadratic = []
    linearYs = []
    quadraticYs = []
    for i in sigmas:
        linear.append(synthetic_datasets(betas, alphas, Xs, i)[0])
        quadratic.append(synthetic_datasets(betas, alphas, Xs, i)[1])
    for i in linear:
        linearYs.append(compute_betas(i, [1])[0])
    for i in quadratic:
        quadraticYs.append(compute_betas(i, [1])[0])
    plt.xscale('log')
    plt.yscale('log')

    line1, = plt.plot(sigmas, linearYs, marker='o')
    line2, = plt.plot(sigmas, quadraticYs, marker='o')
    plt.legend([line1, line2], ['MSE of Linear Dataset', 'MSE of Quadratic Dataset'], loc='upper left')
    plt.savefig('mse.pdf')



if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
dataset = get_dataset('bodyfat.csv')
iterate_gradient(dataset, cols=[1,8], betas=[400,-400,300], T=10, eta=1e-4)
