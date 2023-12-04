from math import *
import numpy as np


def Sigmoid(z):
    return 1/(1 + exp(-z))


# def update_weights(weights, err, outL):
#     for indxW, w in enumerate(weights):
#         if (indxW == 2):
#             adjust_out_weights(err, outL[-1], w)
#         else:
#             adjust_hidden_weights(outL[-1], outL[indxW], w)

#     return


# def summation(a, w):
#     sum = 0
#     for i, j in zip(a, w):
#         sum += (j*i)
#     return sum


# def adjust_out_weights(err, outS, weights):
#     dj = outS*(1-outS)*(err)
#     new_wL = []
#     for w in weights[-1]:
#         new_w = w+(1*dj)
#         new_wL.append(new_w)
#     print(new_wL)
#     return new_wL


# def adjust_hidden_weights(outS, out, weights):
#     outS*(1-outS)*(outS*weight)
#     for weight in weights:


dataSet = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],])
out = np.array([
    [0],
    [1],
    [1],
    [0]])
weights = [[1, 2, -3], [-3, -2, 1], [1, 1, -1]]
for dataIndx, data in enumerate(dataSet):
    print(data)
    outL = []
    for indxL, w in enumerate(weights):
        if (indxL == 2):
            outL.append(1)
            summ = summation(outL, w)
        else:
            summ = summation(data, w)
        outS = Sigmoid(summ)
        print(outS)
        outL.append(outS)
    print("out=>", round(outL[-1]))
    if (round(outL[-1]) != out[dataIndx]):
        err = (outL[-1]-out[1][0])**2
        print("err", err)
        # update_weights(weights, err, outL)


# print(summation(weights[0],))

# for input in dataSet:
#     print(input)
#     cOut = []
#     # cOut.append(input)
#     for j in range(neurons):
#         if (k == 1):
#             summ = summation(input, weights[indx])
#         else:
#             summ = summation(cOut, weights[indx])
#         print(summ)
#         Sout = Sigmoid(summ)
#         cOut.append(Sout)
#         print(Sigmoid(summ))
#         cOut = cOut[2:]
#         indx += 1
