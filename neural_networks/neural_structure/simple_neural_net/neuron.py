import numpy as np


class neuron:
    inputs = []
    weights = []
    testing = False
    ActivationFunctions = ""

    def __init__(self, inputs_number=0, bias=1, nu=0.1, activationFunction=""):
        self.init_weights(inputs_number)
        self.bias = bias
        self.nu = nu
        self.ActivationFunctions = activationFunction

    # activation function SIGN =>
    def Activation_functions(func, res):
        match func:
            case "SIGN":
                SIGN(res)
                return

    def SIGN(self, a):
        return 1 if a > 0 else -1

    def _summation(self):
        res = 0
        for Input, weight in zip(self.inputs, self.weights):
            res = res+(Input*weight)
        return res

    def init_weights_Random(self):
        pass

    def init_weights_manual(self, inputs_number):
        err = True
        while (err):
            print(
                f"the weight's number you have to initialize is :{inputs_number} ")
            w = input(
                "enter the initial weight's values: note!! => use comma after each weight EX:(w1,w2,..,wn) ")
            weights = w.split(",")  # extract the weights
            float_weights = []
            if (len(weights) == inputs_number):
                # take n weights where n=number of the inputs (in case you enter so many weights)
                weights = weights[0:inputs_number]
                # convert from a list of strings to a list of floats
                self.weights = list(map(lambda x: float(x), weights))
                err = False
            else:
                print(
                    f"the weight's number doesn't match the input's number{inputs_number}... try again🔁")
                print("="*50)

    def init_weights(self, inputs_number):
        err = True
        while (err):
            init_type = input(
                "initialize the weights \n1->randomly\n2->manualy\n>")
            if (init_type == "2"):
                self.init_weights_manual(inputs_number)
                err = False
            else:
                if (init_type == "1"):
                    self.init_weights_Random(inputs_number)
                    err = False
                else:
                    err = True

    def update_weights(self, Err):
        updated_weights = []

        for Input, weight in zip(self.inputs, self.weights):
            dj = self.nu*Input*(Err)
            wj = weight + dj
            updated_weights.append(wj)
        self.weights = updated_weights

    def predict(self, inputs):
        self.inputs = np.append(inputs, self.bias)
        res = self._summation()
        calculated_output = self.Activation_functions(
            self.activationFunction, res)
        return (calculated_output, self.weights)
