from neuron import neuron
import numpy as np
import matplotlib.pyplot as pl


class Multi_layer:
    Neurons_number_HL = 0  # hidden layer Neuron's number
    Neurons_number_OutL = 1  # output layer Neuron's number
    inputs_number = 0
    Hidden_layers = []  # stor all the neurons of the H.L
    _layers = []  # include the hidden.L and the output.L
    O_layers = []
    inputs = []
    outputs = []
    Layers_struct = {}
    Layers_number = 0
    dns = []

    # Layers = np.zeros(Layers_number)

    def set_weights(self, _range):
        def func(neuron, params): return neuron.init_weights_Random(
            _range, self.inputs_number)
        self.Network_traversing(func)
        # for layer in self.Hidden_layers:
        # pass

    def set_weights_manual(self, weights):
        def func(neuron, params):
            neuron.SetWeights(params)
        self.Network_traversing(func, weights, probagation=False)

    def Network_traversing(self, func, params=(), probagation=True):
        _layers = self.Hidden_layers
        new_params = []
        out_list = []
        neuronCount = 0
        for layer_indx in range(len(_layers)):
            layer = _layers[layer_indx]
            if (not (probagation)):
                for neuron in layer:
                    param = params[neuronCount]
                    neuronCount += 1
                    out = func(neuron, param)
            else:
                if (layer_indx == 0):
                    for p in params:
                        out_list.append(p)
                    for neuron in layer:
                        out = func(neuron, params)
                        new_params.append(out)
                        out_list.append(out)
                else:
                    # the length of the previous hidden layer
                    print("new params", new_params)
                    Hl_befor = len(_layers[layer_indx-1])
                    for neuron in layer:
                        out = func(neuron, new_params)
                        new_params.append(out)
                        out_list.append(out)
                    new_params = new_params[2:]
        return out_list

    def forward_probagation(self, learning_rate, gen_backpr=False):
        def func(neuron, params):
            _input = params
            print("weights", neuron.predict(_input)[
                  1], "out", neuron.predict(_input)[0])
            return neuron.predict(_input)[0]
        rate = 0
        all_errSum = []
        gen_outputs = []
        while (rate < learning_rate):
            print(f"===================={rate}============================")
            errSumm = 0
            for _input, out in zip(self.inputs, self.outputs):
                print(_input)
                All_outputs = self.Network_traversing(func, (_input))
                gen_outputs.append(All_outputs)
                print("all outputs", All_outputs)
                print("real output", out)
                err = (All_outputs[-1]-out)
                errSumm += err
                Round_err = (round(All_outputs[-1])-out)**2
                print("err=>", err)
                if (err != 0 and not (gen_backpr)):
                    print("back probagation")
                    self.back_probagation(All_outputs, err)
            err_rate = errSumm/len(self.inputs)
            if (gen_backpr):
                # for out in gen_outputs:
                print("true ==================")
                print("==>", err_rate, All_outputs)
                self.back_probagation(All_outputs, err_rate)

            print(f"err sum {rate}  =>", errSumm/len(self.inputs))
            all_errSum.append(errSumm/len(self.inputs))
            rate += 1
        return (all_errSum)

    def back_probagation(self, outputs, err):
        # HLayers = self._layers[0:-1]
        # self.O_layers output layer
        self.adjust_out_weights(err, outputs)
        self.adjust_hidden_weights(err, outputs)

    def adjust_out_weights(self, err, outputs):
        # print(self.O_layers)
        self.dns = []
        prediction_outs = outputs[-self.Neurons_number_OutL:]
        hidden_outs = outputs[0:-self.Neurons_number_OutL]
        hidden_outs = hidden_outs[-self.Neurons_number_HL:]+[1]
        for neuron, out in zip(self.O_layers, prediction_outs):
            hidden_outs[-1] = neuron.bias
            new_weights = []
            dn = self.calculate_dn(out, err)
            self.dns.append(dn)
            for weight, h_out in zip(neuron.weights, hidden_outs):
                dw = neuron.nu*dn*h_out
                new_weight = weight+dw
                new_weights.append(new_weight)
            neuron.SetWeights(new_weights)

        print("outsssss", new_weights)

    def calculate_dn(self, out, err):
        return out*(1-out)*err

    def adjust_hidden_weights(self, err, outputs):
        prediction_outs = outputs[-self.Neurons_number_OutL:]
        hidden_outs = outputs[-self.Neurons_number_HL:]+[1]
        hidden_in = outputs[0:self.Neurons_number_HL:]+[1]
        # print(hidden_outs)
        hidden_Layers = self.Hidden_layers[0:-self.Neurons_number_OutL]
        # print(hidden_Layers[0][0].w)
        summ_dns = 0
        for layer in hidden_Layers:
            indx = 0
            for neuron, out in zip(layer, hidden_outs):
                hidden_outs[-1] = neuron.bias
                new_weights = []
                # print("sss1", neuron.weights)
                # print("sss1", neuron.weights)
                out_indx = 0
                for oustS, Out in zip(prediction_outs, self.O_layers):
                    summ_dns += Out.weights[out_indx] * \
                        self.calculate_dn(oustS, err)
                    out_indx += 1
                print("sss1", summ_dns)
                dnh = out*(1-out)*summ_dns
                for weight, Hin in zip(neuron.weights, hidden_in):
                    dw = neuron.nu*dnh*Hin
                    new_weight = weight+dw
                    new_weights.append(new_weight)
                print("sss", neuron.weights)
                neuron.SetWeights(new_weights)
                print("new We", new_weights)
                indx += 1

    def Network_struct(self):
        print("struct", self.Layers_struct)
        print(len(self.Hidden_layers)-1, " hidden layers")
        print(len(self.Hidden_layers[0])*self.Layers_number +
              self.Neurons_number_OutL, "neurons")

    def get_outputs(self, dataSet):
        Outputs = []
        for i in range(len(dataSet)):
            RowLen = len(dataSet[i])
            for j in range(RowLen):
                if (j == RowLen-1):
                    Outputs.append(dataSet[i][RowLen-1])
        return Outputs

    def get_inputs(self, dataSet):
        Inputs = []
        for i in range(len(dataSet)):
            RowLen = len(dataSet[i])
            row = []
            for j in range(RowLen-1):
                row.append(dataSet[i][j])
            Inputs.append(row)
        return Inputs

    def setup_dataset(self, dataSet):
        self.outputs = self.get_outputs(dataSet)
        self.inputs = self.get_inputs(dataSet)
        self.inputs_number = len(self.inputs[0])
        self.Layers_struct["inputs_layer"] = self.inputs_number

    def init_layers(self, Layers_number):
        self.Layers_struct = {"inputs_layer": self.inputs_number,
                              "hidden_layer": [self.Neurons_number_HL for x in range(Layers_number)],
                              "outputs_layer": self.Neurons_number_OutL}

        for _ in range(Layers_number):
            _Hlayers = [neuron() for x in range(self.Neurons_number_HL)]
            self.Hidden_layers.append(_Hlayers)
        self.O_layers = [neuron() for x in range(self.Neurons_number_OutL)]
        self._layers = [self.Hidden_layers, self.O_layers]
        self.Hidden_layers.append(self.O_layers)
        self.Network_struct()

    def __init__(self, Layers_number=0, Neurons_number=0, Neurons_number_OutL=1,):
        self.Neurons_number_HL = Neurons_number
        self.Neurons_number_OutL = Neurons_number_OutL
        self.Layers_number = Layers_number
        # self.Layers = np.zeros(Layers_number)
        # self.Hidden_Neurons = np.zeros(Neurons_number)
        # self.Out_Neurons = np.zeros(Neurons_number_OutL)
        self.init_layers(Layers_number)

    #     for layer in self.Hidden_layers:
    #         for neuron in layer:
    #             print(neuron.)

    def plot_error_summation(X_range, Y_points):

        # X_points = [x for x in range(X_range)]
        x = np.arange(0, X_range)
        y = np.array(Y_points)
        # print(y)
        # y = np.array(Y_points)
        # y = 2*x+1
        # y = Y_points[X_points]
        pl.xlabel("learning rate")
        pl.ylabel("error ")
        pl.plot(x, y)

        pl.show()
        # pl.savefig("kk.png", format="png")
        return


def main():
    # dataSet = np.array([
    #     [1, 1, 1],
    #     [0, 2, 1],
    #     [1, 3, 1],
    #     [0, -1, 0],
    #     [2, 0, 0],
    #     [4, 2, 0]])
    # dataSet = np.array([
    #     [0.35, 0.9, 0.5]])
    # dataSet = np.array([
    #     [0, 0, 0],
    #     [0, 1, 1],
    #     [1, 0, 1],
    #     [1, 1, 0]])
    dataSet = np.array([
        [1, 1, 1],
        [0, 2, 1],
        [1, 3, 1],
        [0, -1, 0],
        [2, 0, 0],
        [4, 2, 0]])
    learning_rate = 200
    MultiLayer = Multi_layer(
        Layers_number=1, Neurons_number=2, Neurons_number_OutL=1)
    MultiLayer.setup_dataset(dataSet)
    MultiLayer.Network_struct()
    # MultiLayer.set_weights((-6, 3))
    MultiLayer.set_weights_manual(
        [[0.7, -0.2, 0.4], [-0.4, 0.3, 0.6], [0.5, 0.1, -0.3]])
    all_errSumm = MultiLayer.forward_probagation(
        learning_rate, gen_backpr=False)
    allS = np.array(all_errSumm)
    min_err = allS.min()
    print(allS)
    print(min_err)
    Multi_layer.plot_error_summation(learning_rate, all_errSumm)
    # MultiLayer.init_layexrs()


main()
