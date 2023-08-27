from torch import nn


def build_hidden_layers(num_layers, input_size, layers_size, output_size):
    linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
    linears.extend([nn.Linear(layers_size, layers_size) for i in range(1, num_layers - 1)])
    linears.append(nn.Linear(layers_size, output_size))
    print(linears)
    return linears