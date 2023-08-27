from torch import nn


def build_hidden_layers(num_layers, input_size, layers_size, output_size):
    linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
    linears.extend([nn.Linear(layers_size, layers_size) for i in range(1, num_layers - 1)])
    linears.append(nn.Linear(layers_size, output_size))
    print(linears)
    return linears

# c_new: OrderedDict([
#     ('fc1', Linear(in_features=1024, out_features=256, bias=True)),
#     ('relu1', ReLU()),
#     ('drop1', Dropout(p=0.2, inplace=False)),
#     ('hidden0', Linear(in_features=256, out_features=256, bias=True)),
#     ('hrelu0', ReLU()),
#     ('hdrop0', Dropout(p=0.2, inplace=False)),
#     ('fc2', Linear(in_features=256, out_features=1000, bias=True)),
#     ('output', LogSoftmax(dim=1))])
