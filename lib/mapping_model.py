import torch
import torch.nn as nn
import torch.nn.functional as Functional


# https://github.com/Zielon/MICA/blob/8de9025ee155f208f4a4aed61ade303237f9b1db/models/generator.py#L31

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    
class Expression2WNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, hidden=2):
        super().__init__()

        self.network = nn.ModuleList(
            [nn.Linear(z_dim, map_hidden_dim)] +
            [nn.Linear(map_hidden_dim, map_hidden_dim) for i in range(hidden)]
        )

        self.output = nn.Linear(map_hidden_dim, map_output_dim)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.output.weight *= 0.25

    def forward(self, z):
        h = z
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = Functional.leaky_relu(h, negative_slope=0.2)

        output = self.output(h)
        return output