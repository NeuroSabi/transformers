import transformers
import torch
from modular_principle import PrincipleMLP
from configuration_principle import PrincipleConfig

config = PrincipleConfig(
    layer_sizes=[4096, 4096, 4096, 4096, 4096],
    final_out_size=4096
)
print(config.layer_sizes)
print(config.final_out_size)

principle_mlp_layer = PrincipleMLP(config, 0)
batch_size = 32
seq_len = 15
input_tensor = torch.randn((batch_size, seq_len, 4096), dtype=torch.float32)
principle_mlp_layer(input_tensor)
