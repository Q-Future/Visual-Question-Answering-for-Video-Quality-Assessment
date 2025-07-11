import re
import torch.nn as nn

def slowfast_projector():
    projector_type = "mlp2x_gelu"

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(256,3584 )]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(3584,3584 ))
        return nn.Sequential(*modules)

