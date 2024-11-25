import torch

import torch.nn as nn

class KVPredictor(nn.Module):
    '''    
    KVPredictor is a neural network module that projects key and value tensors from one dimension to another
    across multiple layers. It ensures that the number of layers in the second dimension is a multiple of the 
    number of layers in the first dimension.
    Attributes:
        num_layer_1 (int): Number of layers in the first dimension.
        num_layer_2 (int): Number of layers in the second dimension.
        num_heads_1 (int): Number of attention heads in the first dimension.
        num_heads_2 (int): Number of attention heads in the second dimension.
        dim_heads_1 (int): Dimension of each attention head in the first dimension.
        dim_heads_2 (int): Dimension of each attention head in the second dimension.
        dim_model_1 (int): Total dimension of the model in the first dimension (num_heads_1 * dim_heads_1).
        dim_model_2 (int): Total dimension of the model in the second dimension (num_heads_2 * dim_heads_2).
        proj (dict): Mapping from layers in the second dimension to layers in the first dimension.
        Klayers (nn.ModuleList): List of linear layers for projecting keys.
        Vlayers (nn.ModuleList): List of linear layers for projecting values.
    Methods:
        forward(x):
            Projects the input tensors x from the first dimension to the second dimension.
            Args:
                x (list of tuples): Input tensors of shape [num_layer_1][0/1][batch_size, num_heads_1, num_tokens, dim_heads_1].
            Returns:
                outputs (tuple): Projected tensors of shape [num_layer_2][0/1][batch_size, num_heads_2, num_tokens, dim_heads_2].
    '''
    def __init__(self, num_layer_1, num_layer_2, num_heads_1, num_heads_2, dim_heads_1, dim_heads_2):
        super(KVPredictor, self).__init__()
        self.num_layer_1 = num_layer_1
        self.num_layer_2 = num_layer_2
        self.num_heads_1 = num_heads_1
        self.num_heads_2 = num_heads_2
        self.dim_heads_1 = dim_heads_1
        self.dim_heads_2 = dim_heads_2
        self.dim_model_1 = num_heads_1 * dim_heads_1
        self.dim_model_2 = num_heads_2 * dim_heads_2
        self.proj = {}
        
        r = 0
        assert num_layer_1 <= num_layer_2 , "the case of num_layer_1 > num_layer_2 not implemented yet."
        while (self.num_layer_2 - r) % (self.num_layer_1 - r) != 0: # Find the greatest common divisor
            r+=1
        k = (self.num_layer_2 - r) // (self.num_layer_1 - r)
        for i in range(self.num_layer_1 - r):
            for j in range(k):
                self.proj[ k * i + j ] = i
        for i in range(self.num_layer_2 - r, self.num_layer_2):
            self.proj[i] = i - self.num_layer_2 + self.num_layer_1
        
        # Key, Value layers
        self.Klayers = nn.ModuleList([nn.Linear(self.dim_model_1, self.dim_model_2) for _ in range(self.num_layer_2)])
        self.Vlayers = nn.ModuleList([nn.Linear(self.dim_model_1, self.dim_model_2) for _ in range(self.num_layer_2)])
    
    def forward(self, x):
        # x: [num_layer_1][0/1][batch_size, num_heads_1, num_tokens, dim_heads_1]
        
        
        outputs = ()
        
        for i in range(self.num_layer_2):
            inputsK = x[self.proj[i]][0].reshape(x[self.proj[i]][0].shape[0], x[self.proj[i]][0].shape[2], -1)  # [batch_size, num_tokens, dim_model_1]
            inputsV = x[self.proj[i]][1].reshape(x[self.proj[i]][1].shape[0], x[self.proj[i]][1].shape[2], -1)
            
            Kx = self.Klayers[i](inputsK)
            Kx = Kx.unsqueeze(dim=1).reshape(-1, self.num_heads_2, x[0][0].shape[2], self.dim_heads_2)  # [batch_size, num_heads_2, num_tokens, dim_heads_2]
            
            Ky = self.Vlayers[i](inputsV)
            Ky = Ky.unsqueeze(dim=1).reshape(-1, self.num_heads_2, x[0][0].shape[2], self.dim_heads_2)
            
            outputs += ((Kx, Ky),)
            
        
        return outputs  # [num_layer_1][0/1][batch_size, num_heads_2, num_tokens, dim_heads_2]



def Example():
    # Example usage
    batch_size = 32
    num_tokens = 10
    num_heads_1 = 32
    dim_heads_1 = 128
    num_heads_2 = 8
    dim_heads_2 = 128
    num_layer_1 = 32
    num_layer_2 = 32

    model = KVPredictor(num_layer_1, num_layer_2, num_heads_1, num_heads_2, dim_heads_1, dim_heads_2)
    input_tensor = ()
    for i in range(num_layer_1):
        input_tensor += ((torch.randn(batch_size, num_heads_1, num_tokens, dim_heads_1), torch.randn(batch_size, num_heads_1, num_tokens, dim_heads_1)),)
    output_tensor = model(input_tensor)
    print(output_tensor[0][0].shape)  # Should be [batch_size, num_heads_2, num_tokens, dim_heads_2]

if __name__ == "__main__":
    Example()