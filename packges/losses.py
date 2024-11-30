import torch
import logging
from typing import Union

# COSTANTS
SURPORTED_LOSS_FUNC = ["fro", "nuclear", "svd"] # "mse", "mae", "cos", "kl", 

def loss_fn(
    pred_kv,
    base_kv,
    loss_func: Union[str, list[str]] = "all",
    top_k: int = 0,
) -> dict:

    if loss_func == "all":
        loss_func = SURPORTED_LOSS_FUNC
    loss = {idx:[] for idx in loss_func}
    if "fro" in loss_func:          # Frobenius norm, compare every element in the matrix
        loss["fro"] = torch.norm(pred_kv - base_kv, p='fro').item()
    if "nuclear" in loss_func:    # nuclear norm, sum of the singular values of the matrix
        loss["nuclear"] = torch.norm((pred_kv - base_kv).view(pred_kv.shape[1],-1), p='nuc').item()
    # if "mse" in loss_func:
    #     loss["mse"] = torch.nn.MSELoss(pred_kv, base_kv).item()
    # if  "l1" in loss_func or  "mae" in loss_func:
    #     loss["mae"] = torch.nn.L1Loss(pred_kv, base_kv).item()
    # if "cos" in loss_func:    # cosine similarity,use when the matrix can be seen as vectors
    #     loss["cos"] = torch.nn.CosineSimilarity(pred_kv, base_kv).item()
    # if "kl" in loss_func:     # Kullback-Leibler divergence
    #     loss["kl"] = torch.nn.KLDivLoss(pred_kv, base_kv).item()
    if "svd" in loss_func:    # Singular Value Decomposition
        u1, s1, v1 = torch.svd(pred_kv)
        u2, s2, v2 = torch.svd(base_kv)
        if top_k == 0:
            loss["svd"] = torch.norm(s1 - s2, p='fro').item()
        else:
            logging.debug(f"s.shape : {s1.shape}")
            loss["svd"] = torch.norm(s1[:top_k] - s2[:top_k], p='fro').item()
    return loss

class KV_Pred_losses:
    def __init__(
        self,
        num_layers: int,
        loss_func: Union[str, list[str]] = "norm",
    ):
        self.num_layers = num_layers
        if isinstance(loss_func, str):
            loss_func = [loss_func]
        self.loss_func = loss_func
        self.losses = {idx: [[],[]] for idx in self.loss_func}
    
    def reset(
        self,
    ):
        self.losses = [{idx: [] for idx in self.loss_func} * 2]
    
    def update(
        self,
        losses: dict[list],  # [k_loss, v_loss]
    ):
        for i in range(self.num_layers):
            for func in self.loss_func:
                self.losses[func][0].append(losses[func][0][i])
                self.losses[func][1].append(losses[func][1][i])
    
    def get_loss(
        self,
        mode: str = "avg",
    ):
        avg_loss = {idx: [[],[]] for idx in self.loss_func}
        if mode == "sum":
            return self.losses
        # avg mode default
        for i in range(self.num_layers):
            for func in self.loss_func:
                avg_loss[func][0].append(self.losses[func][0][i] / self.num_layers)
                avg_loss[func][1].append(self.losses[func][1][i] / self.num_layers)
        return avg_loss


class CustomLoss(torch.nn.Module):
    def __init__(self, num_layers, rank = -1):
        super(CustomLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.num_layers = num_layers
        if rank == 0:
            self.rank = -1
        else:
            self.rank = rank
        
    def svd_regularization(self, output, target):
        """
        使用SVD对输出和目标进行正则化。
        :param output: 模型输出
        :param target: 目标
        :return: 使用修改后的SVD重构后的输出和目标
        """
        # 将output1，output2，target1，target2的形状变为[bsz, seq_len, dim*num_heads]

        bsz, num_heads, seq_len, dim = output.shape
        output = output.permute(0, 2, 1, 3).reshape(bsz, seq_len, dim * num_heads).to(torch.float32)
        target = target.permute(0, 2, 1, 3).reshape(bsz, seq_len, dim * num_heads).to(torch.float32)
        
        # 使用SVD分解output和target
        u_o, s_o, v_o = torch.svd(output)
        u_t, s_t, v_t = torch.svd(target)
        
        # 保留最大的rank个奇异值
        s_o = s_o[:,:self.rank].to(torch.bfloat16)
        s_t = s_t[:,:self.rank].to(torch.bfloat16)
        u_o = u_o[:, :, :self.rank].to(torch.bfloat16)
        v_o = v_o[:, :, :self.rank].to(torch.bfloat16)
        u_t = u_t[:, :, :self.rank].to(torch.bfloat16)
        v_t = v_t[:, :, :self.rank].to(torch.bfloat16)
        
        # 构造新的output和target矩阵
        output_reconstructed = torch.matmul(u_o, torch.matmul(torch.diag_embed(s_o), v_o.transpose(-2, -1)))
        target_reconstructed = torch.matmul(u_t, torch.matmul(torch.diag_embed(s_t), v_t.transpose(-2, -1)))
        
        return output_reconstructed, target_reconstructed
    
    def forward(self, model_output, targets, use_svd = False):
        total_loss = 0
        
        for i in range(self.num_layers):
            output1, output2 = model_output[i]  # [bsz, num_heads, seq_len, dim]
            target1, target2 = targets[i]
            
            if use_svd:
                output1, target1 = self.svd_regularization(output1, target1)
                output2, target2 = self.svd_regularization(output2, target2)
            
            # 计算每个输出与目标之间的loss
            loss1 = self.loss_fn(output1, target1)
            loss2 = self.loss_fn(output2, target2)

            # 将两个损失相加
            total_loss += loss1 + loss2
        return total_loss
