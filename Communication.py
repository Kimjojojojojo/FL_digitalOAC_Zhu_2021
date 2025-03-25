import torch
import numpy as np
import random
import math
import Functions

##### modeling communication environment #####
##### outputs are power allocated to optimal, CH and allocated sub-carriers #####
def power_allocation(h_k, Ei_g_th, params):
    # 미리 계산된 상수
    rho_0 = params.P0 / (params.M * Ei_g_th)
    sqrt_rho_0 = math.sqrt(rho_0)  #

    h_abs2 = torch.abs(h_k) ** 2
    mask = h_abs2 >= params.g_th

    # 조건을 만족하면 값 할당, 아니면 0
    p_k = mask.to(h_k.dtype) * sqrt_rho_0  # broadcasting 없이 처리
    return p_k.unsqueeze(-1)



def majority_vote_decoder(g_tilde):
    v = torch.where(g_tilde >= 0, torch.ones_like(g_tilde), -torch.ones_like(g_tilde))
    return v

# Ref by Chen et.al., 2021
def digital_OFDMA(h_k, params):
    K = params.K
    M = params.M

    SNR = params.P0 / (params.B_U * params.N0)
    SINR_kn = torch.abs(h_k) ** 2 * SNR
    m3 = torch.tensor(params.m[3], dtype=torch.float, device=h_k.device)
    p_e = 1 - torch.exp(-m3 / SINR_kn)

    rand_vals = torch.rand(K, device=h_k.device)
    tx_UE_set = (rand_vals > torch.diag(p_e)).float()

    return tx_UE_set
