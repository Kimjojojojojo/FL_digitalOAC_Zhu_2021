import torch
import numpy as np
import random
import Functions

##### modeling communication environment #####
##### outputs are power allocated to optimal, CH and allocated sub-carriers #####
def power_allocation(h_k, params):
    Ei_g_th, _ = Functions.exponential_integration(params.g_th)
    rho_0 = params.P0 / (params.M * Ei_g_th)  # scaling factor

    rho_0_tensor = torch.tensor(rho_0, dtype=torch.float, device=h_k.device)
    sqrt_rho_0 = torch.sqrt(rho_0_tensor)

    p_k = torch.zeros((params.K, params.M, 1), dtype=torch.float, device=h_k.device)

    h_abs2 = torch.abs(h_k) ** 2
    mask = h_abs2 >= params.g_th
    p_k = torch.where(mask.unsqueeze(-1), sqrt_rho_0, torch.tensor(0.0, device=h_k.device))

    return p_k


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
