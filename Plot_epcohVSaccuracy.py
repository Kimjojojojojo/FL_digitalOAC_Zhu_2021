import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataclasses import dataclass, field
import os
import time

import Functions
import Communication
import FL
import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@dataclass
class CommunicationParams:
    K: int = 10
    M: int = 10
    q: int = 10
    Nb: int = 1
    Nu: int = 1
    g_th: float = 0.3
    P0: int = 11
    N0: float = 10 ** (-174 / 10) * 10 ** (-3)
    B_U: float = 1 * 10 ** 6
    num_bits: int = 16
    num_sym: int = 128
    m_dB: np.ndarray = field(default_factory=lambda: np.array([5.782, 7.083, -0.983, 0.023, -4.401, -4.312]))
    m: np.ndarray = field(init=False)
    sigma: float = 0

    def __post_init__(self):
        self.sigma = self.N0 * self.B_U
        self.m = 10 ** (self.m_dB / 10)

params = CommunicationParams()
subset_size = int(60000 // params.K)
batch = 100
batch_set = [batch] * params.K
num_epochs = 100
learning_rate = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"

central_model_accuracies_signedSGD = []
central_model_accuracies_analogOTA = []
central_model_accuracies_digitalOFDMA = []

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
mod_tmp = ["signedSGD"]

for mode in mod_tmp:
    print(f"\n=== Running Experiment with {mode} ===")
    train_loaders, test_loader = Dataset.MNIST_dataloader(100, batch_set, subset_size, params)
    models = [FL.FNN().to(device) for _ in range(params.K)]
    optimizers = [optim.Adam(models[k].parameters(), lr=learning_rate) for k in range(params.K)]
    criterions = [nn.CrossEntropyLoss() for _ in range(params.K)]
    central_model = FL.FNN().to(device)
    num_weights = central_model.count_parameters()


    central_model_filename = Functions.generate_model_filename(params, mode)
    central_model_save_path = os.path.join(save_dir, central_model_filename)

    # if os.path.exists(central_model_save_path):
    #     central_model.load_state_dict(torch.load(central_model_save_path))
    #     central_model.eval()
    #     print(f"📂 저장된 모델 불러오기 완료: {central_model_save_path}")
    # else:
    #     central_model.load_state_dict(FL.FNN().state_dict())
    #     print("❌ 저장된 모델이 없습니다. 새로운 모델을 학습하세요.")

    accuracies = []
    best_accuracy = 0
    best_model_path = os.path.join(save_dir, f"best_{central_model_filename}")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # epoch_dir = os.path.join(save_dir, central_model_filename.replace(".pth", ""))
        # os.makedirs(epoch_dir, exist_ok=True)
        # epoch_model_path = os.path.join(epoch_dir, f"epoch_{epoch+1}_{central_model_filename}")

        num_joining_data = 0
        total_loss = 0
        g_k_tilde = []
        g_k_vector = []
        total_grad_sum = None
        grad_tmp = []
        shapes = None

        start = time.time()
        for k in range(params.K):
            loss, total_grad, total_grad_sum = FL.local_NN(
                k, train_loaders, models, optimizers, criterions, device, total_grad_sum
            )
            grad_tmp.append(total_grad)
            total_loss += loss
            num_joining_data += batch_set[k]

            g_k_vector_k, tmp_shapes = Functions.vectorization(total_grad)
            if shapes is None:
                shapes = tmp_shapes
            g_k_vector.append(g_k_vector_k)
            g_k_tilde.append(Functions.sign(g_k_vector_k, shapes))
        end = time.time()
        print(f"⏱️ loac NN 학습 시간: {end - start:.4f}초")
        grad_sum = {name: torch.zeros_like(grad) for name, grad in total_grad_sum.items()}
        num_iter = num_weights // params.M
        g_tilde = torch.zeros((num_weights, 1), dtype=torch.float, device=device)

        if mode == "signedSGD":
            start = time.time()
            for iter in range(num_iter):
                #print(f"\n=== Iteration {iter + 1}/{num_iter} ===")

                h_k = Functions.complex_gaussian((params.K, params.M), std=1.0, device=device)
                p_k = Communication.power_allocation(h_k, params)
                g_tilde_t = torch.zeros((params.M, 1), dtype=torch.float, device=device)

                for k in range(params.K):
                    g_k_tilde_Tr_iter = g_k_tilde[k][iter * params.M:(iter + 1) * params.M]
                    g_k_tilde_Tr_k = g_k_tilde_Tr_iter * p_k[k]
                    g_tilde_t += g_k_tilde_Tr_k

                g_tilde[iter * params.M:(iter + 1) * params.M] = g_tilde_t
            end = time.time()
            print(f"⏱️ 통신 시간: {end - start:.4f}초")
            z = params.sigma * torch.zeros((num_weights, 1), dtype=torch.float, device=device)
            v = Communication.majority_vote_decoder(g_tilde + z)
            X = Functions.inv_vectorization(v, shapes)



        elif mode == "analogOTA":
            for iter in range(num_iter):
                h_k = Functions.complex_gaussian((params.K, params.M), std=1.0, device=device)
                p_k = Communication.power_allocation(h_k, params)
                g_t = torch.zeros((params.M, 1), dtype=torch.float, device=device)

                for k in range(params.K):
                    g_k_Tr_iter = g_k_vector[k][iter * params.M:(iter + 1) * params.M]
                    g_k_Tr_k = g_k_Tr_iter * p_k[k]
                    g_t += g_k_Tr_k

                g_tilde[iter * params.M:(iter + 1) * params.M] = g_t

            z = params.sigma * torch.zeros((num_weights, 1), dtype=torch.float, device=device)
            v = g_tilde + z
            X = Functions.inv_vectorization(v, shapes)

        elif mode == "digitalOFDMA":
            for iter in range(num_weights):
                tx_round = int(params.num_bits // np.log2(params.num_sym))
                success_vec = torch.zeros(params.K, dtype=torch.float, device=device)

                for _ in range(tx_round):
                    h_k = Functions.complex_gaussian((params.K, params.M), std=1.0, device=device)
                    tx_UE_set = Communication.digital_OFDMA(h_k, params)
                    success_vec = success_vec + tx_UE_set

                success_vec /= tx_round

                # torch.where 버전으로 바꿔줌
                mask = (success_vec > 0) & (success_vec < 1)
                success_vec = success_vec.masked_fill(mask, 0)

                for k in range(params.K):
                    if success_vec[k] >= 1:
                        for name in grad_tmp[k].keys():
                            grad_sum[name] += grad_tmp[k][name]

            X = grad_sum

        FL.central_NN(num_joining_data, optimizers, central_model, models, learning_rate, X, total_loss, params)

        central_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = central_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'accuracy={accuracy:.2f}%')
        accuracies.append(accuracy)

        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     torch.save(central_model.state_dict(), best_model_path)

    if mode == "signedSGD":
        central_model_accuracies_signedSGD = accuracies
    elif mode == "analogOTA":
        central_model_accuracies_analogOTA = accuracies
    elif mode == "digitalOFDMA":
        central_model_accuracies_digitalOFDMA = accuracies

    torch.save(central_model.state_dict(), central_model_save_path)

if central_model_accuracies_signedSGD:
    plt.plot(range(1, num_epochs + 1), central_model_accuracies_signedSGD, marker='o', linestyle='-', label="Signed SGD")

if central_model_accuracies_analogOTA:
    plt.plot(range(1, num_epochs + 1), central_model_accuracies_analogOTA, marker='s', linestyle='--', label="Analog OTA")

if central_model_accuracies_digitalOFDMA:
    plt.plot(range(1, num_epochs + 1), central_model_accuracies_digitalOFDMA, marker='^', linestyle='-.', label="Digital OFDMA")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim([0, 100])
plt.title("Accuracy Comparison")
plt.legend()
plt.grid()
plt.show()
