import numpy as np
import scipy.integrate as integrate
import torch
import hashlib
# Define the function to integrate with lower limit k
# ✅ Define the function to integrate with lower limit k (torch 버전)
def exponential_integration(k, steps=100000):
    # 단순한 사다리꼴 적분법으로 torch 버전 구현
    x = torch.linspace(k, 100.0, steps)  # 무한대를 대체하는 큰 값 (100 이상이면 충분)
    dx = x[1] - x[0]
    y = (1 / x) * torch.exp(-x)
    result = torch.trapz(y, dx=dx)
    return result.item(), 0  # torch에선 에러 추정치 없음


# total_grad_sum을 sign 벡터로 변환하는 함수
def vectorization(total_grad_sum):
    shapes = {name: grad.shape for name, grad in total_grad_sum.items()}  # 각 gradient의 shape 저장
    grad_values = [grad.view(-1) for grad in total_grad_sum.values()]
    grad_vector = torch.cat(grad_values).view(-1, 1)
    signed_vector = torch.sign(grad_vector)  # sign 적용
    vector_length = signed_vector.numel()  # 벡터의 길이 계산

    #print(f"Signed Vector Length: {vector_length}")  # 벡터의 길이 출력
    return grad_vector, shapes
 # sign 벡터와 원래 shape 반환

def sign(vector, shapes):
    signed_vector = torch.sign(vector)  # sign 적용
    vector_length = signed_vector.numel()  # 벡터의 길이 계산

    #print(f"Signed Vector Length: {vector_length}")  # 벡터의 길이 출력
    return signed_vector
# sign 벡터를 원래 구조로 복원하는 함수
def inv_vectorization(signed_vector, shapes):
    torch_vector = signed_vector
    restored_dict = {}
    idx = 0
    for name, shape in shapes.items():
        num_elements = torch.prod(torch.tensor(shape))  # 요소 개수
        restored_dict[name] = torch_vector[idx:idx + num_elements].view(shape)  # 원래 shape로 변환
        idx += num_elements

    return restored_dict

def complex_gaussian(shape, device, std=1.0):
    real = torch.randn(shape, dtype=torch.float, device=device) * std
    imag = torch.randn(shape, dtype=torch.float, device=device) * std
    return (real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=device))


def generate_model_filename(params, mode):
    """ 하이퍼파라미터를 기반으로 해시값 생성 후 파일명 반환 """
    param_string = f"K{params.K}_M{params.M}_q{params.q}_Nb{params.Nb}_Nu{params.Nu}_gth{params.g_th}_P0{params.P0}_N0{params.N0}_BU{params.B_U}_bits{params.num_bits}"
    hash_value = hashlib.md5(param_string.encode()).hexdigest()[:8]  # 해시 생성 (앞 8자리만 사용)
    #print(hash_value)
    return f"model_{hash_value}.pth"  # 모델 파일명 반환
# 예제 total_grad_sum (가상의 gradient 값)
# total_grad_sum = {
#     "layer1.weight": torch.tensor([[0.5, -0.3], [0.2, -0.7]]),
#     "layer1.bias": torch.tensor([0.1, -0.4]),
# }
#
# # Step 1: Sign 벡터 변환
# signed_grad_vector, shapes = extract_and_sign(total_grad_sum)
# print("Sign 벡터:\n", shapes)
#
# # Step 2: 원래 shape로 복원
# restored_total_grad_sum = restore_original_shape(signed_grad_vector, shapes)
# print("\n복원된 total_grad_sum:")
# for name, tensor in restored_total_grad_sum.items():
#     print(f"{name}: \n{tensor}")

