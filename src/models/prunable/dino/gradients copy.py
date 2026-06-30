import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader

from src.models.prunable.base import PrunableModel
import torch.nn.functional as F


def SVD(embedding, k = 192):
    _, _, Vt = torch.svd_lowrank(embedding, q=k, niter=1)
    # print(Vt.shape)
    # Pt = Vt @ Vt.T
    return Vt


def subspace_loss(VA, VB):
    k = VA.shape[1]
    M = VA.T @ VB
    sim = torch.norm(M, p='fro') ** 2
    return (2 * k - 2 * sim) / (2 * k)

def calculate_residual(teacher_SVD,embeddings_student_svd):
    feature_template = embeddings_student_svd @ teacher_SVD
    feature_template_res = feature_template @ teacher_SVD.T
    residual = embeddings_student_svd - feature_template_res
    return (residual ** 2).mean() / residual.numel()


def estimate_dino_gradients(
    student: Union[nn.Module, PrunableModel],
    teacher: Union[nn.Module, PrunableModel],
    device: torch.device,
    data_loader: DataLoader,
    estimation_epochs: int = 1,
    skip_same_view: bool = True
):
    student.zero_grad()

    for _ in range(estimation_epochs):
        for images  in tqdm(data_loader):
            images = images.to(device)

            # Select the global and local views (i.e. the first two crops). We first
            # embed the images using the backbone and then project them using either
            # the student or teacher heads.
            B,  C, H, W = images.shape


            with torch.no_grad():
                teacher_embeddings = teacher.get_intermediate_layers(images)[-1]
                embeddings = teacher_embeddings
                B, T, C = embeddings.shape
                embeddings_teacher_svd = embeddings.reshape(B, T * C)
                teacher_SVD = SVD(embeddings_teacher_svd)

            student_embeddings = student.get_intermediate_layers(images)[-1]
            embeddings_student = student_embeddings
            B, T, C = embeddings_student.shape
            embeddings_student_svd = embeddings_student.reshape(B, T * C)
            Student_SVD = SVD(embeddings_student_svd)
            residual_loss = calculate_residual(teacher_SVD,embeddings_student_svd) * 0.005

            loss_mse_SVD = subspace_loss(Student_SVD,teacher_SVD) * 0.005

            cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
            loss_cos_sim = 1.0 - cos_sim.mean()

            print("子空间对齐损失为:%s 原始的特征对齐损失:%s 残差损失为:%s"%(loss_mse_SVD,loss_cos_sim,residual_loss))

            loss = loss_cos_sim + residual_loss + loss_mse_SVD
            loss.backward()
