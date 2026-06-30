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


def calculate_spatial_svd(embeddings):
    B, T, C = embeddings.shape
    svd_Spatial= embeddings.permute(1, 0, 2).contiguous().reshape(T,B*C)
    Spatial_SVD = svd_Spatial @ svd_Spatial.T   # N × N
    SVD_for_spatial = SVD(Spatial_SVD)
    return SVD_for_spatial

def calculate_channel_svd(embeddings):
    B, T, C = embeddings.shape
    svd_channel = embeddings.permute(2, 1, 0).contiguous().reshape(C,T * B)
    Channel_SVD = svd_channel @ svd_channel.T   # C × C
    SVD_for_Channel = SVD(Channel_SVD)
    return SVD_for_Channel


def subspace_residual(teacher_SVD,Student_SVD,student_spatial):
    residual_loss = calculate_residual(teacher_SVD,student_spatial) * 0.005
    loss_mse_SVD = subspace_loss(Student_SVD,teacher_SVD) * 0.005
    return (residual_loss + loss_mse_SVD) / 2.0

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
                teacher_SVD_spatial = calculate_spatial_svd(embeddings) # N × N
                teacher_SVD_channel = calculate_channel_svd(embeddings) # C 

            student_embeddings = student.get_intermediate_layers(images)[-1]
            embeddings_student = student_embeddings
            B, T, C = embeddings_student.shape
            student_SVD_spatial = calculate_spatial_svd(embeddings_student)
            student_SVD_channel = calculate_channel_svd(embeddings_student)

            # print(Student_SVD.shape,embeddings_student_svd.shape)
            # residual_loss = calculate_residual(teacher_SVD,student_spatial) * 0.005

            # loss_mse_SVD = subspace_loss(Student_SVD,teacher_SVD) * 0.005

            spatial_loss = subspace_residual(teacher_SVD_spatial,student_SVD_spatial,embeddings_student.permute(0, 2, 1).contiguous().reshape(B*C,T))
            channel_loss = subspace_residual(teacher_SVD_channel,student_SVD_channel,embeddings_student.contiguous().reshape(B*T, C))

            cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
            loss_cos_sim = 1.0 - cos_sim.mean()

            print("子空间对齐损失为:%s 原始的特征对齐损失:%s 残差损失为:%s"%(spatial_loss,loss_cos_sim,channel_loss))

            # loss = loss_cos_sim + loss_mse_SVD
            # loss = loss_mse_SVD
            loss = loss_cos_sim + spatial_loss + channel_loss
            loss.backward()
