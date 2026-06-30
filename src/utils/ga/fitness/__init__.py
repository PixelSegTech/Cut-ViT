import torch
import logging
import torch.nn as nn

from .pca import GPUPCA
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List, Callable
from src.utils.ga.loss.losses import mse_loss, cosine_similarity_loss
from src.models.prunable import PrunableModel
from src.utils.models import extract_features
from src.utils.ga.optimizers import FitnessFunction
from src.utils.ga.loss import GALossType, load_ga_loss
import time

def fit_embeddings_pca(
    embeddings: torch.Tensor,
    device: torch.device,
    num_pca_components: int = 64,
    seed: int = 42,
) -> Tuple[torch.Tensor, Optional[GPUPCA]]:
    if num_pca_components < embeddings.shape[1]:
        print("开始计算")
        pca_transform = GPUPCA(
            num_components=num_pca_components,
            device=device,
            seed=seed
        ).fit(embeddings)
        # print(pca_transform)

        logging.info(f"fit pca model using {num_pca_components} components")
        logging.info(f"the original dimensionality is: {embeddings.shape[1]}")
        logging.info(f"the pca explained variance ratio is: {pca_transform.explained_variance_ratio_.sum()}")

        compressed_embeddings = pca_transform.transform(embeddings)

        return compressed_embeddings, pca_transform
    else:
        logging.info("the pca dimensionality is less than the original dimensionality, skipping it.")

        return embeddings, None


'''
以下是对特征进行SVD分解得到投影基
'''
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

def build_fitness_function(
    model: PrunableModel,
    teacher: nn.Module,
    pruning_data_loader: DataLoader,
    eval_data_loader: DataLoader,
    device: torch.device,
    pruning_function: Callable,
    loss_types: List[GALossType],
    num_pca_components: int = 64,
    seed: int = 42,
    estimate_pruning_weights: bool = True
) -> FitnessFunction:
    # Load the loss functions
    loss_functions = []

    for loss_type in loss_types:
        logging.info(f"loading loss function of type: {loss_type}")

        loss_functions.append(load_ga_loss(loss_type))  # 相似度损失

    # We only compute the pruning weights once, as they
    # do not change between optimization steps while using
    # a genetic algorithm
    if estimate_pruning_weights:
        model.estimate_pruning_weights(pruning_data_loader)  # 评估每个参数的梯度
    
    # print("开始计算教师的嵌入")
    # # Precompute the teacher logits, as they are static during the GA optimization
    # teacher_embeddings = extract_features(teacher, eval_data_loader, device, use_fp16=True, move_to_cpu=False)
    # print("计算教师的嵌入已经计算完毕")
    # # compressed_teacher_embeddings = teacher_embeddings

    # start = time.time()
    # teacher_base = SVD(teacher_embeddings)
    # # pca_transform = None

    # # Compress the teacher embeddings using PCA, so that we can compute the MSE
    # # in a more semantic space compared to the raw embeddings

    # # compressed_teacher_embeddings, pca_transform = fit_embeddings_pca(
    # #     embeddings=teacher_embeddings,
    # #     device=device,
    # #     num_pca_components=num_pca_components,
    # #     seed=seed
    # # )  # 把teacher_embeddings将维度到192维度，为主成分分析
    # end = time.time()
    # print(f"Elapsed time: {end - start:.6f} seconds")

    def fitness(solution):
        total_loss = 0

        # Generate a list of pruned models for different pruning ratios.
        # We pass a deep clone of the model to avoid side effects.
        pruned_models = pruning_function(model=model, solution=solution)

        for i, prunable in enumerate(pruned_models):
            # Ensure the pruned model is on the correct device
            prunable = prunable.to(device)

            # Measure the loss using the pruned and teacher embeddings
            with prunable.model.head.set_enabled(enabled=False):
                if hasattr(prunable.model, "head_dist"):
                    with prunable.model.head_dist.set_enabled(enabled=False):
                        pruned_embeddings, _ = extract_features(prunable, eval_data_loader, device, use_fp16=True, move_to_cpu=False)
                else:
                    pruned_embeddings = extract_features(prunable.model, eval_data_loader, device, use_fp16=True, move_to_cpu=False)

            # Compress the pruned embeddings using the same PCA transform before computing the loss
            # if pca_transform is not None:
            #     compressed_pruned_embeddings = pca_transform.transform(pruned_embeddings)
            # else:
            #     compressed_pruned_embeddings = pruned_embeddings
            student_base = SVD(pruned_embeddings)

            # for fn, type_ in zip(loss_functions, loss_types):
            #     loss = fn(
            #         student_projection,
            #         teacher_projection
            #     ) * 1500
            loss_mse = subspace_loss(student_base,teacher_base) 
            # loss_cos = cosine_similarity_loss(compressed_pruned_embeddings,compressed_teacher_embeddings)
            # loss_mse = 0
            loss_cos = 0
            # loss = (loss_mse + loss_cos) / 2.0
            loss = loss_mse

            logging.info(f"the MSE and cosine loss for pruned model {i} is: {loss_mse} and {loss_cos} respectively")

            total_loss +=  loss

        # Average the loss across all pruned models
        total_loss /= len(pruned_models)

        logging.info(f"the total loss is: {total_loss}")

        return total_loss

    return fitness
