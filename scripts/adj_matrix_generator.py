import torch
from torch.nn.functional import cosine_similarity

def full_connected_adj_matrix(num_nodes):
    """
    生成全连接图邻接矩阵
    :param num_nodes: 节点数量
    :return: 全连接图邻接矩阵
    """
    adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
    return adj_matrix

# def similarity_based_adj_matrix(encoded_vectors, threshold=0.5):
#     """
#     基于相似度生成邻接矩阵
#     :param encoded_vectors: 编码后的向量，形状为 (num_nodes, num_features)
#     :param threshold: 相似度阈值
#     :return: 基于相似度的邻接矩阵
#     """
#     num_nodes = encoded_vectors.shape[0]
#     # 计算余弦相似度矩阵
#     similarity_matrix = torch.zeros((num_nodes, num_nodes))
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             similarity_matrix[i, j] = cosine_similarity(encoded_vectors[i].unsqueeze(0), encoded_vectors[j].unsqueeze(0))

#     # 根据阈值生成邻接矩阵
#     adj_matrix = (similarity_matrix > threshold).float()
#     # 对角线元素设为 0
#     adj_matrix = adj_matrix - torch.eye(num_nodes)
#     return adj_matrix

import torch
import torch.nn.functional as F

def similarity_based_adj_matrix(encoded_vectors, threshold=0.5):
    """
    基于相似度生成邻接矩阵
    :param encoded_vectors: 编码后的向量，形状为 (num_nodes, num_features)
    :param threshold: 相似度阈值
    :return: 基于相似度的邻接矩阵
    """
    num_nodes = encoded_vectors.shape[0]
    
    # 归一化每个向量
    normed_vectors = F.normalize(encoded_vectors, p=2, dim=1)
    
    # 计算所有节点之间的余弦相似度，使用矩阵乘法
    similarity_matrix = torch.matmul(normed_vectors, normed_vectors.T)  # 计算余弦相似度矩阵
    
    # 根据阈值生成邻接矩阵
    adj_matrix = (similarity_matrix > threshold).float()
    
    # 对角线元素设为 0（即节点与节点自己不相连）
    adj_matrix = adj_matrix - torch.eye(num_nodes).to(adj_matrix.device)
    
    return adj_matrix


def knn_adj_matrix(encoded_vectors, k=3):
    """
    基于 K 近邻生成邻接矩阵
    :param encoded_vectors: 编码后的向量，形状为 (num_nodes, num_features)
    :param k: K 值
    :return: K 近邻邻接矩阵
    """
    num_nodes = encoded_vectors.shape[0]
    # 计算余弦相似度矩阵
    similarity_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            similarity_matrix[i, j] = cosine_similarity(encoded_vectors[i].unsqueeze(0), encoded_vectors[j].unsqueeze(0))

    # 初始化邻接矩阵
    adj_matrix = torch.zeros((num_nodes, num_nodes))

    # 为每个节点选择 K 个最近邻
    for i in range(num_nodes):
        _, indices = torch.topk(similarity_matrix[i], k + 1)  # 加 1 是因为包含自身
        adj_matrix[i, indices[1:]] = 1  # 排除自身

    # 对角线元素设为 0
    adj_matrix = adj_matrix - torch.eye(num_nodes)
    return adj_matrix
