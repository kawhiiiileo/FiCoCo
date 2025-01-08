import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,CLIPConfig,CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPAttention ,CLIPMLP,CLIPEncoderLayer,CLIPVisionTransformer,CLIPEncoder
# from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
# from tome.utils import parse_r
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
import math
from typing import Callable, Tuple,List

#######################################################
def merge_ficoco_l(
    Merge: Callable, 
    input_embeddings: torch.Tensor, 
    merge_indices: torch.Tensor,
    remain_indices: torch.Tensor,
    merge_targets: torch.Tensor,
    top_values: torch.Tensor,
    reduction_factor:int,
    size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    if size is None:
        size = torch.ones_like(input_embeddings[..., 0, None])

    input_embeddings = Compress(
        input_embeddings * size,  # Scale embeddings by their sizes
        merge_indices,
        remain_indices,
        merge_targets,
        top_values,
        reduction_factor,
        mode="sum"
    )
    size = Compress(size, merge_indices, remain_indices,merge_targets,top_values, reduction_factor,mode="sum")

    return input_embeddings, None

def Filter(
    embeddings: torch.Tensor,
    att_score: torch.Tensor,
    v_tokens: int,
    reduction_factor: int,
    include_class_token: bool = False,
    include_distill_token: bool = False,
) -> Union[Tuple[torch.Tensor, ...], Tuple[Any, ...]]:
    num_protected = int(include_class_token) + int(include_distill_token)
    total_tokens = embeddings.shape[1]
    adjusted_reduction = reduction_factor

    if adjusted_reduction <= 0:
        return embeddings, embeddings

    with torch.no_grad():
        att_score_vt = att_score[..., :v_tokens, v_tokens:]
        att_score_tv = att_score[..., v_tokens:, :v_tokens]
        att_score_vv = att_score[..., :v_tokens, :v_tokens]
        att_score_vv_clone = att_score[..., :v_tokens, :v_tokens]
        att_score_list = [att_score_vv, att_score_vt, att_score_tv]
        att_score_vv = att_score_vv.clone()
        diag_indices = torch.arange(v_tokens)
        att_score_vv[..., diag_indices, diag_indices] = -float('inf')
        dir_text2image = att_score[..., :v_tokens, v_tokens:].mean(dim=-1)
        g = 0.6
        att_score_vv = att_score_vv_clone.sum(dim=-1)
        Scores = g * att_score_vv - (1 - g) * dir_text2image

        if num_protected > 0:
            Scores[..., :num_protected] = -float('inf')

        r = adjusted_reduction
        topk_values, topk_indices = torch.topk(Scores, r, dim=-1)
        merge_idx = topk_indices.unsqueeze(-1)
        dst_idx = topk_indices.unsqueeze(-1)
        batch_size = embeddings.shape[0]
        all_indices = torch.arange(v_tokens, device=embeddings.device).unsqueeze(0).expand(batch_size, -1)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask.scatter_(dim=-1, index=merge_idx.squeeze(-1), value=False)
        remain_idx = all_indices[mask].view(batch_size, -1)

        return merge_idx, remain_idx, att_score_list

def Correlate(
    merge_idx: torch.Tensor,
    att_score_list: torch.Tensor,
    att_score: torch.Tensor,
    r: int
) -> torch.Tensor:
    att_score_vv, att_score_vt, att_score_tv = att_score_list[0], att_score_list[1], att_score_list[2]
    indirect_score = torch.einsum('bij,bjk->bik', att_score_vt, att_score_tv)
    a = 0.6
    Con_scores = a * att_score_vv + (1 - a) * indirect_score
    batch_size, num_tokens, _ = Con_scores.shape
    for i in range(batch_size):
        Con_scores[i].fill_diagonal_(-float('inf'))
    for idx in merge_idx.squeeze(-1):
        att_score_vv[:, idx, idx] = -float('inf')
        indirect_score[:, idx, :] = -float('inf')
        indirect_score[:, :, idx] = -float('inf')
    merge_idx_choose = merge_idx.squeeze(-1)
    choose_att_prob = Con_scores[0, merge_idx_choose[0], :]
    choose_att_prob = choose_att_prob.float()
    threshold_values = torch.quantile(choose_att_prob, 0.998, dim=-1, keepdim=True)
    mask = choose_att_prob > threshold_values
    topk_per_token = mask.sum(dim=-1)
    max_topk = topk_per_token.max().item()
    att_prob = nn.functional.softmax(Con_scores, dim=-1)
    top_values, top_indices = att_prob.topk(max_topk, dim=-1)
    expand_merge_indices = merge_idx.expand(-1, -1, top_indices.shape[-1])
    target_indices = top_indices.gather(dim=-2, index=expand_merge_indices)
    return target_indices, top_values

def Compress(
    x: torch.Tensor,
    merge_idx: torch.Tensor,
    remain_idx: torch.Tensor,
    target_indice: torch.Tensor,
    top_values: torch.Tensor,
    reduction_factor: int,
    mode='mean',
    include_distill_token: bool = False,
) -> torch.Tensor:
        n, t, c = x.shape
        total_score = top_values.sum(dim=-1, keepdim=True)
        merge_weights = top_values / total_score
        merge_indices_choose = merge_idx.squeeze(-1)
        chosen_weights = merge_weights.gather(dim=1, index=merge_indices_choose.unsqueeze(-1).expand(-1, -1, merge_weights.size(-1)))
        merge_tokens = x.gather(dim=-2, index=merge_idx.expand(n, reduction_factor, c))
        weighted_tokens = merge_tokens.unsqueeze(2) * chosen_weights.unsqueeze(-1)
        expanded_tokens = weighted_tokens.reshape(n, -1, c)
        flat_indices = target_indice.reshape(n, -1).unsqueeze(-1).expand(-1, -1, c)
        x = x.scatter_add_(1, flat_indices, expanded_tokens)
        remain_idx = remain_idx.unsqueeze(-1)
        final_embeddings = x.gather(dim=-2, index=remain_idx.expand(n, t - reduction_factor, c))
    return final_embeddings
