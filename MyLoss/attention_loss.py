import torch
import torch.nn.functional as F

# def attention_loss(attns, heatmap, has_heatmap, header_attention, head_fusion='attn', loss_fn='KL_div'):
#     if has_heatmap == 0:
#         device = heatmap.device if isinstance(heatmap, torch.Tensor) else torch.device('cpu')
#         return torch.tensor(0.0, device=device, dtype=torch.float32)

#     target = heatmap.squeeze()
#     if not isinstance(attns, (list, tuple)) or len(attns) == 0:
#         raise ValueError('attns must be a non-empty list or tuple of attention tensors')

#     attn_device = attns[0].device
#     compute_dtype = torch.float32

#     if head_fusion == 'max':
#         max_idx_header = torch.argmax(header_attention)
#         attns = [attn[:, max_idx_header, :, :].to(device=attn_device, dtype=compute_dtype) for attn in attns]
#     elif head_fusion == 'attn':
#         header_attention = torch.nan_to_num(
#             header_attention.to(device=attn_device, dtype=compute_dtype),
#             nan=0.0,
#             neginf=-1e8,
#             posinf=1e8,
#         )
#         header_attn = torch.softmax(header_attention, dim=0)
#         attns = [
#             torch.nan_to_num(
#                 attn.to(device=attn_device, dtype=compute_dtype),
#                 nan=0.0,
#                 neginf=-1e8,
#                 posinf=1e8,
#             )
#             for attn in attns
#         ]
#         attns = [torch.einsum('bijk,i->bjk', attn, header_attn).squeeze(1) for attn in attns]
#     else:
#         attns = [attn.to(device=attn_device, dtype=compute_dtype) for attn in attns]

#     pred_attn = attns[0].squeeze()
#     print("pred_attn shape:", pred_attn.shape)
#     print("target shape:", target.shape)
#     target = target.to(device=pred_attn.device, dtype=compute_dtype)

#     pred_attn = torch.nan_to_num(pred_attn, nan=0.0, neginf=0.0, posinf=1e8)
#     target = torch.nan_to_num(target, nan=0.0, neginf=0.0, posinf=1e8)
#     pred_attn = pred_attn.clamp_min(1e-8)
#     target = target.clamp_min(0.0)

#     if target.dim() != pred_attn.dim():
#         target = target.unsqueeze(0) if pred_attn.dim() == 2 else target

#     pred_sum = pred_attn.sum(dim=-1, keepdim=True) if pred_attn.dim() > 1 else pred_attn.sum()
#     target_sum = target.sum(dim=-1, keepdim=True) if target.dim() > 1 else target.sum()
#     pred_attn = pred_attn / (pred_sum + 1e-8)
#     target = target / (target_sum + 1e-8)

#     if loss_fn == 'l1':
#         loss = F.l1_loss(pred_attn, target, reduction='sum')
#     elif loss_fn == 'KL_div':
#         log_pred = torch.log(pred_attn + 1e-8)
#         loss = F.kl_div(log_pred, target, reduction='sum')
#     else:
#         raise ValueError(f"Unknown loss_fn: {loss_fn}")

#     if not torch.isfinite(loss):
#         print("attention_loss became non-finite; returning 0.0 for this sample")
#         return torch.zeros((), device=pred_attn.device, dtype=compute_dtype)

#     return loss

# def attention_loss(attns, heatmap_0, heatmap_1, has_heatmap_0, has_heatmap_1, header_attention, head_fusion='attn', loss_fn='KL_div'):
#     if has_heatmap_0 == 0 and has_heatmap_1 == 0:
#         device = heatmap_0.device if isinstance(heatmap_0, torch.Tensor) else torch.device('cpu')
#         return torch.tensor(0.0, device=device, dtype=torch.float32)

#     device = heatmap_0.device
#     loss_all = torch.tensor(0.0, device=device, dtype=torch.float32)

#     if not isinstance(attns, (list, tuple)) or len(attns) == 0:
#         raise ValueError('attns must be a non-empty list or tuple of attention tensors')

#     attn_device = attns[0].device
#     compute_dtype = torch.float32

#     if head_fusion == 'max':
#         max_idx_header = torch.argmax(header_attention)
#         attns = [attn[:, max_idx_header, :, :].to(device=attn_device, dtype=compute_dtype) for attn in attns]
#     elif head_fusion == 'attn':
#         header_attention = torch.nan_to_num(
#             header_attention.to(device=attn_device, dtype=compute_dtype),
#             nan=0.0,
#             neginf=-1e8,
#             posinf=1e8,
#         )
#         header_attn = torch.softmax(header_attention, dim=0)
#         attns = [
#             torch.nan_to_num(
#                 attn.to(device=attn_device, dtype=compute_dtype),
#                 nan=0.0,
#                 neginf=-1e8,
#                 posinf=1e8,
#             )
#             for attn in attns
#         ]
#         attns = [torch.einsum('bijk,i->bjk', attn, header_attn).squeeze(1) for attn in attns]
#     else:
#         attns = [attn.to(device=attn_device, dtype=compute_dtype) for attn in attns]

#     if has_heatmap_0 > 0:

#         target_0 = heatmap_0.squeeze()

#         pred_attn_0 = attns[0].squeeze()
#         target_0 = target_0.to(device=pred_attn_0.device, dtype=compute_dtype)

#         pred_attn_0 = torch.nan_to_num(pred_attn_0, nan=0.0, neginf=0.0, posinf=1e8)
#         target_0 = torch.nan_to_num(target_0, nan=0.0, neginf=0.0, posinf=1e8)
#         pred_attn_0 = pred_attn_0.clamp_min(1e-8)
#         target_0 = target_0.clamp_min(0.0)

#         if target_0.dim() != pred_attn_0.dim():
#             target_0 = target_0.unsqueeze(0) if pred_attn_0.dim() == 2 else target_0

#         pred_sum_0 = pred_attn_0.sum(dim=-1, keepdim=True) if pred_attn_0.dim() > 1 else pred_attn_0.sum()
#         target_sum_0 = target_0.sum(dim=-1, keepdim=True) if target_0.dim() > 1 else target_0.sum()
#         pred_attn_0 = pred_attn_0 / (pred_sum_0 + 1e-8)
#         target_0 = target_0 / (target_sum_0 + 1e-8)

#         if loss_fn == 'l1':
#             loss_all += F.l1_loss(pred_attn_0, target_0, reduction='sum')
#         elif loss_fn == 'KL_div':
#             log_pred_0 = torch.log(pred_attn_0 + 1e-8)
#             loss_all += F.kl_div(log_pred_0, target_0, reduction='sum')
#         else:
#             raise ValueError(f"Unknown loss_fn: {loss_fn}")


#     if has_heatmap_1 > 0:

#         target_1 = heatmap_1.squeeze()

#         pred_attn_1 = attns[1].squeeze()
#         target_1 = target_1.to(device=pred_attn_1.device, dtype=compute_dtype)

#         pred_attn_1 = torch.nan_to_num(pred_attn_1, nan=0.0, neginf=0.0, posinf=1e8)
#         target_1 = torch.nan_to_num(target_1, nan=0.0, neginf=0.0, posinf=1e8)
#         pred_attn_1 = pred_attn_1.clamp_min(1e-8)
#         target_1 = target_1.clamp_min(0.0)

#         if target_1.dim() != pred_attn_1.dim():
#             target_1 = target_1.unsqueeze(0) if pred_attn_1.dim() == 2 else target_1

#         pred_sum_1 = pred_attn_1.sum(dim=-1, keepdim=True) if pred_attn_1.dim() > 1 else pred_attn_1.sum()
#         target_sum_1 = target_1.sum(dim=-1, keepdim=True) if target_1.dim() > 1 else target_1.sum()
#         pred_attn_1 = pred_attn_1 / (pred_sum_1 + 1e-8)
#         target_1 = target_1 / (target_sum_1 + 1e-8)

#         if loss_fn == 'l1':
#             loss_all += F.l1_loss(pred_attn_1, target_1, reduction='sum')
#         elif loss_fn == 'KL_div':
#             log_pred_1 = torch.log(pred_attn_1 + 1e-8)
#             loss_all += F.kl_div(log_pred_1, target_1, reduction='sum')
#         else:
#             raise ValueError(f"Unknown loss_fn: {loss_fn}")

#     if not torch.isfinite(loss_all):
#         print("attention_loss became non-finite; returning 0.0 for this sample")
#         return torch.zeros((), device=loss_all.device, dtype=compute_dtype)

#     return loss_all


def attention_loss(
    attns,
    heatmap_0,
    heatmap_1,
    has_heatmap_0,
    has_heatmap_1,
    header_attention,
    head_fusion='attn',
    loss_fn='combined',
    alpha=5.0,
    beta=1.0,
    gamma=1.0,
    margin=0.1,
    rank_weight=1.0,
):
    if has_heatmap_0 == 0 and has_heatmap_1 == 0:
        device = heatmap_0.device if isinstance(heatmap_0, torch.Tensor) else torch.device('cpu')
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    device = heatmap_0.device
    loss_all = torch.tensor(0.0, device=device, dtype=torch.float32)

    if not isinstance(attns, (list, tuple)) or len(attns) == 0:
        raise ValueError('attns must be a non-empty list or tuple of attention tensors')

    attn_device = attns[0].device
    compute_dtype = torch.float32

    if head_fusion == 'max':
        max_idx_header = torch.argmax(header_attention)
        attns = [attn[:, max_idx_header, :, :].to(device=attn_device, dtype=compute_dtype) for attn in attns]
    elif head_fusion == 'attn':
        header_attention = torch.nan_to_num(
            header_attention.to(device=attn_device, dtype=compute_dtype),
            nan=0.0,
            neginf=-1e8,
            posinf=1e8,
        )
        header_attn = torch.softmax(header_attention, dim=0)
        attns = [
            torch.nan_to_num(
                attn.to(device=attn_device, dtype=compute_dtype),
                nan=0.0,
                neginf=-1e8,
                posinf=1e8,
            )
            for attn in attns
        ]
        attns = [torch.einsum('bijk,i->bjk', attn, header_attn).squeeze(1) for attn in attns]
    else:
        attns = [attn.to(device=attn_device, dtype=compute_dtype) for attn in attns]

    def compute_rank_loss(pred_attn, target):
        pred_flat = pred_attn.float().reshape(-1)
        target_flat = target.float().reshape(-1)

        high_idx = target_flat > torch.quantile(target_flat, 0.80)
        low_idx = target_flat < torch.quantile(target_flat, 0.20)

        if not high_idx.any() or not low_idx.any():
            return torch.zeros((), device=pred_attn.device, dtype=torch.float32)

        A_high = pred_flat[high_idx].mean()
        A_low = pred_flat[low_idx].mean()

        eps = torch.finfo(A_low.dtype).eps
        ratio = A_high / A_low.clamp_min(eps)

        print("high count", high_idx.sum().item(), "low count", low_idx.sum().item())
        print("A_high", A_high.item(), "A_low", A_low.item(), "ratio", (A_high / A_low.clamp_min(1e-8)).item())


        rank_loss = F.relu(2.0 - ratio)
        return rank_loss

    # def compute_rank_loss(pred_attn, target):
    #     pred_flat = pred_attn.reshape(-1)
    #     target_flat = target.reshape(-1)

    #     high_idx = target_flat > torch.quantile(target_flat, 0.7)
    #     low_idx = target_flat < torch.quantile(target_flat, 0.3)

    #     if not high_idx.any() or not low_idx.any():
    #         return torch.zeros((), device=pred_attn.device, dtype=pred_attn.dtype)

    #     # print(pred_flat.max())
    #     # print(pred_flat.min())
    #     # print(pred_flat.mean())

    #     # print(pred_flat[high_idx].mean())
    #     # print(pred_flat[low_idx].mean())

    #     # return F.relu(
    #     #     margin
    #     #     - pred_flat[high_idx].mean()
    #     #     + pred_flat[low_idx].mean()
    #     # )

    #     high_idx = target_flat > torch.quantile(target_flat, 0.90)
    #     low_idx  = target_flat < torch.quantile(target_flat, 0.10)

    #     A_high = pred_flat[high_idx].mean()
    #     A_low = pred_flat[low_idx].mean()

    #     print("high count", high_idx.sum().item(), "low count", low_idx.sum().item())
    #     print("A_high", A_high.item(), "A_low", A_low.item(), "ratio", (A_high / A_low.clamp_min(1e-8)).item())

    #     rank_loss = F.relu(2.0 - A_high / A_low.clamp_min(1e-4))

    #     return rank_loss

    def compute_attn_loss(pred_attn, target):
        rank = compute_rank_loss(pred_attn, target)
        if loss_fn == 'l1':
            return F.l1_loss(pred_attn, target, reduction='sum')
        if loss_fn == 'KL_div':
            log_pred = torch.log(pred_attn + 1e-8)
            return alpha*F.kl_div(log_pred, target, reduction='sum')
        if loss_fn == 'rank':
            return rank_weight*rank
        if loss_fn == 'combined':
            log_pred = torch.log(pred_attn + 1e-8)
            kl = F.kl_div(log_pred, target, reduction='sum')
            l1 = F.l1_loss(pred_attn, target, reduction='sum')
            cosine = F.cosine_similarity(
                pred_attn.reshape(1, -1),
                target.reshape(1, -1),
                dim=1,
                eps=1e-8,
            ).squeeze()
            print(
                f"kl: {kl.item():.4f}, l1: {l1.item():.4f}, "
                f"cosine_similarity: {cosine.item():.4f}, rank: {rank.item():.4f}"
            )
            return alpha * kl + beta * l1 + gamma * (1.0 - cosine) + rank_weight * rank

            # print(
            #     f"kl: {kl.item():.4f}, "
            #     f"rank: {rank.item():.4f}"
            # )
            # return alpha * kl + rank_weight * rank
        raise ValueError(f"Unknown loss_fn: {loss_fn}")

    if has_heatmap_0 > 0:

        target_0 = heatmap_0.squeeze()

        pred_attn_0 = attns[0].squeeze()
        target_0 = target_0.to(device=pred_attn_0.device, dtype=compute_dtype)

        pred_attn_0 = torch.nan_to_num(pred_attn_0, nan=0.0, neginf=0.0, posinf=1e8)
        target_0 = torch.nan_to_num(target_0, nan=0.0, neginf=0.0, posinf=1e8)
        pred_attn_0 = pred_attn_0.clamp_min(1e-8)
        target_0 = target_0.clamp_min(0.0)

        if target_0.dim() != pred_attn_0.dim():
            target_0 = target_0.unsqueeze(0) if pred_attn_0.dim() == 2 else target_0

        pred_sum_0 = pred_attn_0.sum(dim=-1, keepdim=True) if pred_attn_0.dim() > 1 else pred_attn_0.sum()
        target_sum_0 = target_0.sum(dim=-1, keepdim=True) if target_0.dim() > 1 else target_0.sum()
        pred_attn_0 = pred_attn_0 / (pred_sum_0 + 1e-8)
        target_0 = target_0 / (target_sum_0 + 1e-8)

        loss_all += compute_attn_loss(pred_attn_0, target_0)


    if has_heatmap_1 > 0:

        target_1 = heatmap_1.squeeze()

        pred_attn_1 = attns[1].squeeze()
        target_1 = target_1.to(device=pred_attn_1.device, dtype=compute_dtype)

        pred_attn_1 = torch.nan_to_num(pred_attn_1, nan=0.0, neginf=0.0, posinf=1e8)
        target_1 = torch.nan_to_num(target_1, nan=0.0, neginf=0.0, posinf=1e8)
        pred_attn_1 = pred_attn_1.clamp_min(1e-8)
        target_1 = target_1.clamp_min(0.0)

        if target_1.dim() != pred_attn_1.dim():
            target_1 = target_1.unsqueeze(0) if pred_attn_1.dim() == 2 else target_1

        pred_sum_1 = pred_attn_1.sum(dim=-1, keepdim=True) if pred_attn_1.dim() > 1 else pred_attn_1.sum()
        target_sum_1 = target_1.sum(dim=-1, keepdim=True) if target_1.dim() > 1 else target_1.sum()
        pred_attn_1 = pred_attn_1 / (pred_sum_1 + 1e-8)
        target_1 = target_1 / (target_sum_1 + 1e-8)

        loss_all += compute_attn_loss(pred_attn_1, target_1)

    if not torch.isfinite(loss_all):
        print("attention_loss became non-finite; returning 0.0 for this sample")
        return torch.zeros((), device=loss_all.device, dtype=compute_dtype)

    return loss_all
