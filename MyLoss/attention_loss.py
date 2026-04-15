import torch
import torch.nn.functional as F

def attention_loss(attns, heatmap, has_heatmap, header_attention, head_fusion='attn', loss_fn='KL_div'):
    if not has_heatmap:
        device = heatmap.device if isinstance(heatmap, torch.Tensor) else torch.device('cpu')
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    target = heatmap.squeeze()
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

    pred_attn = attns[0].squeeze()
    target = target.to(device=pred_attn.device, dtype=compute_dtype)

    pred_attn = torch.nan_to_num(pred_attn, nan=0.0, neginf=0.0, posinf=1e8)
    target = torch.nan_to_num(target, nan=0.0, neginf=0.0, posinf=1e8)
    pred_attn = pred_attn.clamp_min(1e-8)
    target = target.clamp_min(0.0)

    if target.dim() != pred_attn.dim():
        target = target.unsqueeze(0) if pred_attn.dim() == 2 else target

    pred_sum = pred_attn.sum(dim=-1, keepdim=True) if pred_attn.dim() > 1 else pred_attn.sum()
    target_sum = target.sum(dim=-1, keepdim=True) if target.dim() > 1 else target.sum()
    pred_attn = pred_attn / (pred_sum + 1e-8)
    target = target / (target_sum + 1e-8)

    if loss_fn == 'l1':
        loss = F.l1_loss(pred_attn, target, reduction='sum')
    elif loss_fn == 'KL_div':
        log_pred = torch.log(pred_attn + 1e-8)
        loss = F.kl_div(log_pred, target, reduction='sum')
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn}")

    if not torch.isfinite(loss):
        print("attention_loss became non-finite; returning 0.0 for this sample")
        return torch.zeros((), device=pred_attn.device, dtype=compute_dtype)

    return loss
