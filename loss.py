import torch
import torch.nn.functional as F


def ibot_patch_loss(student_patch_tokens, teacher_patch_tokens, student_temp=1.0, teacher_temp=1.0):
    """
    Compute cross-entropy between softmax outputs of teacher and student networks.

    Args:
        student_patch_tokens (torch.Tensor): shape (B, N, D)
        teacher_patch_tokens (torch.Tensor): shape (B, N, D)
        student_temp (float): temperature for student logits
        teacher_temp (float): temperature for teacher logits

    Returns:
        torch.Tensor: scalar loss
    """
    t = teacher_patch_tokens
    s = student_patch_tokens

    softmax_teacher = F.softmax(t / teacher_temp, dim=-1)
    log_probs = F.log_softmax(s / student_temp, dim=-1)
    loss = torch.sum(softmax_teacher * log_probs, dim=-1)  # shape: (B, N)
    return -loss.mean()

