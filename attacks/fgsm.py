import torch

from train.loss import captcha_loss


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    FGSM perturbation with L_inf budget epsilon in [0, 1] pixel scale.

    x_adv = clip(x + epsilon * sign(grad_x L), 0, 1)
    """
    model.eval()
    adv_images = images.clone().detach().requires_grad_(True)
    logits = model(adv_images)
    loss = captcha_loss(logits, targets)
    model.zero_grad(set_to_none=True)
    loss.backward()
    perturbation = epsilon * adv_images.grad.sign()
    adv = torch.clamp(adv_images + perturbation, 0.0, 1.0)
    return adv.detach()
