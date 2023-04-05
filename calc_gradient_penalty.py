import torch


def calc_gradient_penalty(critic, real_x, generated_x, epsilon, GP_lambda=10):
    """Calculate gradient penalty of critic."""
    epsilon = epsilon.view(-1, 1, 1, 1)
    interpolated_x = epsilon*real_x + (1-epsilon)*generated_x
    interpolated_score = critic(interpolated_x.float())

    gradient = torch.autograd.grad(interpolated_score, interpolated_x, 
                                    grad_outputs=torch.ones_like(interpolated_score), create_graph=True)[0]
    gradient = torch.flatten(gradient, 1)
    # print('normed shape =', torch.norm(gradient, 2, dim=1).shape)
    gradient_penalty = GP_lambda * ((gradient.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty