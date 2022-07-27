import datetime
import os

import torch
import wandb


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def save_model(model, model_path, model_name='best_model.pth'):
    fn_out = os.path.join(model_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), fn_out)
    print(rf"Saved new best model to: {fn_out}")


def get_model_name(log_progress):
    if log_progress:
        model_name = wandb.run.name
    else:
        model_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return model_name


def unit_vector(vector):
    vector = torch.as_tensor(vector, dtype=torch.float)
    return vector / torch.norm(vector)


def angle_between(v1, v2, degrees=False):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = torch.arccos(torch.clip(torch.dot(v1_u, v2_u), -1.0, 1.0))
    if degrees:
        angle = angle / torch.pi * 180
    return angle
