from Models.ps_torch.ps import eval_ps

def predictive_score_metrics(ori_data, generated_data, device=0, epoch=40):
    return eval_ps(ori_data, generated_data, device, epoch)