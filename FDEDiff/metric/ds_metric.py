from Models.ds_torch.ds import eval_ds

def discriminative_score_metrics(ori_data, generated_data, device=0, epoch=25):
    return eval_ds(ori_data, generated_data, device, epoch)
