import copy
import os.path

import numpy as np
import torch
import time
import sklearn.metrics as m


def get_context_mask(patch_num,
                     ratio: list or tuple):
    min_len, max_len = round(ratio[0] * patch_num), round(ratio[1] * patch_num)
    rand_len = torch.randint(min_len, max_len + 1, (1,))
    rand_start_index = torch.randint(0, (patch_num - rand_len[0]) + 1, (1,))

    mask = torch.zeros(patch_num)
    start = rand_start_index[0]
    end = rand_len[0]
    mask[start:start + end] = 1
    return mask


def get_target_mask(patch_num,
                    ratio: list or tuple,
                    target_num):
    min_len, max_len = round(ratio[0] * patch_num), round(ratio[1] * patch_num)
    rand_len = torch.randint(min_len, max_len + 1, (1,))
    rand_start_index = torch.randint(0, patch_num - rand_len[0], (target_num,))

    masks = []
    for n in range(target_num):
        mask = torch.zeros(patch_num)
        start = rand_start_index[n]
        end = rand_len[0]
        mask[start:start + end] = 1
        masks += [mask]
    return masks


def get_random_masks(patch_num,
                     context_ratio: list or tuple,
                     context_keep,
                     target_ratio: list or tuple,
                     target_num):
    assert target_num * target_ratio[1] < (context_ratio[0] - context_keep), \
        "The target_num * min_target_ratio should < {}.".format(context_ratio[0] - context_keep)
    start_token = time.time()
    context_mask = get_context_mask(patch_num, context_ratio)
    target_masks = get_target_mask(patch_num, target_ratio, target_num)
    for mask in target_masks:
        context_mask -= mask
    context_mask[context_mask > 0] = 1
    context_mask[context_mask < 0] = 0
    context_index = context_mask.argwhere().squeeze()
    target_indexes = []
    for target_mask in target_masks:
        target_indexes += [target_mask.argwhere().squeeze()]
    print("taken {:.2f}s".format(time.time() - start_token))
    return [context_index], target_indexes


def get_batch_masks(batch,
                    patch_num,
                    min_keep,
                    context_ratio: list or tuple,
                    target_ratio: list or tuple,
                    target_num):
    """
    :param batch: batch size.
    :param patch_num: patch num.
    :param min_keep: the minimum size of context mask.
    :param context_ratio: a list, context size ratio in [0, 1].
    :param target_ratio: a list, target size ratio in [0, 1].
    :param target_num: the number of target masks.
    :return: a list [context_mask (batch, c_patch_num), target_masks (batch, target_num, t_patch_num)]
    """
    context_masks = []
    target_masks = []
    min_context_mask_size = patch_num
    min_target_mask_size = patch_num
    for _ in range(batch):
        context_mask = get_context_mask(patch_num, context_ratio)
        valid_masks = []
        for n in range(target_num):
            valid_mask = False
            while not valid_mask:
                target_mask = get_context_mask(patch_num, target_ratio)
                temp_mask = context_mask - target_mask
                temp_mask_size = temp_mask[temp_mask > 0].numel()
                # 如果context mask减去目标mask后剩余区域大于最小区域，则找到一个合适的目标mask
                if temp_mask_size > min_keep:
                    valid_mask = True
                    context_mask = temp_mask
                    target_mask = torch.where(target_mask > 0)[0]
                    target_mask_size = target_mask.numel()
                    valid_masks.append(target_mask)
                    # 记录最小的mask面积
                    min_context_mask_size = min(min_context_mask_size, temp_mask_size)
                    min_target_mask_size = min(min_target_mask_size, target_mask_size)
        context_masks.append(torch.where(context_mask > 0)[0])
        target_masks.append(valid_masks)
    # 修正最小长度，保证所有mask长度相同
    for i in range(len(context_masks)):
        context_masks[i] = context_masks[i][:min_context_mask_size]
        for n in range(target_num):
            target_masks[i][n] = target_masks[i][n][:min_target_mask_size]
        target_masks[i] = torch.stack(target_masks[i], dim=0)
    context_masks = torch.stack(context_masks, dim=0)
    target_masks = torch.stack(target_masks, dim=0)
    return context_masks, target_masks


def get_patch_num(length,
                  patch_size,
                  patch_step):
    return (length - patch_size) // patch_step + 1


def apply_batch_mask(feature: torch.Tensor, mask_index: torch.Tensor):
    """
    :param feature: A original features with shape (Batch, Num_patch, Feature_dim)
    :param mask_index: A mask Tensor with shape (Batch, Keep_num_patch).
    """
    if feature.device != mask_index.device:
        mask_index = mask_index.to(feature.device)
    mask_index = mask_index.unsqueeze(dim=-1).repeat((1, 1, feature.shape[-1]))
    return torch.gather(feature, dim=1, index=mask_index)


def get_batch_continuous_freq_mask(batch,
                                   time_len,
                                   ratio: list or tuple,
                                   target_num):
    assert 0 <= ratio[0] < ratio[1] < 1, "The mask ratio must in the range of [0, 1), but got {}".format(ratio)
    freq_len = time_len // 2 + 1
    masks = torch.ones((batch, target_num, freq_len))
    min_len = round(ratio[0] * freq_len)
    max_len = round(ratio[1] * freq_len)
    for b in range(batch):
        for t in range(target_num):
            rand_len = torch.randint(min_len, max_len, (1,))
            rand_start = torch.randint(0, freq_len - rand_len[0], (1,))
            masks[b, t, rand_start[0]: rand_start[0] + rand_len[0]] = 0
    return masks


def get_batch_discrete_freq_mask(batch,
                                 time_len,
                                 ratio: list or tuple,
                                 target_num):
    assert 0 <= ratio[0] <= ratio[1] < 1, "The mask ratio must in the range of [0, 1), but got {}".format(ratio)
    freq_len = time_len // 2 + 1
    masks = torch.ones((batch, target_num, freq_len))
    constant = torch.ones((freq_len,))
    min_len = round(ratio[0] * freq_len)
    max_len = round(ratio[1] * freq_len)
    for b in range(batch):
        for t in range(target_num):
            rand_len = torch.randint(min_len, max_len + 1, (1,))
            freq_indexes = torch.randperm(freq_len)[:rand_len]
            masks[b, t, :] = torch.scatter(constant[:], dim=-1, index=freq_indexes, value=0)
    return masks


def get_batch_temporal_mask(batch,
                            time_len,
                            ratio: list or tuple,
                            target_num):
    assert 0 <= ratio[0] <= ratio[1] < 1, "The mask ratio must in the range of [0, 1), but got {}".format(ratio)
    masks = torch.ones((batch, target_num, time_len))
    constant = torch.ones((time_len,))
    min_len = round(ratio[0] * time_len)
    max_len = round(ratio[1] * time_len)
    for b in range(batch):
        for t in range(target_num):
            rand_len = torch.randint(min_len, max_len + 1, (1,))
            indexes = torch.randperm(time_len)[:rand_len]
            masks[b, t, :] = torch.scatter(constant[:], dim=-1, index=indexes, value=0)
    return masks


def apply_freq_reduce_mask(x: torch.Tensor, masks: torch.Tensor):
    """
    Apply the mask as a REDUCING mask, which set the target frequency component to 0.

    :param x: A original time series with shape (Batch, Length, Channels)
    :param masks: A mask Tensor with shape (Batch, Target_num, Freq_len).
    """
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    target_num = masks.shape[1]
    fft_x = torch.fft.rfft(x, dim=-2)
    outs = []
    for c in range(C):
        result = copy.deepcopy(fft_x)
        result = result[:, :, c].unsqueeze(dim=1).repeat(1, target_num, 1)
        result *= masks
        outs.append(torch.fft.irfft(result, dim=-1))
    return torch.stack(outs, dim=-1)


def apply_temporal_mask(x: torch.Tensor, masks: torch.Tensor):
    """
    Apply the mask as a REDUCING mask, which set the target frequency component to 0.

    :param x: A original time series with shape (Batch, Length, Channels)
    :param masks: A mask Tensor with shape (Batch, Target_num, Freq_len).
    """
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    target_num = masks.shape[1]
    outs = []
    for c in range(C):
        result = copy.deepcopy(x)
        result = result[:, :, c].unsqueeze(dim=1).repeat(1, target_num, 1)
        result *= masks
        outs.append(result)
    return torch.stack(outs, dim=-1)


def apply_freq_aug_mask(x: torch.Tensor, masks: torch.Tensor):
    """
    Apply the mask as a AUGMENTATION mask, which set the target frequency component to
    the maximum value of the 'x' frequency.

    :param x: A original time series with shape (Batch, Length, Channels)
    :param masks: A mask Tensor with shape (Batch, Target_num, Freq_len).
    """
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    target_num = masks.shape[1]
    masks[:, :, 0] = 1  # The first frequency component will not be augmented.
    # Transfer the mask to the inverse version
    fft_x = torch.fft.rfft(x, dim=-2)  # (Batch, F_L, Channels)
    max_ind = fft_x.abs().max(dim=-2).indices  # (Batch, Channels)
    max_amp = torch.gather(fft_x, -2, max_ind.unsqueeze(dim=-2))  # (Batch, 1, Channels)
    mask_inv = 1 - masks
    outs = []
    mask_amps = []  # The amplitude mask will be returned as the prompt.
    for c in range(C):
        result = copy.deepcopy(fft_x)
        result = result[:, :, c].unsqueeze(dim=1).repeat(1, target_num, 1)  # (Batch, Target_num, Freq_len)
        mask_amp = mask_inv * max_amp[..., c:c + 1]
        result = result * masks + mask_amp  # make the target freq components to be 0
        # mask位置中原本已经是最大振幅的频率处从mask的prompt中去掉
        mask_amp[torch.arange(mask_amp.size(0)).unsqueeze(1),
                 torch.arange(mask_amp.size(1)).unsqueeze(0),
                 max_ind[:, c].unsqueeze(1)] = 0
        mask_amps.append(torch.abs(mask_amp))
        outs.append(torch.fft.irfft(result, dim=-1))
    return torch.stack(outs, dim=-1), torch.stack(mask_amps, dim=-1)


def evaluate_cls_performance(path):
    label = np.load(os.path.join(path, "model_test_labels_part1.npy"))
    pred = np.load(os.path.join(path, "model_test_output_part1.npy"))
    pred = pred.argmax(axis=-1)
    print("Recall: {:.4f}".format(m.recall_score(pred, label, average="macro")))
    print("Precision: {:.4f}".format(m.precision_score(pred, label, average="macro")))
    print("F1: {:.4f}".format(m.f1_score(pred, label, average="macro")))
    print("Accuracy: {:.4f}".format(m.accuracy_score(pred, label)))


if __name__ == '__main__':
    import time

    s = time.time()
    masks = get_batch_temporal_mask(1, 10, [0.5, 0.8], 1)
    print("{:.2f}s".format(time.time() - s))
    values = torch.randn(3, 10, 1)
    out = apply_temporal_mask(values, masks)
