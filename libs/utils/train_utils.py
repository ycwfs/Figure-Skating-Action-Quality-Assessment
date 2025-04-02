import os
import shutil
import time
import pickle
import json

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from scipy.stats import spearmanr


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm, torch.nn.LayerNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            elif pn.endswith('A_log') or pn.endswith("D_b") or pn.endswith("D") or pn.endswith("A_b_log") or pn.endswith("forward_embed") or pn.endswith("backward_embed"):
                # corner case for mamba
                decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 100
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses = model(video_list)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            # torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return

# 定义IoU计算函数
def calculate_iou(segment, label):
    start_seg, end_seg = segment
    start_label, end_label = label

    # 计算交集
    intersection_start = max(start_seg, start_label)
    intersection_end = min(end_seg, end_label)
    intersection = max(0, intersection_end - intersection_start)

    # 计算并集
    union = (end_seg - start_seg) + (end_label - start_label) - intersection

    # 计算IoU
    iou = intersection / union if union > 0 else 0
    return iou

# def covert_res(p):
#     k = list(p.keys())
#     assert len(p[k[0]]) == len(p[k[1]])
#     assert len(p[k[1]]) == len(p[k[2]])
#     assert len(p[k[2]]) == len(p[k[3]])

#     pred_dict = {}
#     for i in range(len(p[k[0]])):
#         video_id = p[k[0]][i]
#         s = p[k[1]][i]
#         e = p[k[2]][i] 
#         c = p[k[3]][i]
#         ps = round(p[k[4]][i]*22, 2)
#         sc = p[k[5]][i]
#         sl = p[k[6]][i]
#         cl = p[k[7]][i]
#         psl = [round(i*22, 2) for i in p[k[8]][i]]
#         pp = round(p[k[9]][i]*100, 2)
#         pl = p[k[10]][i]
#         # if s == e:
#         #     continue
#         if video_id not in pred_dict:
#             pred_dict[video_id] = []
#         pred_dict[video_id].append({
#                 'segments': [s,e],
#                 'class': c,
#                 'pred_score': ps,
#                 'pred_score_labels': psl,
#                 'score': sc,
#                 'seg_labels': sl,
#                 'cls_labels': cl,
#                 'pcs': pp,
#                 'pcs_label': pl
#             })
#     return pred_dict

# def valid_one_epoch(
#     val_loader,
#     model,
#     curr_epoch,
#     ext_score_file = None,
#     evaluator = None,
#     output_file = None,
#     tb_writer = None,
#     cls_ignore = False,
#     print_freq = 100
# ):
#     """Test the model on the validation set"""
#     # either evaluate the results or save the results
#     assert (evaluator is not None) or (output_file is not None)

#     # set up meters
#     batch_time = AverageMeter()
#     # switch to evaluate mode
#     model.eval()
#     # dict for results (for our evaluation code)
#     results = {
#         'video-id': [],
#         't-start' : [],
#         't-end': [],
#         'label': [],
#         'pred_score': [],
#         'score': [],
#         'seg_labels': [],
#         'cls_labels': [],
#         'score_labels': [],
#         'pcs_score': [],
#         'pcs_label': []
#     }
#     result_dict = {}

#     iou_thresholds = np.arange(0.50, 1.0, 0.05)  # 从0.50到0.95，步长为0.05
#     acc = {t: [0] for t in iou_thresholds}
#     acc_class_ignore = {t: [0] for t in iou_thresholds}
#     label_numbers = 0

#     # loop over validation set
#     start = time.time()
#     for iter_idx, video_list in enumerate(val_loader, 0):
#         # forward the model (wo. grad)
#         with torch.no_grad():
#             output = model(video_list)
#             video_id = video_list[0]['video_id']
#             result_dict[video_id] = {}
#             result_dict[video_id]['segments'] = output[0]['segments'].numpy().tolist()
#             result_dict[video_id]['labels'] = output[0]['labels'].numpy().tolist()
#             result_dict[video_id]['element_scores'] = output[0]['element_scores'].numpy().tolist()
#             result_dict[video_id]['pcs'] = output[0]['pcs'].numpy().tolist()
#             result_dict[video_id]['pcs_label'] = output[0]['pcs_label'].numpy().tolist()

#             seg_labels = video_list[0]['segments'].numpy().tolist()
#             cls_labels = video_list[0]['labels'].numpy().tolist()
#             score_labels = video_list[0]['element_scores'].numpy().tolist()
#             pcs_label = video_list[0]['pcs'].numpy().tolist()
#             label_numbers += len(seg_labels)
#             # 对每个样本计算不同IoU阈值下的准确度
#             for iou_threshold in iou_thresholds:
#                 seg_labels = video_list[0]['segments'].numpy().tolist()
#                 cls_labels = video_list[0]['labels'].numpy().tolist()
#                 assert len(seg_labels) == len(cls_labels)
#                 segments = output[0]['segments'].numpy().tolist()
#                 cls_preds = output[0]['labels'].numpy().tolist()
#                 # 遍历每个预测的segment
#                 for idxp,segment in enumerate(segments):
#                     # 遍历每个真实label
#                     for idx,label in enumerate(seg_labels):
#                         iou = calculate_iou(segment, label)
#                         cls_label = cls_labels[idx]
#                         # idx 不一样，wc，又写错了，md，是说结果怎么有问题，要不然有几个index和segment没对上
#                         cls_pred = cls_preds[idxp]
#                         if iou >= iou_threshold:
#                             acc_class_ignore[iou_threshold][0] += 1
#                             if cls_label == cls_pred:
#                                 acc[iou_threshold][0] += 1
#                             seg_labels.remove(label)  # 从seg_labels中删除已经匹配的label
#                             break  # 只要匹配到一个label即可

#             # seg_labels remove before, need to improve the logic
#             seg_labels = video_list[0]['segments'].numpy().tolist()
#             # unpack the results into ANet format
#             num_vids = len(output)
#             for vid_idx in range(num_vids):
#                 if output[vid_idx]['segments'].shape[0] > 1:
#                     results['video-id'].extend(
#                         [output[vid_idx]['video_id']] *
#                         output[vid_idx]['segments'].shape[0]
#                     )
#                     results['seg_labels'].extend(
#                         [seg_labels] *
#                         output[vid_idx]['segments'].shape[0]
#                     )
#                     results['cls_labels'].extend(
#                         [cls_labels] *
#                         output[vid_idx]['segments'].shape[0]
#                     )
#                     results['score_labels'].extend(
#                         [score_labels] *
#                         output[vid_idx]['segments'].shape[0]
#                     )
#                     results['pcs_label'].extend(
#                         [pcs_label] *
#                         output[vid_idx]['segments'].shape[0]
#                     )
#                     results['pcs_score'].extend(
#                         [output[vid_idx]['pcs']] *
#                         output[vid_idx]['segments'].shape[0]
#                     )
#                 else:
#                     results['video-id'].append(output[vid_idx]['video_id'])
#                     results['seg_labels'].append(seg_labels)
#                     results['cls_labels'].append(cls_labels)
#                 results['t-start'].append(output[vid_idx]['segments'][:, 0])
#                 results['t-end'].append(output[vid_idx]['segments'][:, 1])
#                 results['label'].append(output[vid_idx]['labels'])
#                 # aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, wc 先写成score了，我去
#                 results['pred_score'].extend([o.item() for o in output[vid_idx]['pred_score']])
#                 results['score'].append(output[vid_idx]['scores'])

#         # printing
#         if (iter_idx != 0) and iter_idx % (print_freq) == 0:
#             # measure elapsed time (sync all kernels)
#             torch.cuda.synchronize()
#             batch_time.update((time.time() - start) / print_freq)
#             start = time.time()

#             # print timing
#             print('Test: [{0:05d}/{1:05d}]\t'
#                   'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
#                   iter_idx, len(val_loader), batch_time=batch_time))

#     # 计算每个IoU阈值下的平均准确度
#     accs = []; strs = []; accs_class_ignore = []
#     print(f"total segments number: {label_numbers}")
#     print(f"total samples number: {len(val_loader)}")
#     strs.append(f"total segments number: {label_numbers} \n")
#     for iou_threshold in iou_thresholds:
#         avg_accuracy = (acc[iou_threshold][0]/label_numbers) * 100  # 转换为百分比
#         accs.append(avg_accuracy)
#         print(f"|tIoU = {iou_threshold:.2f}: acc_samples: {acc[iou_threshold][0]}, Accuracy = {avg_accuracy:.2f} (%)")
#         strs.append(f"|tIoU = {iou_threshold:.2f}: acc_samples: {acc[iou_threshold][0]}, Accuracy = {avg_accuracy:.2f} (%)")
#     # 计算总平均准确度
#     print(f"average accuracy = {sum(accs)/len(accs):.2f} (%)")
#     strs.append(f"average accuracy = {sum(accs)/len(accs):.2f} (%)\n")

#     print("--------------------cls_ignore--------------------")
#     for iou_threshold in iou_thresholds:
#         cls_ignore_avg_accuracy = (acc_class_ignore[iou_threshold][0]/label_numbers) * 100  # 转换为百分比
#         accs_class_ignore.append(cls_ignore_avg_accuracy)
#         print(f"|tIoU = {iou_threshold:.2f}: acc_samples: {acc_class_ignore[iou_threshold][0]}, Accuracy = {cls_ignore_avg_accuracy:.2f} (%)")
#         strs.append(f"|tIoU = {iou_threshold:.2f}: acc_samples: {acc_class_ignore[iou_threshold][0]}, Accuracy = {cls_ignore_avg_accuracy:.2f} (%)")
#     print(f"average accuracy = {sum(accs_class_ignore)/len(accs_class_ignore):.2f} (%)")
#     strs.append(f"average accuracy = {sum(accs_class_ignore)/len(accs_class_ignore):.2f} (%)\n")

#     # gather all stats and evaluate
#     results['t-start'] = torch.cat(results['t-start']).numpy()
#     results['t-end'] = torch.cat(results['t-end']).numpy()
#     results['label'] = torch.cat(results['label']).numpy()
#     results['score'] = torch.cat(results['score']).numpy()

#     if evaluator is not None:
#         if ext_score_file is not None and isinstance(ext_score_file, str):
#             results = postprocess_results(results, ext_score_file)
#         # call the evaluator
#         _, mAP, _ = evaluator.evaluate(results, verbose=True)
#     else:
#         # dump to a pickle file that can be directly used for evaluation
#         results = covert_res(results)
#         with open(output_file, "wb") as f:
#             pickle.dump(results, f)
#         with open(output_file.split('.')[0] + '.json', 'w') as f1:
#             json.dump(results, f1,indent=2, cls=CustomJSONEncoder)
#         mAP = 0.0

#         pred_scores = []; pred_total_show_score = []; pcs_score = []
#         score_labels = []; total_show_score = []; pcs_labels = []

#         for sample in results:
#             ps = 0
#             for segment in results[sample]:
#                 best_iou = -1
#                 best_index = -1
#                 best_interval = None
#                 seg = segment['segments']
#                 seg_labels = segment['seg_labels']
#                 pred_score = segment['pred_score']
#                 pred_scores.append(pred_score)
#                 pred_score_labels = segment['pred_score_labels']
#                 ps += pred_score
#                 for idx, seg_label in enumerate(seg_labels):
#                     iou = calculate_iou(seg, seg_label)
#                     if iou > best_iou:
#                         best_iou = iou
#                         best_index = idx
#                         best_interval = seg_label
#                 score_labels.append(pred_score_labels[best_index])
#             total_show_score.append(pred_score_labels[0])
#             pred_total_show_score.append(ps)
#             pcs_score.append(segment['pcs'])
#             pcs_labels.append(segment['pcs_label'])
#         print("spearman correlation coefficient between predicted scores and ground truth labels for each action: ", spearmanr(pred_scores, score_labels))
#         strs.append("spearman correlation coefficient between predicted scores and ground truth labels for each actions: " + str(spearmanr(pred_scores, score_labels)))
#         print("spearman correlation coefficient between predicted scores and ground truth labels for each show: ", spearmanr(pred_total_show_score, total_show_score))
#         strs.append("spearman correlation coefficient between predicted scores and ground truth labels for each show: " + str(spearmanr(pred_total_show_score, total_show_score)))
#         print("spearman correlation coefficient between predicted pcs scores and ground truth labels for each show: ", spearmanr(pcs_score, pcs_labels))
#         strs.append("spearman correlation coefficient between predicted pcs scores and ground truth labels for each show: " + str(spearmanr(pcs_score, pcs_labels)))


#     # log mAP to tb_writer
#     if tb_writer is not None:
#         tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

#     return mAP, strs

# def calculate_iou(interval_a, interval_b):
#     a1, a2 = interval_a
#     b1, b2 = interval_b

#     # 计算交集
#     intersection = max(0, min(a2, b2) - max(a1, b1))
#     # 计算并集
#     union = max(a2, b2) - min(a1, b1)
#     # 计算 IoU
#     iou = intersection / union if union > 0 else 0
#     return iou

# # 自定义 JSON 编码器
# class CustomJSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.float32):
#             return float(obj)
#         elif isinstance(obj, np.int64):
#             return int(obj)
#         return super().default(obj)

def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    cls_ignore = False,
    print_freq = 100,
    dataset_name = None
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    
    # dict for storing all results
    result_dict = {}

    iou_thresholds = np.arange(0.50, 1.0, 0.05)  # 从0.50到0.95，步长为0.05
    acc = {t: [0] for t in iou_thresholds}
    acc_class_ignore = {t: [0] for t in iou_thresholds}
    label_numbers = 0

    # For evaluation metrics
    pred_scores = []
    score_labels = []
    pred_total_show_score = []
    total_show_score = []
    pcs_scores = []
    pcs_labels = []

    # loop over validation set
    start = time.time()

    if dataset_name == 'finefs':
        for iter_idx, video_list in enumerate(val_loader, 0):
            # forward the model (wo. grad)
            with torch.no_grad():
                output = model(video_list)
                video_id = video_list[0]['video_id']
                
                # Store all data in result_dict
                result_dict[video_id] = {
                    'pred_segments': [[round(x, 2) for x in segment] for segment in output[0]['segments'].numpy().tolist()],
                    'pred_class': output[0]['labels'].numpy().tolist(),
                    'pred_score': [round(o.item() * 22,2) for o in output[0]['pred_score']],
                    'pred_pcs': round(float(output[0]['pcs']),2),
                    'seg_labels': video_list[0]['segments'].numpy().tolist(),
                    'cls_labels': video_list[0]['labels'].numpy().tolist(),
                    'score_labels': [round(x * 22, 2) for x in video_list[0]['element_scores'].numpy().tolist()],             
                    'pcs_label': round(float(video_list[0]['pcs'].numpy().tolist()),2),
                }

                seg_labels = video_list[0]['segments'].numpy().tolist()
                cls_labels = video_list[0]['labels'].numpy().tolist()
                score_labels_per_video = video_list[0]['element_scores'].numpy().tolist()
                pcs_label_per_video = video_list[0]['pcs'].numpy().tolist()
                
                # Calculate metrics for current video
                pred_score_sum = sum(result_dict[video_id]['pred_score'])
                pred_total_show_score.append(pred_score_sum)
                total_show_score.append(sum(video_list[0]['element_scores'].numpy().tolist()))
                pcs_scores.append(output[0]['pcs'].tolist())
                pcs_labels.append(pcs_label_per_video)
                
                # Add to total segments count
                label_numbers += len(seg_labels)
                
                # Calculate IoU accuracy
                for iou_threshold in iou_thresholds:
                    seg_labels_copy = seg_labels.copy()
                    cls_labels_copy = cls_labels.copy()
                    segments = output[0]['segments'].numpy().tolist()
                    cls_preds = output[0]['labels'].numpy().tolist()
                    
                    # For each predicted segment
                    for idxp, segment in enumerate(segments):
                        # Add to prediction scores collection for correlation
                        pred_score = output[0]['pred_score'].numpy().tolist()[idxp]
                        pred_scores.append(round(pred_score*22, 2))
                        
                        # For each ground truth label
                        best_iou = -1
                        best_idx = -1
                        for idx, label in enumerate(seg_labels_copy):
                            iou = calculate_iou(segment, label)
                            if iou >= iou_threshold and iou > best_iou:
                                best_iou = iou
                                best_idx = idx
                        
                        if best_idx >= 0:
                            acc_class_ignore[iou_threshold][0] += 1
                            if cls_labels_copy[best_idx] == cls_preds[idxp]:
                                acc[iou_threshold][0] += 1
                                                    
                            # Remove matched label to prevent double-counting
                            seg_labels_copy.pop(best_idx)
                            cls_labels_copy.pop(best_idx)
                        # Add to score labels for correlation wheather cls is matched or not
                        score_labels.append(round(score_labels_per_video[best_idx]*22, 2))


            # printing
            if (iter_idx != 0) and iter_idx % (print_freq) == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time() - start) / print_freq)
                start = time.time()

                # print timing
                print('Test: [{0:05d}/{1:05d}]\t'
                    'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                    iter_idx, len(val_loader), batch_time=batch_time))

            # Calculate accuracy metrics
        accs = []; strs = []; accs_class_ignore = []
        print(f"total segments number: {label_numbers}")
        print(f"total samples number: {len(val_loader)}")
        strs.append(f"total segments number: {label_numbers} \n")
        
        for iou_threshold in iou_thresholds:
            avg_accuracy = (acc[iou_threshold][0]/label_numbers) * 100  # 转换为百分比
            accs.append(avg_accuracy)
            print(f"|tIoU = {iou_threshold:.2f}: acc_samples: {acc[iou_threshold][0]}, Accuracy = {avg_accuracy:.2f} (%)")
            strs.append(f"|tIoU = {iou_threshold:.2f}: acc_samples: {acc[iou_threshold][0]}, Accuracy = {avg_accuracy:.2f} (%)")
        # 计算总平均准确度
        print(f"average accuracy = {sum(accs)/len(accs):.2f} (%)")
        strs.append(f"average accuracy = {sum(accs)/len(accs):.2f} (%)\n")

        print("--------------------cls_ignore--------------------")
        for iou_threshold in iou_thresholds:
            cls_ignore_avg_accuracy = (acc_class_ignore[iou_threshold][0]/label_numbers) * 100  # 转换为百分比
            accs_class_ignore.append(cls_ignore_avg_accuracy)
            print(f"|tIoU = {iou_threshold:.2f}: acc_samples: {acc_class_ignore[iou_threshold][0]}, Accuracy = {cls_ignore_avg_accuracy:.2f} (%)")
            strs.append(f"|tIoU = {iou_threshold:.2f}: acc_samples: {acc_class_ignore[iou_threshold][0]}, Accuracy = {cls_ignore_avg_accuracy:.2f} (%)")
        print(f"average accuracy = {sum(accs_class_ignore)/len(accs_class_ignore):.2f} (%)")
        strs.append(f"average accuracy = {sum(accs_class_ignore)/len(accs_class_ignore):.2f} (%)\n")

        # Evaluator is for old result format
        if evaluator is not None:
            # Convert result_dict to old format for evaluator
            results = convert_to_old_format(result_dict)
            if ext_score_file is not None and isinstance(ext_score_file, str):
                results = postprocess_results(results, ext_score_file)
            # call the evaluator
            _, mAP, _ = evaluator.evaluate(results, verbose=True)
        else:
            # Save results to output file
            with open(output_file, "wb") as f:
                pickle.dump(result_dict, f)
            with open(output_file.split('.')[0] + '.json', 'w') as f1:
                json.dump(result_dict, f1, indent=2, cls=CustomJSONEncoder)
            mAP = 0.0

            element_tes_spearman = spearmanr(pred_scores, score_labels); total_tes_spearman = spearmanr(pred_total_show_score, total_show_score); pcs_spearman = spearmanr(pcs_scores, pcs_labels)
            # Calculate correlation metrics
            print("spearman correlation coefficient between predicted scores and ground truth labels for each action: ", element_tes_spearman)
            strs.append("spearman correlation coefficient between predicted scores and ground truth labels for each actions: " + str(element_tes_spearman))
            print("spearman correlation coefficient between predicted scores and ground truth labels for each show: ", total_tes_spearman)
            strs.append("spearman correlation coefficient between predicted scores and ground truth labels for each show: " + str(total_tes_spearman))
            print("spearman correlation coefficient between predicted pcs scores and ground truth labels for each show: ", pcs_spearman)
            strs.append("spearman correlation coefficient between predicted pcs scores and ground truth labels for each show: " + str(pcs_spearman))

        # log mAP to tb_writer
        if tb_writer is not None:
            tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

        return mAP, strs
    else:
        for iter_idx, video_list in enumerate(val_loader, 0):
            with torch.no_grad():
                output = model(video_list)
                video_id = video_list[0]['video_id']
                pcs_label = video_list[0]['pcs_label'].numpy().tolist()
                tes_label = video_list[0]['tes_label'].numpy().tolist()
                pred_elemnet_score = [round(o.item() * 22,2) for o in output[0]['pred_score']]
                pred_tes_score = sum(output[0]['pred_score'])*22
                pred_pcs = float(output[0]['pcs'])
                
                # Store all data in result_dict
                result_dict[video_id] = {
                    'pred_segments': [[round(x, 2) for x in segment] for segment in output[0]['segments'].numpy().tolist()],
                    'pred_class': output[0]['labels'].numpy().tolist(),
                    'pred_elemnet_score': pred_elemnet_score,
                    'pred_tes_score': pred_tes_score,
                    'pred_pcs': round(pred_pcs*100,2),
                    'pcs_label': round(pcs_label,2),
                    'tes_label': round(tes_label,2),
                }
                # pcs_label = round(pcs_label,2)                
                # Calculate metrics for current video
                pred_total_show_score.append(pred_tes_score)
                total_show_score.append(tes_label)
                pcs_scores.append(output[0]['pcs'].tolist()*100)
                pcs_labels.append(pcs_label)

        with open(output_file, "wb") as f:
            pickle.dump(result_dict, f)
        with open(output_file.split('.')[0] + '.json', 'w') as f1:
            json.dump(result_dict, f1, indent=2, cls=CustomJSONEncoder)
        mAP = 0.0

        strs = []
        total_tes_spearman = spearmanr(pred_total_show_score, total_show_score); pcs_spearman = spearmanr(pcs_scores, pcs_labels)
        # Calculate correlation metrics
        print("spearman correlation coefficient between predicted scores and ground truth labels for each show: ", total_tes_spearman)
        strs.append("spearman correlation coefficient between predicted scores and ground truth labels for each show: " + str(total_tes_spearman))
        print("spearman correlation coefficient between predicted pcs scores and ground truth labels for each show: ", pcs_spearman)
        strs.append("spearman correlation coefficient between predicted pcs scores and ground truth labels for each show: " + str(pcs_spearman))

        # log mAP to tb_writer
        if tb_writer is not None:
            tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

        return mAP, strs
                


def convert_to_old_format(result_dict):
    """Convert result_dict to old format for evaluator"""
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'pred_score': [],
        'score': [],
        'seg_labels': [],
        'cls_labels': [],
        'score_labels': [],
        'pcs_score': [],
        'pcs_label': []
    }
    
    for video_id, data in result_dict.items():
        num_segments = len(data['segments'])
        if num_segments > 0:
            results['video-id'].extend([video_id] * num_segments)
            results['seg_labels'].extend([data['seg_labels']] * num_segments)
            results['cls_labels'].extend([data['cls_labels']] * num_segments)
            results['score_labels'].extend([data['score_labels']] * num_segments)
            results['pcs_label'].extend([data['pcs_label']] * num_segments)
            results['pcs_score'].extend([data['pcs']] * num_segments)
            
            # Convert to tensors for concat later
            t_start = torch.tensor([seg[0] for seg in data['segments']])
            t_end = torch.tensor([seg[1] for seg in data['segments']])
            labels = torch.tensor(data['labels'])
            scores = torch.tensor(data['scores'])
            pred_scores = torch.tensor(data['pred_score'])
            
            results['t-start'].append(t_start)
            results['t-end'].append(t_end)
            results['label'].append(labels)
            results['pred_score'].extend([p for p in pred_scores])
            results['score'].append(scores)
    
    # Convert lists of tensors to single tensors
    results['t-start'] = torch.cat(results['t-start'])
    results['t-end'] = torch.cat(results['t-end'])
    results['label'] = torch.cat(results['label'])
    results['score'] = torch.cat(results['score'])
    
    return results

def calculate_iou(interval_a, interval_b):
    a1, a2 = interval_a
    b1, b2 = interval_b

    # 计算交集
    intersection = max(0, min(a2, b2) - max(a1, b1))
    # 计算并集
    union = max(a2, b2) - min(a1, b1)
    # 计算 IoU
    iou = intersection / union if union > 0 else 0
    return iou

# 自定义 JSON 编码器
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return round(float(obj), 2)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif torch.is_tensor(obj):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, float):
            return round(obj, 2)
        return super().default(obj)