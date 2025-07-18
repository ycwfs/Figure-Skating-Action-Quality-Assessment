# Functions for 1D NMS, modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import torch

import nms_1d_cpu


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs, pred_score,
        iou_threshold, min_score, max_num
    ):
        # vanilla nms will not change the score, so we can filter segs first
        is_filtering_by_score = (min_score > 0)
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        # nms op; return inds that is sorted by descending order
        inds = nms_1d_cpu.nms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            iou_threshold=float(iou_threshold))
        # cap by max number
        if max_num > 0:
            inds = inds[:min(max_num, len(inds))]
        # return the sorted segs / scores
        sorted_segs = segs[inds]
        sorted_scores = scores[inds]
        sorted_cls_idxs = cls_idxs[inds]
        sorted_pred_score = pred_score[inds]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone(), sorted_pred_score.clone()


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs, pred_score,
        iou_threshold, sigma, min_score, method, max_num
    ):
        # pre allocate memory for sorted results
        dets = segs.new_empty((segs.size(0), 3), device='cpu')
        # softnms op, return dets that stores the sorted segs / scores
        inds = nms_1d_cpu.softnms(
            segs.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method))
        # cap by max number
        if max_num > 0:
            n_segs = min(len(inds), max_num)
        else:
            n_segs = len(inds)
        sorted_segs = dets[:n_segs, :2]
        sorted_scores = dets[:n_segs, 2]
        sorted_cls_idxs = cls_idxs[inds]
        sorted_cls_idxs = sorted_cls_idxs[:n_segs]
        sorted_pred_score = pred_score[inds]
        sorted_pred_score = sorted_pred_score[:n_segs]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone(), sorted_pred_score.clone()


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    """
        blur localization results by incorporating side segs.
        this is known as bounding box voting in object detection literature.
        slightly boost the performance around iou_threshold
        这段Python代码实现了一个称为“边界框投票”(bounding box voting)的技术,在目标检测领域中用于模糊定位结果,通过结合相邻的边界框(segments)来提高性能,
        特别是在给定的交并比(IoU)阈值附近。这种方法在处理分割任务时尤其有用,比如实例分割,其中需要精确定位物体的边界。下面是代码的详细解释：
        nms_segs: 经过非极大值抑制(NMS)后保留的边界框集合,形状为N_i x 2,其中N_i是边界框的数量,每个边界框由两个元素组成,表示其起始和结束位置(或坐标)。
        all_segs: 所有候选边界框的集合,形状为N x 2,其中N是候选边界框的总数。
        all_scores: 每个候选边界框的得分,长度为N。
        iou_threshold: 用于确定哪些候选边界框与NMS后的边界框足够接近(通过IoU衡量),以参与投票的阈值。
        score_offset: 得分偏移量,用于调整候选边界框的得分,默认值为1.5。
    """

    # *_segs : N_i x 2, all_scores: N,
    # apply offset
    offset_scores = all_scores + score_offset

    # computer overlap between nms and all segs
    # construct the distance matrix of # N_nms x # N_all
    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms_segs = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all_segs = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)

    # compute intersection
    left = torch.maximum(ex_nms_segs[:, :, 0], ex_all_segs[:, :, 0])
    right = torch.minimum(ex_nms_segs[:, :, 1], ex_all_segs[:, :, 1])
    inter = (right-left).clamp(min=0)

    # lens of all segments
    nms_seg_lens = ex_nms_segs[:, :, 1] - ex_nms_segs[:, :, 0]
    all_seg_lens = ex_all_segs[:, :, 1] - ex_all_segs[:, :, 0]

    # iou
    iou = inter / (nms_seg_lens + all_seg_lens - inter)

    # get neighbors (# N_nms x # N_all) / weights
    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :] * iou
    seg_weights /= torch.sum(seg_weights, dim=1, keepdim=True)
    refined_segs = seg_weights @ all_segs

    return refined_segs

def batched_nms(
    segs,
    scores,
    cls_idxs,
    pred_score,
    iou_threshold,
    min_score,
    max_seg_num,
    use_soft_nms=True,
    multiclass=True,
    sigma=0.5,
    voting_thresh=0.75,
):
    # Based on Detectron2 implementation,
    num_segs = segs.shape[0]
    # corner case, no prediction outputs
    if num_segs == 0:
        return torch.zeros([0, 2]),\
               torch.zeros([0,]),\
               torch.zeros([0,], dtype=cls_idxs.dtype)

    if multiclass:
        # multiclass nms: apply nms on each class independently
        new_segs, new_scores, new_cls_idxs = [], [], []
        for class_id in torch.unique(cls_idxs):
            curr_indices = torch.where(cls_idxs == class_id)[0]
            # soft_nms vs nms
            if use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs, new_pred_score = SoftNMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    pred_score[curr_indices],
                    iou_threshold,
                    sigma,
                    min_score,
                    2,
                    max_seg_num
                )
            else:
                sorted_segs, sorted_scores, sorted_cls_idxs, new_pred_score = NMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    pred_score[curr_indices],
                    iou_threshold,
                    min_score,
                    max_seg_num
                )
            # disable seg voting for multiclass nms, no sufficient segs

            # fill in the class index
            new_segs.append(sorted_segs)
            new_scores.append(sorted_scores)
            new_cls_idxs.append(sorted_cls_idxs)

        # cat the results
        new_segs = torch.cat(new_segs)
        new_scores = torch.cat(new_scores)
        new_cls_idxs = torch.cat(new_cls_idxs)

    else:
        # class agnostic
        if use_soft_nms:
            new_segs, new_scores, new_cls_idxs, new_pred_score = SoftNMSop.apply(
                segs, scores, cls_idxs, pred_score, iou_threshold,
                sigma, min_score, 2, max_seg_num
            )
        else:
            new_segs, new_scores, new_cls_idxs, new_pred_score = NMSop.apply(
                segs, scores, cls_idxs, pred_score,iou_threshold,
                min_score, max_seg_num
            )
        # seg voting
        if voting_thresh > 0:
            new_segs = seg_voting(
                new_segs,
                segs,
                scores,
                voting_thresh
            )

    # sort based on scores and return
    # truncate the results based on max_seg_num
    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    # needed for multiclass NMS
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    new_pred_score = new_pred_score[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs, new_pred_score
