import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_roi_extractor
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.losses.cross_entropy_loss import generate_block_target


@DETECTORS.register_module()
class CoDETR(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 query_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 mask_iou_head=None,                 
                 rpn_head=None,
                 roi_head=[None],
                 bbox_head=[None],
                 train_cfg=[None, None],
                 test_cfg=[None, None],
                 pretrained=[None, None],
                 init_cfg=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 eval_module='detr',
                 eval_index=0):
        super(CoDETR, self).__init__(init_cfg)
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
        # Module for evaluation, ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module
        # Module index for evaluation
        self.eval_index = eval_index
        self.backbone = build_backbone(backbone)

        head_idx = 0

        if neck is not None:
            self.neck = build_neck(neck)

        if query_head is not None:
            query_head.update(train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            self.query_head = build_head(query_head)
            self.query_head.init_weights()
            head_idx += 1

        if mask_head is not None:
            """Initialize ``mask_head``"""
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.mask_head = build_head(mask_head)
            rcnn_train_cfg = train_cfg[head_idx] if (train_cfg and train_cfg[head_idx] is not None) else None
            self.rcnn_train_cfg = rcnn_train_cfg
            self.rcnn_test_cfg = test_cfg[head_idx]
            if rcnn_train_cfg is not None:
                assigner = rcnn_train_cfg.assigner
                sampler = rcnn_train_cfg.sampler
                self.bbox_assigner = build_assigner(assigner)
                self.bbox_sampler = build_sampler(
                    sampler, context=self)
            head_idx += 1

        if mask_iou_head is not None:
            self.mask_iou_head = build_head(mask_iou_head)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg[head_idx].rpn if (train_cfg is not None and train_cfg[head_idx] is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                rcnn_train_cfg = train_cfg[i+head_idx].rcnn if (train_cfg and train_cfg[i+head_idx] is not None) else None
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg[i+head_idx].rcnn)
                self.roi_head.append(build_head(roi_head[i]))
                self.roi_head[-1].init_weights()

        self.bbox_head = nn.ModuleList()
        for i in range(len(bbox_head)):
            if bbox_head[i]:
                bbox_head[i].update(train_cfg=train_cfg[i+head_idx+len(self.roi_head)] if (train_cfg and train_cfg[i+head_idx+len(self.roi_head)] is not None) else None)
                bbox_head[i].update(test_cfg=test_cfg[i+head_idx+len(self.roi_head)])
                self.bbox_head.append(build_head(bbox_head[i]))  
                self.bbox_head[-1].init_weights() 

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None and len(self.bbox_head)>0))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0 and self.roi_head[0].with_mask)
    
    def extract_feat(self, img, img_metas=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.query_head(x, dummy_img_metas)
        return outs

    def _mask_forward_train(self, x, sampling_results, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in training."""

        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        mask_results = self._mask_forward(x, pos_rois, torch.cat(pos_labels))
        stage_mask_targets = self.mask_head.get_targets(pos_bboxes, pos_assigned_gt_inds, gt_masks)
        loss_mask = self.mask_head.loss(mask_results['stage_instance_preds'], mask_results['hidden_states'], stage_mask_targets)
        mask_results.update(loss_mask=loss_mask)

        if hasattr(self, "mask_iou_head"):
            # mask iou head forward and loss
            pos_mask_pred = mask_results['stage_instance_preds'][1].squeeze(1)
            mask_iou_pred = self.mask_iou_head(mask_results['mask_feats'],
                                            pos_mask_pred)
            pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                            torch.cat(pos_labels)]
            mask_iou_targets = self.mask_iou_head.get_targets(
                sampling_results, gt_masks, pos_mask_pred,
                stage_mask_targets[1], self.rcnn_train_cfg)
            loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred,
                                                    mask_iou_targets)
            mask_results.update(loss_mask_iou=loss_mask_iou)
            
        return mask_results

    def _mask_forward(self, x, rois, roi_labels):
        """Mask head forward function used in both training and testing."""
        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_instance_preds, hidden_states = self.mask_head(ins_feats, x[0], rois, roi_labels)
        return dict(stage_instance_preds=stage_instance_preds, hidden_states=hidden_states, mask_feats=ins_feats)
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)

        losses = dict()
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        # DETR encoder and decoder forward
        if self.with_query_head:
            detr_forward_results = self.query_head.forward_train(x, img_metas, gt_bboxes,
                                                          gt_labels, gt_bboxes_ignore)
            bbox_losses, x = detr_forward_results[:2]
            if len(detr_forward_results)==3:
                results_list = detr_forward_results[2]
            losses.update(bbox_losses)
            

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal',
                                              self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if hasattr(self, "mask_head"):
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    results_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    results_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
            if 'loss_mask_iou' in mask_results:
                losses.update(mask_results['loss_mask_iou'])

        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else: 
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')     
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)
            
        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].forward_train(x, img_metas, gt_bboxes,
                                                        gt_labels, gt_bboxes_ignore)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    tmp = bbox_losses.pop('pos_coords')
            bbox_losses = upd_loss(bbox_losses, idx=i+len(self.roi_head))
            losses.update(bbox_losses)

        if self.with_pos_coord and len(positive_coords)>0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore, positive_coords[i], i)
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)

        return losses


    def simple_test_roi_head(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head[self.eval_index].simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_query_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        results_list, x = self.query_head.simple_test(
            x, img_metas, rescale=rescale, return_encoder_output=True)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        if hasattr(self, "mask_head"):
            det_bboxes, det_labels = [], []
            for det_bbox, det_label in results_list:
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
            det_bboxes = torch.stack(det_bboxes, dim=0)
            det_labels = torch.stack(det_labels, dim=0)            
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))        
        return bbox_results

    def simple_test_bbox_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        results_list = self.bbox_head[self.eval_index].simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head[self.eval_index].num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Obtain mask prediction without augmentation."""
        segm_results = []
        mask_scores = []
        for img_idx in range(len(img_metas)):
            ori_shape = img_metas[img_idx]['ori_shape']
            scale_factor = img_metas[img_idx]['scale_factor']
            det_bboxes = det_bboxes[img_idx]
            det_labels = det_labels[img_idx]            
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
                mask_score = [[] for _ in range(self.mask_head.stage_num_classes[0])]
            else:
                # if det_bboxes is rescaled to the original image size, we need to
                # rescale it back to the testing scale to obtain RoIs.
                if rescale and not isinstance(scale_factor, float):
                    scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
                _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                mask_rois = bbox2roi([_bboxes])

                interval = 150  # to avoid memory overflow
                segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
                mask_score = [[] for _ in range(self.mask_head.stage_num_classes[0])]
                for i in range(0, det_labels.shape[0], interval):
                    mask_results = self._mask_forward(x, mask_rois[i: i + interval], det_labels[i: i + interval])

                    # mask scoring
                    if hasattr(self, "mask_iou_head"):
                        # get mask scores with mask iou head
                        mask_feats = mask_results['mask_feats']
                        mask_pred = mask_results['stage_instance_preds'][1].squeeze(1)
                        mask_iou_pred = self.mask_iou_head(
                            mask_feats, mask_pred)
                        chunk_mask_score = self.mask_iou_head.get_mask_scores(
                            mask_iou_pred, det_bboxes[i: i + interval], det_labels[i: i + interval], return_score=True)

                    # refine instance masks from stage 1
                    stage_instance_preds = mask_results['stage_instance_preds'][1:]
                    for idx in range(len(stage_instance_preds) - 1):
                        instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
                        non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
                        non_boundary_mask = F.interpolate(
                            non_boundary_mask.float(),
                            stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
                        pre_pred = F.interpolate(
                            stage_instance_preds[idx],
                            stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                        stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
                    instance_pred = stage_instance_preds[-1]

                    chunk_segm_result = self.mask_head.get_seg_masks(
                        instance_pred, _bboxes[i: i + interval], det_labels[i: i + interval],
                        self.rcnn_test_cfg, ori_shape, scale_factor, rescale)

                    if hasattr(self, "mask_iou_head"):
                        for c, segm, score in zip(det_labels[i: i + interval], chunk_segm_result, chunk_mask_score):
                            segm_result[c].append(segm)     
                            mask_score[c].append(score)                   
                    else:
                        for c, segm in zip(det_labels[i: i + interval], chunk_segm_result):
                            segm_result[c].append(segm)
            segm_results.append(segm_result)
            mask_scores.append(mask_score)
        if hasattr(self, "mask_iou_head"):
            return list(zip(segm_results, mask_scores))
        return segm_results
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']
        if self.with_bbox and self.eval_module=='one-stage':
            return self.simple_test_bbox_head(img, img_metas, proposals, rescale)
        if self.with_roi_head and self.eval_module=='two-stage':
            return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        return self.simple_test_query_head(img, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.query_head, 'aug_test'), \
            f'{self.query_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.query_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.query_head.forward_onnx(x, img_metas)[:2]
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        # TODO support NMS
        # det_bboxes, det_labels = self.query_head.onnx_export(
        #     *outs, img_metas, with_nms=with_nms)
        det_bboxes, det_labels = self.query_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels