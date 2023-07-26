import torch
from torch import nn


class AbaViTrack(nn.Module):

    def __init__(self, transformer, box_head):

        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                t_mask=None,
                s_mask=None
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    t_mask=t_mask,
                                    s_mask=s_mask)
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['rho_token'] = self.backbone.rho_token
        out['halting_score_layer'] = self.backbone.halting_score_layer
        out['distr_target'] = self.backbone.distr_target
        out['kl_metric'] = self.backbone.kl_loss
        out['rho_token_weight'] = self.backbone.rho_token_weight
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new,
               'score_map': score_map_ctr,
               'size_map': size_map,
               'offset_map': offset_map}
        return out