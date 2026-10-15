import copy
import lightning as L
import math
import torch
import torchmetrics

from torch import nn
from torch.nn import LayerNorm
from torchmetrics.detection import MeanAveragePrecision
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrObjectDetectionOutput,
    DeformableDetrConvEncoder,
    DeformableDetrConvModel,
    DeformableDetrModelOutput,
    DeformableDetrHungarianMatcher,
    DeformableDetrMLPPredictionHead,
    DeformableDetrLoss,
    DeformableDetrEncoder,
    DeformableDetrDecoder,
    DeformableDetrSinePositionEmbedding,
    inverse_sigmoid,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.pytorch_utils import meshgrid
from typing import Optional, Tuple, Union


class DeformableDetr(L.LightningModule):
    """
    Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks such as COCO detection.
    Taken and modified from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1864
    """

    def __init__(
        self,
        train=False,
        batch_size=4,
        num_channels=3,
        positional_encoding="sine",
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        two_stage=True,
        box_refine=True,
        bbox_loss_coefficient=5,
        bbox_cost=5,
        classification_loss_coefficient=1,
        class_cost=3,
        giou_loss_coefficient=2,
        giou_cost=2,
        focal_alpha=0.25,
        use_auxiliary_loss=False,
        auxiliary_loss_coefficient=0,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        config = DefDETRConfig(
            use_pretrained_backbone=False,
            position_embedding_type=positional_encoding,
            num_channels=num_channels,
            bbox_loss_coefficient=bbox_loss_coefficient,
            bbox_cost=bbox_cost,
            class_cost=class_cost,
            giou_cost=giou_cost,
            giou_loss_coefficient=giou_loss_coefficient,
            focal_alpha=focal_alpha,
            auxiliary_loss=use_auxiliary_loss,
            auxiliary_loss_coefficient=auxiliary_loss_coefficient,
            classification_loss_coefficient=classification_loss_coefficient,
            id2label={0: "Background", 1: "Alive", 2: "Dead"},
            two_stage=two_stage,
            with_box_refine=box_refine,
        )

        self.model = DeformableDetrTransformer(config, train)

        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=4,
            num_layers=3,
        )

        matcher = DeformableDetrHungarianMatcher(
            class_cost=config.class_cost,
            bbox_cost=config.bbox_cost,
            giou_cost=config.giou_cost,
        )
        self.loss_func = DeformableDetrLoss(
            matcher=matcher,
            num_classes=config.num_labels,
            focal_alpha=config.focal_alpha,
            losses=["labels", "boxes", "cardinality"],
        )

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        )

        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.model.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None
        if config.two_stage:
            # hack implementation for two-stage
            self.model.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.config = config

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)

        hidden_states = (
            outputs.intermediate_hidden_states
        )  # [batch_size, number_of_queries, 256]
        init_reference = (
            outputs.init_reference_points
        )  # [batch_size, number_of_queries, 2]
        inter_references = (
            outputs.intermediate_reference_points
        )  # [batch_size, number_of_layers (6), number_of_queries), 2]

        dec_output = outputs.last_hidden_state

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []

        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[level](hidden_states[:, level])
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(
                    f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}"
                )
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        return logits, pred_boxes, outputs_class, outputs_coord, dec_output

    def _step(self, batch):
        pixel_values, labels = batch

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        logits, pred_boxes, outputs_class, outputs_coord, _ = self.forward(pixel_values)

        loss, loss_dict, auxiliary_outputs = None, None, None

        # Second: compute the losses, based on outputs and labels
        if labels is not None:
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes

            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            loss_dict = self.loss_func(outputs_loss, labels)

            # Third: compute total loss, as a weighted sum of the various losses
            weight_dict = {
                "loss_ce": self.config.classification_loss_coefficient,
                "loss_bbox": self.config.bbox_loss_coefficient,
                "loss_giou": self.config.giou_loss_coefficient,
            }

            loss = sum(
                (
                    loss_dict[k] * weight_dict[k]
                    if k in weight_dict
                    else (
                        loss_dict[k] * eval(self.config.auxiliary_loss_coefficient)
                        if "cardinality" not in k
                        else 0
                    )
                )
                for k in loss_dict.keys()
            )

        return loss, loss_dict

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def _log(self, loss, loss_dict, mode):
        self.log(
            f"{mode}/Loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        self.log_dict(
            {f"{mode}/{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", verbose=True
            ),
            "monitor": "Val/Loss_epoch",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch)

        self._log(loss, loss_dict, "Train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch)

        self._log(loss, loss_dict, "Val")

        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch)

        self._log(loss, loss_dict, "Test")

        return loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDetrTransformer(nn.Module):
    """
    The bare Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    Taken from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1507
    """

    def __init__(self, config, train):
        super().__init__()

        # Create backbone + positional encoding
        backbone = DeformableDetrConvEncoder(config)
        embedding_dim = config.d_model // 2
        if config.position_embedding_type == "sine":
            position_embeddings = DeformableDetrSinePositionEmbedding(
                embedding_dim, normalize=True
            )
        elif config.position_embedding_type == "learned":
            position_embeddings = DeformableDetrLearnedPositionEmbedding(embedding_dim)

        self.backbone = DeformableDetrConvModel(backbone, position_embeddings)

        # Create input projection layers
        num_backbone_outs = len(backbone.intermediate_channel_sizes)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = backbone.intermediate_channel_sizes[_]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                    nn.GroupNorm(32, config.d_model),
                )
            )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        config.d_model,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(32, config.d_model),
                )
            )
            in_channels = config.d_model
        self.input_proj = nn.ModuleList(input_proj_list)

        if not config.two_stage:
            self.query_position_embeddings = nn.Embedding(
                config.num_queries, config.d_model * 2
            )

        self.encoder = DeformableDetrEncoder(config)
        self.decoder = DeformableDetrDecoder(config)

        self.level_embed = nn.Parameter(
            torch.Tensor(config.num_feature_levels, config.d_model)
        )

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            self.pos_trans = nn.Linear(config.d_model * 2, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
        else:
            self.reference_points = nn.Linear(config.d_model, 2)

        if train:
            self.unfreeze_backbone()

        self.config = config

    def freeze_backbone(self):
        print("Freezing backbone")
        for _, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        print("Unfreezing backbone")
        for _, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_height = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals):
        """Get the position embedding of the proposals."""

        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.int64, device=proposals.device
        ).float()
        dim_t = temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
        )
        # batch_size, num_queries, 4
        proposals = proposals.sigmoid() * scale
        # batch_size, num_queries, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # batch_size, num_queries, 4, 64, 2 -> batch_size, num_queries, 512
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (Tensor[num_feature_levels, 2]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(
                batch_size, height, width, 1
            )
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(
                    0, height - 1, height, dtype=torch.float32, device=enc_output.device
                ),
                torch.linspace(
                    0, width - 1, width, dtype=torch.float32, device=enc_output.device
                ),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat(
                [valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1
            ).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
        output_proposals = torch.log(
            output_proposals / (1 - output_proposals)
        )  # inverse sigmoid
        output_proposals = output_proposals.masked_fill(
            padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DeformableDetrModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        >>> model = DeformableDetrModel.from_pretrained("SenseTime/deformable-detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(
                ((batch_size, height, width)), dtype=torch.long, device=device
            )

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(self.input_proj[level](source))
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj[level](features[-1][0])
                else:
                    source = self.input_proj[level](sources[-1])
                mask = nn.functional.interpolate(
                    pixel_mask[None].float(), size=source.shape[-2:]
                ).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        # Create queries
        query_embeds = None
        if not self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(
            zip(sources, masks, position_embeddings_list)
        ):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=source_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1
        )
        valid_ratios = valid_ratios.float()

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=source_flatten,
                attention_mask=mask_flatten,
                position_embeddings=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, prepare decoder inputs
        batch_size, _, num_channels = encoder_outputs[0].shape
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        if self.config.two_stage:
            object_query_embedding, output_proposals = (
                self.gen_encoder_output_proposals(
                    encoder_outputs[0], ~mask_flatten, spatial_shapes
                )
            )

            # hack implementation for two-stage Deformable DETR
            # apply a detection head to each pixel (A.4 in paper)
            # linear projection for bounding box binary classification (i.e. foreground and background)
            enc_outputs_class = self.decoder.class_embed[-1](object_query_embedding)
            # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
            delta_bbox = self.decoder.bbox_embed[-1](object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals

            # only keep top scoring `config.two_stage_num_proposals` proposals
            topk = self.config.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_logits = torch.gather(
                enc_outputs_coord_logits,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )

            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            init_reference_points = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_logits))
            )
            query_embed, target = torch.split(pos_trans_out, num_channels, dim=2)
        else:
            query_embed, target = torch.split(query_embeds, num_channels, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_points = reference_points

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(
                value
                for value in [enc_outputs_class, enc_outputs_coord_logits]
                if value is not None
            )
            tuple_outputs = (
                (init_reference_points,)
                + decoder_outputs
                + encoder_outputs
                + enc_outputs
            )

            return tuple_outputs

        return DeformableDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


class DeformableDetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(embedding_dim, embedding_dim)
        self.column_embeddings = nn.Embedding(embedding_dim, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(height, 1, 1),
                y_emb.unsqueeze(1).repeat(1, width, 1),
            ],
            dim=-1,
        )
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


class DefDETRConfig(DeformableDetrConfig):
    """
    Configuration class to store the configuration for `DeformableDetr`. This class is used to instantiate DeformableDetr model.
    Primarily uses the same parameters as `DeformableDetrConfig` with some additional parameters for computing the loss.
    """

    model_type = "deformable_detr"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self, classification_loss_coefficient=1, auxiliary_loss_coefficient=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.classification_loss_coefficient = classification_loss_coefficient
        self.auxiliary_loss_coefficient = auxiliary_loss_coefficient
