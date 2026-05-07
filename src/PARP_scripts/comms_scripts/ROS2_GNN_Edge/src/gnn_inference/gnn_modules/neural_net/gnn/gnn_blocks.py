# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : Common GNN building blocks
# ---------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from typing import List, Optional
from torch_geometric.nn.conv import MessagePassing
from gnn_modules.neural_net.common import ffn_block, layer_normalization, channel_normalization, group_normalization
from gnn_modules.neural_net.constants import (
    _CLS_CONV_MEAN_INIT_, _CLS_CONV_STD_INIT_, _CLS_CONV_BIAS_INIT_,
    _REG_CONV_MEAN_INIT_, _REG_CONV_STD_INIT_, _REG_CONV_BIAS_INIT_ )

# ---------------------------------------------------------------------------------------------------------------
def normalize_features(features, min_vals, max_vals):
    return (features - min_vals) / (max_vals - min_vals)

# ---------------------------------------------------------------------------------------------------------------
class graph_feature_encoding(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        stem_channels: List[int],
        activation: str,
        norm_layer: str,
        num_groups: int):
        super().__init__()

        encoder = []
        for i, stem_channel in enumerate(stem_channels):
            if i == 0:
                blk = ffn_block(in_channels=in_channels, out_channels=stem_channel, activation=activation)
            else:
                blk = ffn_block(
                    in_channels=in_channels, out_channels=stem_channel, activation=activation,
                    norm_layer = norm_layer, num_groups = num_groups)
            in_channels = stem_channel
            encoder.append(blk)
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

# ---------------------------------------------------------------------------------------------------------------
class residual_graph_conv_block(MessagePassing):
    def __init__(
        self, 
        in_node_channels: int,
        in_edge_channels: int,
        mlp_stem_channels_msg: List[int],
        mlp_stem_channels_upd: List[int],
        aggregation: str,
        activation: str,
        norm_layer: str,
        num_groups: int,
        in_extra_feature_dim: Optional[int] = None):
        super().__init__(aggr=aggregation, flow="source_to_target")

        msg = []
        in_channels = 2 * in_node_channels + in_edge_channels
        for stem_channel in mlp_stem_channels_msg:
            blk = ffn_block(
                in_channels=in_channels, out_channels=stem_channel, activation=activation,
                norm_layer = norm_layer, num_groups = num_groups)
            in_channels = stem_channel
            msg.append(blk)
        self.msg = nn.Sequential(*msg)

        self.in_extra_feature_dim = in_extra_feature_dim 
        in_channels = in_node_channels + mlp_stem_channels_msg[-1]
        if in_extra_feature_dim != None:
            in_channels = in_channels + in_extra_feature_dim 

        upd = []
        for stem_channel in mlp_stem_channels_upd:
            blk = ffn_block(
                in_channels=in_channels, out_channels=stem_channel, activation=activation,
                norm_layer = norm_layer, num_groups = num_groups)
            in_channels = stem_channel
            upd.append(blk)
        self.upd = nn.Sequential(*upd)

        # residual connection
        self.match_channels = False
        if in_node_channels != mlp_stem_channels_upd[-1]:
            self.match_channels = True
        
        self.residual_connection = None
        if self.match_channels:
            ffn_ = nn.Linear(in_features = in_node_channels,  out_features = mlp_stem_channels_upd[-1], bias = True)
            if norm_layer == 'layer_normalization': norm_ = layer_normalization()
            elif norm_layer == 'channel_normalization': norm_ = channel_normalization()
            elif norm_layer == 'group_normalization': norm_ = group_normalization(num_groups)
            self.residual_connection = nn.Sequential(ffn_, norm_)
        
    def forward(
        self, 
        node_features: torch.Tensor,    # dimension: (|V| x d_node)
        edge_features: torch.Tensor,    # dimension: (|E| x d_edge)
        edge_index: torch.Tensor,       # dimension: (2 x |E|)
        extra_features: Optional[torch.Tensor] = None):  # dimension: (|V| x d_aug)

        if self.match_channels: identity = self.residual_connection(node_features)
        else: identity = node_features
        #print("🧪 Step 5: residual_graph_conv_block")
        #print("  edge_index shape before propagate:", edge_index.shape)
        #print("  edge_index max:", edge_index.max().item(), "min:", edge_index.min().item())
        #print("  node_features shape:", node_features.shape)

        # ✅ Sanity check right before propagate
        if edge_index.min().item() < 0 or edge_index.max().item() >= node_features.shape[0]:
            print("❌ ERROR: edge_index contains out-of-bounds indices!")
            print("edge_index:\n", edge_index[:, :10])  # Print first few
            raise ValueError("edge_index has invalid indices")
        x = self.propagate(edge_index, x=node_features, edge_attr=edge_features)
        if self.in_extra_feature_dim != None: x = torch.concat((node_features, extra_features, x), dim=-1)
        else: x = torch.concat((node_features, x), dim=-1)
        x = identity + self.upd(x)
        return x
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor):
        return self.msg(torch.concat((x_i, x_j, edge_attr), dim=-1))

# ---------------------------------------------------------------------------------------------------------------
class graph_convolution(nn.Module):
    def __init__(
        self, 
        in_node_channels: int,
        in_edge_channels: int,
        stem_channels: List[int],
        msg_mlp_hidden_dim: int,
        activation: str,
        aggregation: str,
        norm_layer: str,
        num_groups: int,
        append_extra_features: Optional[List[bool]] = None,
        in_extra_feature_dim: Optional[int] = None):
        super().__init__()

        self.conv_blk = nn.ModuleList()
        for i, stem_channel in enumerate(stem_channels):
            in_extra_feature = None
            if append_extra_features != None:
                if append_extra_features[i] == True and in_extra_feature_dim != None:
                    in_extra_feature = in_extra_feature_dim

            self.conv_blk.append( residual_graph_conv_block(
                in_node_channels = in_node_channels,
                in_extra_feature_dim = in_extra_feature,
                in_edge_channels = in_edge_channels,
                mlp_stem_channels_msg = [msg_mlp_hidden_dim, stem_channel],
                mlp_stem_channels_upd = [stem_channel],
                aggregation = aggregation,
                activation = activation,
                norm_layer = norm_layer,
                num_groups = num_groups) )
            in_node_channels = stem_channel
            
    def forward(
        self, 
        node_features: torch.Tensor,    # dimension: (|V| x d_node)
        edge_features: torch.Tensor,    # dimension: (|E| x d_edge)
        edge_index: torch.Tensor,       # dimension: (2 x |E|)
        extra_features: Optional[torch.Tensor]):  # dimension: (|V| x d_aug)

        x = node_features
        #print("🧪 Step 4: graph_convolution.forward")
        #print("  edge_index max:", edge_index.max().item())
        #print("  edge_index shape:", edge_index.shape)
        #print("  node_features shape:", node_features.shape)

        for conv_blk in self.conv_blk:
            x = conv_blk(
                node_features = x,
                edge_features = edge_features,
                edge_index = edge_index,
                extra_features = extra_features)
        return x
    
# ---------------------------------------------------------------------------------------------------------------
class FFN_TaskSpecificHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        norm_layer: str,
        num_groups: int,
        init_weight_mu: torch.Tensor,
        init_weight_sigma: torch.Tensor,
        init_bias: torch.Tensor):
        super().__init__()

        _ffn_block = ffn_block(
            in_channels = in_channels,
            out_channels = in_channels,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        _ffn = nn.Linear(
            in_features = in_channels, 
            out_features = out_channels, 
            bias = True)
        
        torch.nn.init.normal_(_ffn.weight, mean=init_weight_mu, std=init_weight_sigma)
        torch.nn.init.constant_(_ffn.bias, init_bias) 
        self.head = nn.Sequential(_ffn_block, _ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
    
# ---------------------------------------------------------------------------------------------------------------
class node_segmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: List[int],
        num_classes: int, 
        activation: str,
        norm_layer: str,
        num_groups: int):
        super().__init__()

        stem_blks = []
        for stem_channel in stem_channels:
            blk = ffn_block(
                in_channels=in_channels, out_channels=stem_channel, activation=activation,
                norm_layer = norm_layer, num_groups = num_groups)
            in_channels = stem_channel
            stem_blks.append(blk)
        self.stem = nn.Sequential(*stem_blks)

        # Object Class prediction head
        self.pred_cls = FFN_TaskSpecificHead(
            in_channels = stem_channels[-1],
            out_channels = num_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups,
            init_weight_mu = _CLS_CONV_MEAN_INIT_,
            init_weight_sigma = _CLS_CONV_STD_INIT_,
            init_bias = _CLS_CONV_BIAS_INIT_)
        
    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        pred_cls = self.pred_cls(x)
        return pred_cls
    
# ---------------------------------------------------------------------------------------------------------------
class node_offset_predictions(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: List[int],
        reg_offset_dim: int, 
        activation: str,
        norm_layer: str,
        num_groups: int,):
        super().__init__()

        stem_blks = []
        for stem_channel in stem_channels:
            blk = ffn_block(
                in_channels=in_channels, out_channels=stem_channel, activation=activation,
                norm_layer = norm_layer, num_groups = num_groups)
            in_channels = stem_channel
            stem_blks.append(blk)
        self.stem = nn.Sequential(*stem_blks)

        # Box regression head
        self.pred_offsets = FFN_TaskSpecificHead(
            in_channels = stem_channels[-1],
            out_channels = reg_offset_dim,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups,
            init_weight_mu = _REG_CONV_MEAN_INIT_,
            init_weight_sigma = _REG_CONV_STD_INIT_,
            init_bias = _REG_CONV_BIAS_INIT_)
        
    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        pred_offsets = self.pred_offsets(x)
        return pred_offsets
    
# ---------------------------------------------------------------------------------------------------------------

class edge_formation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_blocks: int,
        activation: str,
        norm_layer: str,
        num_groups: int):
        super().__init__()

        stem_blks = []
        for _ in range(num_blocks):
            blk = ffn_block(
                in_channels=in_channels,
                out_channels=in_channels,
                activation=activation,
                norm_layer=norm_layer,
                num_groups=num_groups)
            stem_blks.append(blk)

        self.stem = nn.Sequential(*stem_blks)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            x: [N, F]
            edge_index: [2, E] with (i, j) pairs
        Returns:
            edge features: [E, F]
        """
        x = self.stem(x)
        src = edge_index[0]
        dst = edge_index[1]
        edge_feat = x[src] + x[dst]
        return edge_feat


# ---------------------------------------------------------------------------------------------------------------
class link_predictions(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_blks_for_edges: int,
        stem_channels: List[int],
        num_classes: int,
        activation: str,
        norm_layer: str,
        num_groups: int,):
        super().__init__()

        self.compute_edge = edge_formation(
            in_channels = in_channels,
            num_blocks = num_blks_for_edges,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)

        stem_blks = []
        for stem_channel in stem_channels:
            blk = ffn_block(
                in_channels=in_channels, out_channels=stem_channel, activation=activation,
                norm_layer = norm_layer, num_groups = num_groups)
            in_channels = stem_channel
            stem_blks.append(blk)
        self.stem = nn.Sequential(*stem_blks)

        # Edge Class prediction head
        self.pred_cls = FFN_TaskSpecificHead(
            in_channels = stem_channels[-1],
            out_channels = num_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups,
            init_weight_mu = _CLS_CONV_MEAN_INIT_,
            init_weight_sigma = _CLS_CONV_STD_INIT_,
            init_bias = _CLS_CONV_BIAS_INIT_)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.compute_edge(x, edge_index)
        x = self.stem(x)
        x = self.pred_cls(x)
        return x

# ---------------------------------------------------------------------------------------------------------------

class object_classification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: List[int],
        num_classes: int,
        activation: str,
        norm_layer: str,
        num_groups: int,):
        super().__init__()

        stem_blks = []
        for stem_channel in stem_channels:
            blk = ffn_block(
                in_channels=in_channels, out_channels=stem_channel, activation=activation,
                norm_layer=norm_layer, num_groups=num_groups)
            in_channels = stem_channel
            stem_blks.append(blk)
        self.stem = nn.Sequential(*stem_blks)

        # Object Class prediction head
        self.pred_cls = FFN_TaskSpecificHead(
            in_channels=stem_channels[-1],
            out_channels=num_classes,
            activation=activation,
            norm_layer=norm_layer,
            num_groups=num_groups,
            init_weight_mu=_CLS_CONV_MEAN_INIT_,
            init_weight_sigma=_CLS_CONV_STD_INIT_,
            init_bias=_CLS_CONV_BIAS_INIT_)

    def forward(self, x: torch.Tensor, cluster_node_idx: List[torch.Tensor]):
        x = self.stem(x)
        feature_dim = x.shape[1]  # inferred from stem output
        features = []

        for idx in cluster_node_idx:
            if isinstance(idx, torch.Tensor):
                #print(f"[DEBUG] Checking cluster idx: {idx}")
                #print(f"[DEBUG] x.shape = {x.shape}")
                #print(f"[DEBUG] idx.max() = {idx.max().item()}")

                cluster_x = x[idx]
                if cluster_x.dim() == 1:
                    cluster_x = cluster_x.unsqueeze(0)  # [1, D]

                if cluster_x.shape[1] != feature_dim:
                    print(f"⚠️ cluster_x shape mismatch: {cluster_x.shape}, expected: [*, {feature_dim}]")
                    cluster_x = cluster_x.T

                if cluster_x.size(0) > 1:
                    feat, _ = torch.max(cluster_x, dim=0, keepdim=True)
                else:
                    feat = cluster_x  # already [1, D]

                features.append(feat)
                #print(f"cluster_x shape: {cluster_x.shape}")
                #rint(f"feat shape: {feat.shape}")
            else:
                print(f"⚠️ Skipping cluster — not a Tensor: {type(idx)}")

        if len(features) == 0:
            #print("⚠️ No features to classify for clusters — returning zeros")
            return torch.zeros((0, self.pred_cls.head[-1].out_features), device=x.device)



        features = torch.concat(features, dim=0)
        pred_cls = self.pred_cls(features)
        return pred_cls

    
# ---------------------------------------------------------------------------------------------------------------
class node_predictions(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: List[int],
        num_classes: int, 
        reg_offset_dim: int,
        activation: str,
        norm_layer: str,
        num_groups: int,):
        super().__init__()

        stem_blks = []
        for stem_channel in stem_channels:
            blk = ffn_block(
                in_channels=in_channels, out_channels=stem_channel, activation=activation,
                norm_layer = norm_layer, num_groups = num_groups)
            in_channels = stem_channel
            stem_blks.append(blk)
        self.stem = nn.Sequential(*stem_blks)

        # Object Class prediction head
        self.pred_cls = FFN_TaskSpecificHead(
            in_channels = stem_channels[-1],
            out_channels = num_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups,
            init_weight_mu = _CLS_CONV_MEAN_INIT_,
            init_weight_sigma = _CLS_CONV_STD_INIT_,
            init_bias = _CLS_CONV_BIAS_INIT_)
        
        # Box regression head
        self.pred_offsets = FFN_TaskSpecificHead(
            in_channels = stem_channels[-1],
            out_channels = reg_offset_dim,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups,
            init_weight_mu = _REG_CONV_MEAN_INIT_,
            init_weight_sigma = _REG_CONV_STD_INIT_,
            init_bias = _REG_CONV_BIAS_INIT_)
        
    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        pred_cls = self.pred_cls(x)
        pred_offsets = self.pred_offsets(x)
        return pred_cls, pred_offsets