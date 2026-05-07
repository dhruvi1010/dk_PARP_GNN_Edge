# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : Integrated model for inference and training
# ---------------------------------------------------------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict
from collections import namedtuple
det_named_tuple = namedtuple('det_named_tuple', ['node_class_logits', 'node_reg_deltas', 'edge_class_logits', 'obj_class_logits'])

from gnn_modules.neural_net.gnn.gnn_blocks import ( node_segmentation, node_offset_predictions, node_predictions,
    graph_feature_encoding, link_predictions, object_classification, graph_convolution )
from gnn_modules.neural_net.gnn.gnn_attention import graph_attention
from gnn_modules.neural_net.gnn.loss import Loss_Graph, Loss_Object_Class
from gnn_modules.compute_groundtruth.compute_offsets import normalize_gt_offsets
from gnn_modules.compute_groundtruth.compute_offsets import unnormalize_gt_offsets

from gnn_modules.inference.clustering import Simple_DBSCAN

# --------------------------------------------------------------------------------------------------------------
@ torch.no_grad()
def compute_accuracy(predicted_class, gt_class):
    if predicted_class.device != gt_class.device:
        gt_class = gt_class.to(predicted_class.device)
    _, cls_idx = torch.max(predicted_class, dim=-1)
    num_true_positives = (cls_idx == gt_class).sum()
    accuracy = num_true_positives / gt_class.shape[0]
    return accuracy
""" 
def align_edge_class_labels(edge_class_list, adj_matrix_list):
    aligned_edge_class_list = []

    for i, adj in enumerate(adj_matrix_list):
        edge_class_full = edge_class_list[i]
        adj = adj.to(edge_class_full.device)  # ensure same device

        adj_triu = torch.triu(adj, diagonal=1)
        edge_indices = adj_triu.nonzero(as_tuple=False)  # (E, 2)

        n = adj.shape[0]
        flat_indices = edge_indices[:, 0] * n + edge_indices[:, 1]

        print(f"[DEBUG] Sample {i}")
        print(f"  adj shape: {adj.shape}")
        print(f"  adj_triu nonzeros: {edge_indices.shape}")
        print(f"  flat_indices max: {flat_indices.max().item()} vs edge_class size: {edge_class_full.shape[0]}")

        # 🛡️ Defensive check to prevent indexing overflow
        if flat_indices.max() >= edge_class_full.numel():
            raise IndexError(
                f"Flat index out of bounds for sample {i}: "
                f"flat max {flat_indices.max().item()} vs edge_class_full size {edge_class_full.numel()}"
            )

        aligned_gt = edge_class_full[flat_indices]
        aligned_edge_class_list.append(aligned_gt)

    return torch.cat(aligned_edge_class_list, dim=0)
 """
# --------------------------------------------------------------------------------------------------------------
class Model_Inference(nn.Module):
    def __init__(self, net_config, extract_proposals=False, eps=1.4, compute_adj_mat_from_links=False):
        super().__init__()

        self.extract_proposals = extract_proposals

        self.reg_mu = net_config.reg_mu
        self.reg_sigma = net_config.reg_sigma

        activation = net_config.activation
        aggregation = net_config.aggregation
        norm_layer = net_config.norm_layer
        num_groups = net_config.num_groups

        input_node_feat_dim = net_config.input_node_feat_dim
        input_edge_feat_dim = net_config.input_edge_feat_dim
        
        node_feat_enc_stem_channels = net_config.node_feat_enc_stem_channels
        edge_feat_enc_stem_channels = net_config.edge_feat_enc_stem_channels
        graph_convolution_stem_channels = net_config.graph_convolution_stem_channels
        msg_mlp_hidden_dim = net_config.msg_mlp_hidden_dim

        num_blocks_to_compute_edge = net_config.num_blocks_to_compute_edge
        link_pred_stem_channels = net_config.link_pred_stem_channels
        node_pred_stem_channels = net_config.node_pred_stem_channels

        num_edge_classes = net_config.num_edge_classes
        num_classes = net_config.num_classes
        reg_offset_dim = net_config.reg_offset_dim

        self.encode_node_feat = graph_feature_encoding(
            in_channels = input_node_feat_dim, 
            stem_channels = node_feat_enc_stem_channels,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.encode_edge_feat = graph_feature_encoding(
            in_channels = input_edge_feat_dim, 
            stem_channels = edge_feat_enc_stem_channels,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.pass_messages = graph_convolution(
            in_node_channels = node_feat_enc_stem_channels[-1],
            in_edge_channels = edge_feat_enc_stem_channels[-1],
            stem_channels = graph_convolution_stem_channels,
            msg_mlp_hidden_dim = msg_mlp_hidden_dim,
            activation = activation,
            aggregation = aggregation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.predict_node = node_segmentation(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            num_classes = num_classes, 
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.predict_offset = node_offset_predictions(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            reg_offset_dim = reg_offset_dim,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.predict_link = link_predictions(
            in_channels = graph_convolution_stem_channels[-1],
            num_blks_for_edges = num_blocks_to_compute_edge,
            stem_channels = link_pred_stem_channels,
            num_classes = num_edge_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.predict_class = object_classification(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            num_classes = num_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        if extract_proposals == True:
            self.set_param_for_proposal_extraction(eps, compute_adj_mat_from_links)

    @staticmethod
    def freeze_weights(nn_module):
        for param in nn_module.parameters():
            param.requires_grad = False
        return nn_module

    def freeze_layers_except_object_class_predictor(self):
        self.encode_node_feat = self.freeze_weights(self.encode_node_feat)
        self.encode_edge_feat = self.freeze_weights(self.encode_edge_feat)
        self.pass_messages = self.freeze_weights(self.pass_messages)
        self.predict_node = self.freeze_weights(self.predict_node)
        self.predict_offset = self.freeze_weights(self.predict_offset)
        self.predict_link = self.freeze_weights(self.predict_link)

    def set_param_for_proposal_extraction(self, eps, compute_adj_mat_from_links):
        print("proposal extraction called")
        self.compute_adj_mat_from_links = compute_adj_mat_from_links
        self.extract_proposals = True
        self.meas_noise_cov = 0.5 * np.eye(2, dtype=np.float32)
        self.clustering_obj = Simple_DBSCAN(eps, compute_adj_mat_from_links)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        adj_matrix: torch.Tensor,
        cluster_node_idx: Optional[List[torch.Tensor]] = None,
        other_features: Optional[torch.Tensor] = None,
        augmented_features: Optional[torch.Tensor] = None):
        #print("🧪 Step 3: Model_Inference.forward")
        #print("  node_features before encoding:", node_features.shape)
        #print("  edge_features before encoding:", edge_features.shape)
        #node_features = node_features.to(self.device)
        node_features = self.encode_node_feat(node_features)
        edge_features = self.encode_edge_feat(edge_features)
        #print("  node_features encoded:", node_features.shape)
        #print("  edge_features encoded:", edge_features.shape)
        #print("  edge_index:", edge_index.shape)
        node_features = self.pass_messages(node_features, edge_features, edge_index, augmented_features)
        node_cls_predictions = self.predict_node(node_features)
        node_offsets_predictions = self.predict_offset(node_features)
        ##print("📌 [DEBUG] Before predict_link")
        #print("    node_features shape:", node_features.shape)
        #print("    adj_matrix shape:", adj_matrix.shape)
        link_cls_predictions = self.predict_link(node_features, edge_index)
        #print("📌 [DEBUG] link_cls_predictions shape:", link_cls_predictions.shape)
        #print("    link_cls_predictions min/max:", link_cls_predictions.min().item(), link_cls_predictions.max().item())

        edge_cls_prob= F.softmax(link_cls_predictions, dim=-1)
        _, edge_cls_idx = torch.max(edge_cls_prob, dim=-1)
        obj_cls_predictions = None
        cluster_members_list = None
        # if cluster_node_idx != None:
        #     #print(f"[s] → node_feat.shape = {node_features.shape}, cluster count = {len(cluster_node_idx)}")
        #     obj_cls_predictions = self.predict_class(node_features, cluster_node_idx)

        # else:
        #     # compute offsets & perform dbscan clustering
        #     node_offsets_predictions_cpy = node_offsets_predictions.clone().detach()
        #     reg_deltas = unnormalize_gt_offsets(node_offsets_predictions_cpy, self.reg_mu, self.reg_sigma)
        #     pred_cluster_centers_xy = other_features[:, :2] + reg_deltas

        #     if self.compute_adj_mat_from_links == True:
        #         self.clustering_obj.cluster_nodes(
        #             pred_cluster_centers_xy.detach().cpu().numpy(),
        #             edge_cls_idx.detach().cpu().numpy(), 
        #             adj_matrix.detach().cpu().numpy())
                
        #     else:
        #         self.clustering_obj.cluster_nodes(pred_cluster_centers_xy.detach().cpu().numpy())

        #     # extract clusters
        #     cluster_members_list = []
        #     for i in range(self.clustering_obj.num_clusters):
        #         cluster_members = np.nonzero(self.clustering_obj.meas_to_cluster_id == i)[0]
        #         cluster_members = torch.tensor(cluster_members).to(node_features.device).to(int)
        #         cluster_members_list.append(cluster_members)

        #     # object class prediction
        #     obj_cls_predictions = self.predict_class(node_features, cluster_members_list)

        if self.extract_proposals == True:
            return \
                node_cls_predictions, \
                node_offsets_predictions, \
                link_cls_predictions, \
                obj_cls_predictions, \
                cluster_members_list
        else:
            return \
                node_cls_predictions, \
                node_offsets_predictions, \
                link_cls_predictions, \
                obj_cls_predictions
    
# ---------------------------------------------------------------------------------------------------------------
class Model_Inference_v1(nn.Module):
    def __init__(self, net_config):
        super().__init__()

        activation = net_config.activation
        aggregation = net_config.aggregation
        norm_layer = net_config.norm_layer
        num_groups = net_config.num_groups

        input_node_feat_dim = net_config.input_node_feat_dim
        input_edge_feat_dim = net_config.input_edge_feat_dim
        
        node_feat_enc_stem_channels = net_config.node_feat_enc_stem_channels
        edge_feat_enc_stem_channels = net_config.edge_feat_enc_stem_channels
        graph_convolution_stem_channels = net_config.graph_convolution_stem_channels
        msg_mlp_hidden_dim = net_config.msg_mlp_hidden_dim

        num_blocks_to_compute_edge = net_config.num_blocks_to_compute_edge
        link_pred_stem_channels = net_config.link_pred_stem_channels
        node_pred_stem_channels = net_config.node_pred_stem_channels

        num_edge_classes = net_config.num_edge_classes
        num_classes = net_config.num_classes
        reg_offset_dim = net_config.reg_offset_dim

        self.encode_node_feat = graph_feature_encoding(
            in_channels = input_node_feat_dim, 
            stem_channels = node_feat_enc_stem_channels,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.encode_edge_feat = graph_feature_encoding(
            in_channels = input_edge_feat_dim, 
            stem_channels = edge_feat_enc_stem_channels,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.pass_messages = graph_convolution(
            in_node_channels = node_feat_enc_stem_channels[-1],
            in_edge_channels = edge_feat_enc_stem_channels[-1],
            stem_channels = graph_convolution_stem_channels,
            msg_mlp_hidden_dim = msg_mlp_hidden_dim,
            activation = activation,
            aggregation = aggregation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        # self.predict_node = node_segmentation(
        #     in_channels = graph_convolution_stem_channels[-1],
        #     stem_channels = node_pred_stem_channels,
        #     num_classes = num_classes, 
        #     activation = activation)
        
        # self.predict_offset = node_offset_predictions(
        #     in_channels = graph_convolution_stem_channels[-1],
        #     stem_channels = node_pred_stem_channels,
        #     reg_offset_dim = reg_offset_dim,
        #     activation = activation)

        self.predict_node = node_predictions(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            num_classes = num_classes, 
            reg_offset_dim = reg_offset_dim,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.predict_link = link_predictions(
            in_channels = graph_convolution_stem_channels[-1],
            num_blks_for_edges = num_blocks_to_compute_edge,
            stem_channels = link_pred_stem_channels,
            num_classes = num_edge_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.predict_class = object_classification(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            num_classes = num_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        adj_matrix: torch.Tensor,
        cluster_node_idx: List[torch.Tensor],
        augmented_features: Optional[torch.Tensor] = None):

        node_features = self.encode_node_feat(node_features)
        edge_features = self.encode_edge_feat(edge_features)
        node_features = self.pass_messages(node_features, edge_features, edge_index, augmented_features)
        # node_cls_predictions = self.predict_node(node_features)
        # node_offsets_predictions = self.predict_offset(node_features)
        node_cls_predictions, node_offsets_predictions = self.predict_node(node_features)
        link_cls_predictions = self.predict_link(node_features, adj_matrix)
        obj_cls_predictions = self.predict_class(node_features, cluster_node_idx)

        return \
            node_cls_predictions, \
            node_offsets_predictions, \
            link_cls_predictions, \
            obj_cls_predictions
    
# ---------------------------------------------------------------------------------------------------------------
class Model_Inference_v2(nn.Module):
    def __init__(self, net_config):
        super().__init__()

        activation = net_config.activation
        norm_layer = net_config.norm_layer
        num_groups = net_config.num_groups

        input_node_feat_dim = net_config.input_node_feat_dim
        input_edge_feat_dim = net_config.input_edge_feat_dim
        
        node_feat_enc_stem_channels = net_config.node_feat_enc_stem_channels
        edge_feat_enc_stem_channels = net_config.edge_feat_enc_stem_channels
        graph_convolution_stem_channels = net_config.graph_convolution_stem_channels

        hidden_node_channels_GAT = net_config.hidden_node_channels_GAT
        num_heads_GAT = net_config.num_heads_GAT

        num_blocks_to_compute_edge = net_config.num_blocks_to_compute_edge
        link_pred_stem_channels = net_config.link_pred_stem_channels
        node_pred_stem_channels = net_config.node_pred_stem_channels

        num_edge_classes = net_config.num_edge_classes
        num_classes = net_config.num_classes
        reg_offset_dim = net_config.reg_offset_dim

        self.encode_node_feat = graph_feature_encoding(
            in_channels = input_node_feat_dim, 
            stem_channels = node_feat_enc_stem_channels,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.encode_edge_feat = graph_feature_encoding(
            in_channels = input_edge_feat_dim, 
            stem_channels = edge_feat_enc_stem_channels,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.pass_messages = graph_attention(
            in_node_channels = node_feat_enc_stem_channels[-1],
            in_edge_channels = edge_feat_enc_stem_channels[-1],
            stem_channels = graph_convolution_stem_channels,
            hidden_node_channels = hidden_node_channels_GAT,
            num_heads = num_heads_GAT,
            activation = activation)
        
        self.predict_node = node_segmentation(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            num_classes = num_classes, 
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.predict_offset = node_offset_predictions(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            reg_offset_dim = reg_offset_dim,
            activation = activation)
        
        self.predict_link = link_predictions(
            in_channels = graph_convolution_stem_channels[-1],
            num_blks_for_edges = num_blocks_to_compute_edge,
            stem_channels = link_pred_stem_channels,
            num_classes = num_edge_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
        self.predict_class = object_classification(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            num_classes = num_classes,
            activation = activation,
            norm_layer = norm_layer,
            num_groups = num_groups)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        adj_matrix: torch.Tensor,
        cluster_node_idx: List[torch.Tensor],
        augmented_features: Optional[torch.Tensor] = None):

        node_features = self.encode_node_feat(node_features)
        edge_features = self.encode_edge_feat(edge_features)
        node_features = self.pass_messages(node_features, edge_features, edge_index, augmented_features)
        node_cls_predictions = self.predict_node(node_features)
        node_offsets_predictions = self.predict_offset(node_features)
        link_cls_predictions = self.predict_link(node_features, adj_matrix)
        obj_cls_predictions = self.predict_class(node_features, cluster_node_idx)

        return \
            node_cls_predictions, \
            node_offsets_predictions, \
            link_cls_predictions, \
            obj_cls_predictions

# ---------------------------------------------------------------------------------------------------------------
class Model_Training(nn.Module):
    def __init__(self, net_config, device):
        super().__init__()   
        self.pred = Model_Inference(net_config)
        self.loss = Loss_Graph(net_config, device)
        self.device = device
        self.offset_mu = net_config.offset_mu
        self.offset_sigma = net_config.offset_sigma

    def forward(
        self,
        node_features: List[torch.Tensor],
        edge_features: List[torch.Tensor],
        edge_index: List[torch.Tensor],
        adj_matrix: List[torch.Tensor],
        labels: Dict[str, List[torch.Tensor]]
    ):
        node_cls_pred_list = []
        node_reg_pred_list = []
        edge_cls_pred_list = []
        obj_cls_pred_list = []

        #model_device = next(self.parameters()).device
        cluster_node_idx = labels['cluster_node_idx']

        for node_feat, edge_feat, edge_idx, adj_mat, node_idx in zip(
            node_features, edge_features, edge_index, adj_matrix, cluster_node_idx
        ):
            # === Move graph components to model device ===
            #node_feat = node_feat.to(model_device)
            #edge_feat = edge_feat.to(model_device)
            #edge_idx = edge_idx.to(model_device)
            #adj_mat = adj_mat.to(model_device)
            #edge_cls_gt_i = labels['edge_class'][i].to(model_device)

            # === Cluster Info ===
            #print(f"\n🧪 Step 2: Model_Training.forward — sample {i}")
            #print(f"\n🧪 Step 2: Model_Training.forward ")
            #print(f"  node_feat shape: {node_feat.shape}")
            #print(f"  edge_feat shape: {edge_feat.shape}")
            #print(f"  edge_idx shape: {edge_idx.shape}")
            #print(f"  adj_mat shape: {adj_mat.shape}")

            #if isinstance(node_idx, list):
            #    print(f"  ✅ num clusters in this sample: {len(node_idx)}")
                #node_idx = [c.to(model_device) for c in node_idx if isinstance(c, torch.Tensor)]
                #for j, cluster in enumerate(node_idx):
                #    if cluster.numel() > 0:
                #        print(f"    cluster[{j}]: max idx = {cluster.max().item()}, size = {cluster.numel()}")
                #    else:
                #        print(f"    ⚠️ cluster[{j}] is empty")
            #else:
            #    print(f"  ❌ cluster_node_idx is not a list → {type(node_idx)}")

            # === Edge Class Check ===
            #expected_edges = torch.triu(adj_mat, diagonal=1).nonzero(as_tuple=False).shape[0]
            #print(f"[DEBUG] Edge sanity:")
            #print(f"  - edge_index: {edge_idx.shape[1]}")
            #print(f"  - edge_class: {edge_cls_gt_i.shape[0]}")
            #print(f"  - expected from adj_matrix: {expected_edges}")

            # === Forward through model ===
            node_feat = node_feat.to(self.device)
            edge_feat = edge_feat.to(self.device)
            edge_idx = edge_idx.to(self.device)
            node_cls_pred, node_reg_pred, edge_cls_pred, obj_cls_pred = self.pred(
                node_feat, edge_feat, edge_idx, adj_mat, node_idx
            )

            #print(f"  ✅ Prediction shapes:")
            #print(f"    node_cls_pred: {node_cls_pred.shape}")
            #print(f"    node_reg_pred: {node_reg_pred.shape}")
            #print(f"    edge_cls_pred: {edge_cls_pred.shape}")
            #print(f"    obj_cls_pred:  {obj_cls_pred.shape}")
            #print(f"  🧾 GT edge_class[{i}]: {edge_cls_gt_i.shape}")

            #if edge_cls_pred.shape[0] != edge_cls_gt_i.shape[0]:
            #    print("❌ Edge prediction mismatch!")
            #    print(f"    Pred shape: {edge_cls_pred.shape}")
            #    print(f"    GT shape:   {edge_cls_gt_i.shape}")
            #    print(f"    edge_index: {edge_idx.shape}")
            #    print(f"    edge_feat:  {edge_feat.shape}")

            node_cls_pred_list.append(node_cls_pred)
            node_reg_pred_list.append(node_reg_pred)
            edge_cls_pred_list.append(edge_cls_pred)
            obj_cls_pred_list.append(obj_cls_pred)

        # === Concatenate predictions ===
        node_cls_pred = torch.cat(node_cls_pred_list, dim=0)
        node_reg_pred = torch.cat(node_reg_pred_list, dim=0)
        edge_cls_pred = torch.cat(edge_cls_pred_list, dim=0)
        obj_cls_pred = torch.cat(obj_cls_pred_list, dim=0)

        #print(f"\n📦 Final prediction shapes before loss:")
        #print(f"  node_cls_pred: {node_cls_pred.shape}")
        #print(f"  node_reg_pred: {node_reg_pred.shape}")
        #print(f"  edge_cls_pred: {edge_cls_pred.shape}")
        #print(f"  obj_cls_pred:  {obj_cls_pred.shape}")

        predictions = det_named_tuple(node_cls_pred, node_reg_pred, edge_cls_pred, obj_cls_pred)

        # === Prepare ground truths on device ===
        #obj_cls_gt = torch.cat([x.to(model_device) for x in labels['cluster_labels']], dim=0)
        #edge_cls_gt = torch.cat([x.to(model_device) for x in labels['edge_class']], dim=0)
        #node_cls_gt = torch.cat([x.to(model_device) for x in labels['node_class']], dim=0)
        #node_reg_gt = normalize_gt_offsets(
        #    torch.cat([x.to(model_device) for x in labels['node_offsets']], dim=0),
        #    self.offset_mu, self.offset_sigma
        #)
        obj_cls_gt = torch.cat(labels['cluster_labels'], dim=0)
        edge_cls_gt = torch.cat(labels['edge_class'], dim=0)
        node_cls_gt = torch.cat(labels['node_class'], dim=0)
        node_reg_gt = normalize_gt_offsets(
            torch.cat(labels['node_offsets'], dim=0),
            self.offset_mu, self.offset_sigma
        )

        #print(f"\n📦 Final ground truth shapes before loss:")
        #print(f"  node_cls_gt: {node_cls_gt.shape}")
        #print(f"  node_reg_gt: {node_reg_gt.shape}")
        #print(f"  edge_cls_gt: {edge_cls_gt.shape}")
        #print(f"  obj_cls_gt:  {obj_cls_gt.shape}")

        groundtruths = det_named_tuple(node_cls_gt, node_reg_gt, edge_cls_gt, obj_cls_gt)

        for name, tensor in predictions._asdict().items():
            if tensor.device != self.device:
                predictions = predictions._replace(**{name: tensor.to(self.device)})

        for name, tensor in groundtruths._asdict().items():
            if tensor.device != self.device:
                groundtruths = groundtruths._replace(**{name: tensor.to(self.device)})

        # === Compute loss and accuracy ===
        loss = self.loss(predictions, groundtruths)

        accuracy = {
            'segment_accuracy': compute_accuracy(node_cls_pred, node_cls_gt),
            'edge_accuracy': compute_accuracy(edge_cls_pred, edge_cls_gt),
            'object_accuracy': compute_accuracy(obj_cls_pred, obj_cls_gt),
        } 

        return loss, accuracy


    
# ---------------------------------------------------------------------------------------------------------------
class Model_Object_Classifier_Finetuning(nn.Module):
    def __init__(self, net_config):
        super().__init__()   
        self.pred = Model_Inference(net_config, extract_proposals=True, eps=net_config.clustering_eps)
        self.loss = Loss_Object_Class(net_config)

    def forward(
        self,
        node_features: List[torch.Tensor],
        edge_features: List[torch.Tensor],
        other_features: List[torch.Tensor],
        edge_index: List[torch.Tensor],
        adj_matrix: List[torch.Tensor],
        node_class_labels: List[torch.Tensor]):

        obj_cls_gt_list = []
        obj_cls_pred_list = []
        
        for node_feat, edge_feat, other_feat, edge_idx, adj_mat, node_gt_class \
            in zip(node_features, edge_features, other_features, edge_index, adj_matrix, node_class_labels):
            
            node_cls_pred, node_reg_pred, edge_cls_pred, obj_cls_pred, cluster_mem_list \
                = self.pred(node_features = node_feat, 
                            edge_features = edge_feat, 
                            other_features = other_feat,
                            edge_index = edge_idx, 
                            adj_matrix = adj_mat)
            
            obj_cls_pred_list.append(obj_cls_pred)
            
            for cluster_mem in cluster_mem_list:
                cluster_meas_labels = node_gt_class[cluster_mem]
                obj_cls_gt_list.append(torch.argmax(torch.bincount(cluster_meas_labels)))

        obj_cls_gt = torch.stack(obj_cls_gt_list, dim=0)
        obj_cls_pred = torch.concat(obj_cls_pred_list, dim=0)
        loss = self.loss(obj_cls_pred, obj_cls_gt)
        accuracy = compute_accuracy(obj_cls_pred, obj_cls_gt)
        return loss, accuracy

            