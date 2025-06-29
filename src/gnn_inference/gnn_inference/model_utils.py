import torch, sys
import numpy as np
from gnn_object_segmentation.utils.config_loader import config
from gnn_object_segmentation.utils.param_loader import set_parameters_for_inference

def load_model(node):
    config_path = node.get_parameter_or("config_path", "/path/to/config.yaml")
    weights_path = node.get_parameter_or("weights_path", "/path/to/model_weights.pt")
    module_root = node.get_parameter_or("module_root", "/path/to/gnn_object_segmentation")

    config_obj = config(config_path)
    param_obj = set_parameters_for_inference(module_root, config_obj, weights_path)
    detector = param_obj['detector']
    detector.set_param_for_proposal_extraction(eps=1.0, compute_adj_mat_from_links=False)

    return detector

def run_model_inference(msg, detector):
    N = msg.num_nodes
    D = msg.node_feature_dim
    E = msg.num_edges
    D_e = msg.edge_attr_dim

    node_feats = torch.tensor(msg.node_features, dtype=torch.float32).reshape(N, D)
    edge_index = torch.tensor(msg.edge_index, dtype=torch.int64).reshape(2, E)
    edge_attr = torch.tensor(msg.edge_attr, dtype=torch.float32).reshape(E, D_e)

    with torch.no_grad():
        outputs = detector(
            node_features=node_feats,
            edge_features=edge_attr,
            other_features=node_feats,
            edge_index=edge_index,
            adj_matrix=None
        )

    return outputs[-1]  # cluster_members_list

# --------------------------------------------------------------------------------------------------------------
def set_parameters_for_inference(module_rootdir, config_obj, trained_weights_path):

    sys.path.append(module_rootdir)
    from modules.neural_net.gnn.gnn_detector import Model_Training
    from modules.compute_features.grid_features import grid_properties
    from modules.set_configurations.common import get_device

    device = get_device()

    # ================================================> INIT NETWORK STRUCTURE <========================================
    # incase we would like to resume training from a model weight checkpoint, set 'load_model_weights' as True and
    # set the weights_path
    weights_path = trained_weights_path
    detector_train = Model_Training(config_obj, device)
    detector_train.load_state_dict(torch.load(weights_path, map_location="cpu"))
    detector_train = detector_train.to(device)
    
    # ==============================================> DATASET & DATALOADER <===================================================
    grid_obj = grid_properties(
        min_x = config_obj.min_x, max_x = config_obj.max_x, 
        min_y = config_obj.min_y, max_y = config_obj.max_y, 
        min_sigma_x = config_obj.min_sigma_x, max_sigma_x = config_obj.max_sigma_x, 
        min_sigma_y = config_obj.min_sigma_y, max_sigma_y = config_obj.max_sigma_y, 
        dx = config_obj.dx, dy = config_obj.dy)

    return {
        'device': device,
        'grid': grid_obj,
        'detector': detector_train.pred.eval()}