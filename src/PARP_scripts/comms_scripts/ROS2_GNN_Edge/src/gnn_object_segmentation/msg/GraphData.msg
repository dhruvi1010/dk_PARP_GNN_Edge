# gnn_object_segmentation/msg/GraphData.msg

# Tensor-like flat arrays with size metadata
float32[] node_features       # flattened [N x 10]
uint32 node_feature_dim       # always 10

int64[] edge_index            # flattened [2 x E]
float32[] edge_attr           # flattened [E x 5]
uint32 edge_attr_dim          # always 5

# Optional metadata
uint32 num_nodes
uint32 num_edges
