"""FREEDOM with a second, configurable masked user-item branch.

The original FREEDOM structure is preserved: collaborative U-I output is
added to an item representation propagated through the frozen multimodal I-I
graph.  Masking only creates a second U-I view.  With separate item tables,
both tables also pass through the same frozen I-I graph and their outputs are
combined by a weighted sum.
"""

import math
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds

from models.freedom import FREEDOM, _config_value


class _ObservedEdgeSparseMM(torch.autograd.Function):
    """Sparse matrix multiplication with edge-only adjacency gradients."""

    @staticmethod
    def forward(ctx, indices, values, size, embeddings):
        adjacency = torch.sparse_coo_tensor(
            indices,
            values,
            size,
            dtype=values.dtype,
            device=values.device,
        ).coalesce()
        ctx.adjacency_size = tuple(size)
        ctx.save_for_backward(
            adjacency.indices(), adjacency.values(), embeddings
        )
        return torch.sparse.mm(adjacency, embeddings)

    @staticmethod
    def backward(ctx, grad_output):
        indices, values, embeddings = ctx.saved_tensors
        row, col = indices

        grad_values = None
        if ctx.needs_input_grad[1]:
            grad_values = (
                grad_output.index_select(0, row)
                * embeddings.index_select(0, col)
            ).sum(dim=1)

        grad_embeddings = None
        if ctx.needs_input_grad[3]:
            transpose = torch.sparse_coo_tensor(
                torch.stack((col, row), dim=0),
                values,
                (ctx.adjacency_size[1], ctx.adjacency_size[0]),
                dtype=values.dtype,
                device=values.device,
            ).coalesce()
            grad_embeddings = torch.sparse.mm(transpose, grad_output)

        return None, grad_values, None, grad_embeddings


class FREEDOM_MASKED(FREEDOM):
    """FREEDOM whose collaborative path has original and masked views."""

    ITEM_INPUT_MODES = {
        'id', 'multimodal', 'multimodal_concat', 'hybrid'
    }
    MASK_GRAPH_MODES = {
        'soft',
        'hard',
        'double_full',
        'svd',
        'local_prunning',
        'random_fixed',
        'random_dynamic',
    }

    def __init__(self, config, dataset):
        super(FREEDOM_MASKED, self).__init__(config, dataset)

        self.mask_weight = float(_config_value(config, 'mask_weight', 0.1))
        self.mask_keep_ratio = float(
            _config_value(config, 'mask_keep_ratio', 0.3)
        )
        self.mask_binary_weight = float(
            _config_value(config, 'mask_binary_weight', 0.1)
        )
        self.mask_degree_mode = str(
            _config_value(config, 'mask_degree_mode', 'full')
        ).lower()
        self.mask_graph_mode = str(
            _config_value(config, 'mask_graph_mode', 'hard')
        ).lower()
        self.hard_mask_temperature = float(
            _config_value(config, 'hard_mask_temperature', 1.0)
        )
        self.random_mask_seed = int(
            _config_value(config, 'random_mask_seed', 999)
        )

        self.user_embedding_mode = str(
            _config_value(config, 'user_embedding_mode', 'separate')
        ).lower()
        self.item_embedding_mode = str(
            _config_value(config, 'item_embedding_mode', 'separate')
        ).lower()
        self.item_input_mode = str(
            _config_value(config, 'item_input_mode', 'id')
        ).lower()
        self.hybrid_mm_weight = float(
            _config_value(config, 'hybrid_mm_weight', 0.5)
        )
        self.branch_embedding_dim = (
            2 * self.embedding_dim
            if self.item_input_mode == 'multimodal_concat'
            else self.embedding_dim
        )
        self.ui_branch_mode = str(
            _config_value(config, 'ui_branch_mode', 'dual')
        ).lower()
        self.ui_fusion_mode = str(
            _config_value(config, 'ui_fusion_mode', 'gated_sum')
        ).lower()
        self.ui_gate_mode = str(
            _config_value(config, 'ui_gate_mode', 'separate')
        ).lower()
        self.mm_gate_mode = str(
            _config_value(config, 'mm_gate_mode', 'reuse_ui_item')
        ).lower()
        self.gate_init_mode = str(
            _config_value(config, 'gate_init_mode', 'xavier')
        ).lower()
        self.gate_initial_original_weight = float(
            _config_value(config, 'gate_initial_original_weight', 0.9)
        )

        self.cl_weight = float(_config_value(config, 'cl_weight', 0.5))
        self.cl_temperature = float(
            _config_value(config, 'cl_temperature', 0.2)
        )
        self.aux_bpr_mode = str(
            _config_value(config, 'aux_bpr_mode', 'none')
        ).lower()
        self.aux_bpr_weight = float(
            _config_value(config, 'aux_bpr_weight', 0.0)
        )
        self.mask_relation_mode = str(
            _config_value(config, 'mask_relation_mode', 'none')
        ).lower()
        self.mask_relation_weight = float(
            _config_value(config, 'mask_relation_weight', 0.0)
        )
        self.mask_relation_temperature = float(
            _config_value(config, 'mask_relation_temperature', 1.0)
        )
        self.mask_relation_pairs_per_user = int(
            _config_value(config, 'mask_relation_pairs_per_user', 32)
        )
        self.mask_relation_min_history = int(
            _config_value(config, 'mask_relation_min_history', 2)
        )
        self.mask_relation_min_relevance_gap = float(
            _config_value(config, 'mask_relation_min_relevance_gap', 0.05)
        )
        self.mask_relation_user_ratio = float(
            _config_value(config, 'mask_relation_user_ratio', 1.0)
        )
        self.mask_relation_max_users = int(
            _config_value(config, 'mask_relation_max_users', 0)
        )
        self.mask_relation_warmup_epochs = int(
            _config_value(config, 'mask_relation_warmup_epochs', 0)
        )
        self.mask_relation_seed = int(
            _config_value(config, 'mask_relation_seed', 20000)
        )
        self._validate_masked_config()

        self._initialize_item_input_modules()
        self._initialize_masked_embedding_tables()
        self._initialize_fusion_modules()
        self._initialize_mask_graph()
        self._initialize_mask_relation_history()

        # U-I and I-I outputs always share a dimension, so the FREEDOM
        # residual remains valid. multimodal_concat deliberately keeps 2d.
        self.final_embedding_dim = (
            2 * self.branch_embedding_dim
            if self.ui_fusion_mode == 'gloria_concat'
            and self.ui_branch_mode == 'dual'
            else self.branch_embedding_dim
        )
        if self.image_embedding is not None:
            self.image_aux_projection = self._make_aux_projection(
                self.final_embedding_dim
            )
        if self.text_embedding is not None:
            self.text_aux_projection = self._make_aux_projection(
                self.final_embedding_dim
            )
        self.latest_loss_components = {}
        self.mask_relation_epoch = -1
        self._mask_relation_rng = random.Random(self.mask_relation_seed)

    def _validate_masked_config(self):
        if not 0.0 < self.mask_keep_ratio < 1.0:
            raise ValueError('mask_keep_ratio must be between 0 and 1.')
        if self.mask_weight < 0.0 or self.mask_binary_weight < 0.0:
            raise ValueError('Mask loss weights cannot be negative.')
        if self.mask_degree_mode not in {'full', 'masked'}:
            raise ValueError(
                "mask_degree_mode must be either 'full' or 'masked'."
            )
        if self.mask_graph_mode not in self.MASK_GRAPH_MODES:
            raise ValueError(
                'Unsupported mask_graph_mode: {}.'.format(
                    self.mask_graph_mode
                )
            )
        if self.hard_mask_temperature <= 0.0:
            raise ValueError('hard_mask_temperature must be positive.')
        if self.user_embedding_mode not in {'shared', 'separate'}:
            raise ValueError(
                "user_embedding_mode must be 'shared' or 'separate'."
            )
        if self.item_embedding_mode not in {'shared', 'separate'}:
            raise ValueError(
                "item_embedding_mode must be 'shared' or 'separate'."
            )
        if self.item_input_mode not in self.ITEM_INPUT_MODES:
            raise ValueError(
                'Unsupported item_input_mode: {}.'.format(
                    self.item_input_mode
                )
            )
        if self.item_input_mode != 'id':
            if self.image_embedding is None and self.text_embedding is None:
                raise ValueError(
                    'Multimodal item input requires image or text features.'
                )
            if self.item_embedding_mode != 'shared':
                raise ValueError(
                    "item_embedding_mode must be 'shared' when "
                    "item_input_mode uses multimodal features."
                )
        if self.item_input_mode == 'multimodal_concat' and (
            self.image_embedding is None or self.text_embedding is None
        ):
            raise ValueError(
                'multimodal_concat requires both image and text features.'
            )
        if not 0.0 < self.hybrid_mm_weight < 1.0:
            raise ValueError('hybrid_mm_weight must be between 0 and 1.')
        if self.ui_branch_mode not in {'dual', 'masked_only'}:
            raise ValueError(
                "ui_branch_mode must be 'dual' or 'masked_only'."
            )
        if self.ui_fusion_mode not in {
            'gated_sum', 'gated_concat', 'gloria_concat'
        }:
            raise ValueError(
                "ui_fusion_mode must be 'gated_sum', 'gated_concat', "
                "or 'gloria_concat'."
            )
        if (
            self.ui_fusion_mode == 'gloria_concat'
            and self.ui_branch_mode != 'dual'
        ):
            raise ValueError(
                "ui_fusion_mode 'gloria_concat' requires "
                "ui_branch_mode 'dual'."
            )
        if self.ui_gate_mode not in {'shared', 'separate'}:
            raise ValueError(
                "ui_gate_mode must be 'shared' or 'separate'."
            )
        if self.mm_gate_mode not in {'reuse_ui_item', 'separate'}:
            raise ValueError(
                "mm_gate_mode must be 'reuse_ui_item' or 'separate'."
            )
        if self.gate_init_mode not in {'xavier', 'constant'}:
            raise ValueError(
                "gate_init_mode must be 'xavier' or 'constant'."
            )
        if not 0.0 < self.gate_initial_original_weight < 1.0:
            raise ValueError(
                'gate_initial_original_weight must be between 0 and 1.'
            )
        if self.cl_weight < 0.0:
            raise ValueError('cl_weight cannot be negative.')
        if self.cl_temperature <= 0.0:
            raise ValueError('cl_temperature must be positive.')
        if self.aux_bpr_mode not in {'none', 'branches'}:
            raise ValueError(
                "aux_bpr_mode must be 'none' or 'branches'."
            )
        if self.aux_bpr_weight < 0.0:
            raise ValueError('aux_bpr_weight cannot be negative.')
        if (
            self.aux_bpr_mode == 'branches'
            and self.ui_branch_mode != 'dual'
        ):
            raise ValueError(
                "aux_bpr_mode 'branches' requires ui_branch_mode 'dual'."
            )
        if self.mask_relation_mode not in {'none', 'masked_gcn'}:
            raise ValueError(
                "mask_relation_mode must be 'none' or 'masked_gcn'."
            )
        if self.mask_relation_weight < 0.0:
            raise ValueError('mask_relation_weight cannot be negative.')
        if self.mask_relation_temperature <= 0.0:
            raise ValueError('mask_relation_temperature must be positive.')
        if self.mask_relation_pairs_per_user <= 0:
            raise ValueError(
                'mask_relation_pairs_per_user must be positive.'
            )
        if self.mask_relation_min_history < 2:
            raise ValueError('mask_relation_min_history must be at least 2.')
        if self.mask_relation_min_relevance_gap < 0.0:
            raise ValueError(
                'mask_relation_min_relevance_gap cannot be negative.'
            )
        if not 0.0 < self.mask_relation_user_ratio <= 1.0:
            raise ValueError(
                'mask_relation_user_ratio must be in (0, 1].'
            )
        if self.mask_relation_max_users < 0:
            raise ValueError('mask_relation_max_users cannot be negative.')
        if self.mask_relation_warmup_epochs < 0:
            raise ValueError(
                'mask_relation_warmup_epochs cannot be negative.'
            )
        if (
            self.mask_relation_mode != 'none'
            and self.mask_relation_weight > 0.0
            and self.mask_graph_mode not in {'soft', 'hard'}
        ):
            raise ValueError(
                'Mask relation loss requires a learnable soft or hard mask.'
            )

    def _initialize_item_input_modules(self):
        self.item_mm_input_projection = None
        self.image_item_input_projection = None
        self.text_item_input_projection = None
        self.concat_user_image_embedding = None
        self.concat_user_text_embedding = None
        self.hybrid_item_norm = None
        self.register_parameter('hybrid_mm_logit', None)
        if self.item_input_mode == 'id':
            return

        if self.item_input_mode == 'multimodal_concat':
            self.image_item_input_projection = (
                self._new_modality_input_projection()
            )
            self.text_item_input_projection = (
                self._new_modality_input_projection()
            )
            self.concat_user_image_embedding = nn.Embedding(
                self.n_users, self.embedding_dim
            )
            self.concat_user_text_embedding = nn.Embedding(
                self.n_users, self.embedding_dim
            )
            nn.init.xavier_uniform_(
                self.concat_user_image_embedding.weight
            )
            nn.init.xavier_uniform_(
                self.concat_user_text_embedding.weight
            )
            return

        modality_count = int(self.image_embedding is not None)
        modality_count += int(self.text_embedding is not None)
        input_dim = modality_count * self.feat_embed_dim
        if input_dim == self.embedding_dim:
            self.item_mm_input_projection = nn.Identity()
        else:
            self.item_mm_input_projection = nn.Linear(
                input_dim, self.embedding_dim
            )
            nn.init.xavier_uniform_(self.item_mm_input_projection.weight)
            nn.init.zeros_(self.item_mm_input_projection.bias)

        if self.item_input_mode == 'hybrid':
            initial_logit = math.log(
                self.hybrid_mm_weight / (1.0 - self.hybrid_mm_weight)
            )
            self.hybrid_mm_logit = nn.Parameter(
                torch.tensor(initial_logit, dtype=torch.float32)
            )
            self.hybrid_item_norm = nn.LayerNorm(self.embedding_dim)

    def _new_modality_input_projection(self):
        if self.feat_embed_dim == self.embedding_dim:
            return nn.Identity()
        projection = nn.Linear(self.feat_embed_dim, self.embedding_dim)
        nn.init.xavier_uniform_(projection.weight)
        nn.init.zeros_(projection.bias)
        return projection

    def _initialize_masked_embedding_tables(self):
        self.masked_user_image_embedding = None
        self.masked_user_text_embedding = None
        if (
            self.item_input_mode == 'multimodal_concat'
            and self.user_embedding_mode == 'separate'
        ):
            self.masked_user_embedding = None
            self.masked_user_image_embedding = nn.Embedding(
                self.n_users, self.embedding_dim
            )
            self.masked_user_text_embedding = nn.Embedding(
                self.n_users, self.embedding_dim
            )
            nn.init.xavier_uniform_(
                self.masked_user_image_embedding.weight
            )
            nn.init.xavier_uniform_(
                self.masked_user_text_embedding.weight
            )
        elif self.user_embedding_mode == 'separate':
            self.masked_user_embedding = nn.Embedding(
                self.n_users, self.embedding_dim
            )
            nn.init.xavier_uniform_(self.masked_user_embedding.weight)
        else:
            self.masked_user_embedding = None

        if self.item_embedding_mode == 'separate':
            self.masked_item_id_embedding = nn.Embedding(
                self.n_items, self.embedding_dim
            )
            nn.init.xavier_uniform_(self.masked_item_id_embedding.weight)
        else:
            self.masked_item_id_embedding = None

    def _multimodal_item_input(self):
        modality_inputs = []
        if self.image_embedding is not None:
            image_input = F.normalize(
                self.image_trs(self.image_embedding.weight), dim=-1
            )
            if self.item_input_mode == 'multimodal_concat':
                image_input = self.image_item_input_projection(image_input)
            modality_inputs.append(image_input)
        if self.text_embedding is not None:
            text_input = F.normalize(
                self.text_trs(self.text_embedding.weight), dim=-1
            )
            if self.item_input_mode == 'multimodal_concat':
                text_input = self.text_item_input_projection(text_input)
            modality_inputs.append(text_input)
        multimodal_input = torch.cat(modality_inputs, dim=-1)
        if self.item_input_mode == 'multimodal_concat':
            return multimodal_input
        return self.item_mm_input_projection(multimodal_input)

    def _original_item_table(self):
        if self.item_input_mode == 'id':
            return self.item_id_embedding.weight

        multimodal_items = self._multimodal_item_input()
        if self.item_input_mode in {'multimodal', 'multimodal_concat'}:
            return multimodal_items

        mm_weight = torch.sigmoid(self.hybrid_mm_logit)
        return self.hybrid_item_norm(
            self.item_id_embedding.weight + mm_weight * multimodal_items
        )

    def _new_gate(self, embedding_dim):
        gate = nn.Linear(2 * embedding_dim, embedding_dim)
        if self.gate_init_mode == 'constant':
            initial_bias = math.log(
                self.gate_initial_original_weight
                / (1.0 - self.gate_initial_original_weight)
            )
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, initial_bias)
        else:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)
        return gate

    @staticmethod
    def _new_concat_projection(embedding_dim):
        projection = nn.Linear(2 * embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(projection.weight)
        nn.init.zeros_(projection.bias)
        return projection

    def _initialize_fusion_modules(self):
        self.fusion_gate = None
        self.user_fusion_gate = None
        self.item_fusion_gate = None
        self.fusion_projection = None
        self.user_concat_projection = None
        self.item_concat_projection = None
        self.mm_fusion_gate = None

        if self.ui_branch_mode != 'dual':
            return
        if self.ui_fusion_mode == 'gloria_concat':
            return

        if self.ui_gate_mode == 'shared':
            self.fusion_gate = self._new_gate(self.branch_embedding_dim)
            if self.ui_fusion_mode == 'gated_concat':
                self.fusion_projection = self._new_concat_projection(
                    self.branch_embedding_dim
                )
        else:
            self.user_fusion_gate = self._new_gate(
                self.branch_embedding_dim
            )
            self.item_fusion_gate = self._new_gate(
                self.branch_embedding_dim
            )
            if self.ui_fusion_mode == 'gated_concat':
                self.user_concat_projection = (
                    self._new_concat_projection(self.branch_embedding_dim)
                )
                self.item_concat_projection = (
                    self._new_concat_projection(self.branch_embedding_dim)
                )

        if (
            self.item_embedding_mode == 'separate'
            and self.mm_gate_mode == 'separate'
        ):
            self.mm_fusion_gate = self._new_gate(self.branch_embedding_dim)

    def _initialize_mask_graph(self):
        initial_logit = math.log(
            self.mask_keep_ratio / (1.0 - self.mask_keep_ratio)
        )
        if self.mask_graph_mode in {
            'double_full',
            'svd',
            'local_prunning',
            'random_fixed',
            'random_dynamic',
        }:
            self.register_parameter('mask_logits', None)
        else:
            self.mask_logits = nn.Parameter(
                torch.full((self.num_interactions,), initial_logit)
            )

        self.register_buffer(
            'hard_train_indices',
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'hard_eval_indices',
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

        random_mode = self.mask_graph_mode in {
            'random_fixed', 'random_dynamic'
        }
        if random_mode:
            random_train_indices = self._sample_seeded_random_indices(0)
            if self.mask_graph_mode == 'random_fixed':
                random_eval_indices = random_train_indices.clone()
            else:
                random_eval_indices = self._sample_seeded_random_indices(1)
        else:
            random_train_indices = torch.empty(0, dtype=torch.long)
            random_eval_indices = torch.empty(0, dtype=torch.long)
        self.register_buffer(
            'random_train_indices',
            random_train_indices,
            persistent=random_mode,
        )
        self.register_buffer(
            'random_eval_indices',
            random_eval_indices,
            persistent=random_mode,
        )

        self.register_buffer('svd_adj', None)
        self.register_buffer('local_pruned_adj', None)
        if self.mask_graph_mode == 'svd':
            self.svd_adj = self._svd_subgraph_extraction(self.norm_adj)
        elif self.mask_graph_mode == 'local_prunning':
            self.local_pruned_adj = self._sample_local_pruned_adjacency()

    def _initialize_mask_relation_history(self):
        """Index every training-history edge by user without Python lists."""
        forward_edges = self.ui_edge_index[:, :self.num_interactions]
        forward_users = forward_edges[0]
        history_order = torch.argsort(forward_users)
        user_counts = torch.bincount(
            forward_users, minlength=self.n_users
        )
        history_ptr = torch.cat((
            user_counts.new_zeros(1),
            torch.cumsum(user_counts, dim=0),
        ))
        self.register_buffer(
            'mask_relation_history_order', history_order, persistent=False
        )
        self.register_buffer(
            'mask_relation_history_ptr', history_ptr, persistent=False
        )
        self.register_buffer(
            'mask_relation_forward_items',
            forward_edges[1] - self.n_users,
            persistent=False,
        )

    def _svd_subgraph_extraction(self, adjacency):
        adjacency = adjacency.coalesce().cpu()
        indices = adjacency.indices().numpy()
        values = adjacency.values().numpy().astype(np.float64, copy=False)
        scipy_adjacency = sp.coo_matrix(
            (values, (indices[0], indices[1])),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsc()

        rank = min(self.embedding_dim, self.n_nodes - 1)
        if rank < 1:
            raise ValueError('SVD masking requires at least two graph nodes.')
        left, singular_values, right = svds(scipy_adjacency, k=rank)
        order = np.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        left = left[:, order]
        right = right[order, :]

        selected_rank = max(1, int(0.25 * rank))
        products = (
            singular_values[:selected_rank]
            * singular_values[-selected_rank:]
        )
        product_matrix = (
            left[:, :selected_rank]
            @ np.diag(products)
            @ right[:selected_rank, :]
        )
        product_matrix *= np.abs(product_matrix) >= 1e-3
        product_sparse = sp.coo_matrix(product_matrix)
        svd_indices = torch.from_numpy(
            np.vstack((product_sparse.row, product_sparse.col)).astype(
                np.int64, copy=False
            )
        ).to(self.ui_edge_index.device)
        svd_values = torch.from_numpy(
            product_sparse.data.astype(np.float32, copy=False)
        ).to(self.full_norm_edge_weights.device)
        return torch.sparse_coo_tensor(
            svd_indices,
            svd_values,
            (self.n_nodes, self.n_nodes),
            device=svd_values.device,
        ).coalesce()

    @property
    def hard_keep_count(self):
        return max(
            1,
            min(
                self.num_interactions,
                int(round(self.num_interactions * self.mask_keep_ratio)),
            ),
        )

    @property
    def random_keep_count(self):
        return self.hard_keep_count

    @property
    def local_keep_count(self):
        return max(
            1,
            min(
                self.num_interactions,
                int(self.num_interactions * self.mask_keep_ratio),
            ),
        )

    def _normalized_ui_edge_weights(self, edge_weights, edge_index=None):
        if edge_index is None:
            edge_index = self.ui_edge_index
        row, col = edge_index
        degree = torch.zeros(
            self.n_nodes,
            dtype=edge_weights.dtype,
            device=edge_weights.device,
        )
        degree = degree.index_add(0, row, edge_weights)
        inverse_sqrt = degree.clamp_min(1e-12).pow(-0.5)
        inverse_sqrt = torch.where(
            degree > 0, inverse_sqrt, torch.zeros_like(inverse_sqrt)
        )
        return inverse_sqrt[row] * edge_weights * inverse_sqrt[col]

    def _ui_adjacency_from_weights(self, edge_weights, edge_index=None):
        if edge_index is None:
            edge_index = self.ui_edge_index
        return torch.sparse_coo_tensor(
            edge_index,
            edge_weights,
            (self.n_nodes, self.n_nodes),
            device=edge_weights.device,
        ).coalesce()

    def _normalized_ui_adjacency(self, edge_weights, edge_index=None):
        return self._ui_adjacency_from_weights(
            self._normalized_ui_edge_weights(edge_weights, edge_index),
            edge_index,
        )

    @torch.no_grad()
    def _sample_local_pruned_adjacency(self):
        kept = torch.multinomial(
            self.edge_values,
            self.local_keep_count,
            replacement=False,
        )
        kept_undirected = torch.cat(
            (kept, kept + self.num_interactions), dim=0
        )
        edge_index = self.ui_edge_index[:, kept_undirected]
        edge_weights = torch.ones(
            edge_index.shape[1],
            dtype=self.full_norm_edge_weights.dtype,
            device=edge_index.device,
        )
        return self._normalized_ui_adjacency(edge_weights, edge_index)

    @torch.no_grad()
    def _sample_seeded_random_indices(self, seed_offset=0):
        generator = torch.Generator()
        generator.manual_seed(self.random_mask_seed + seed_offset)
        indices = torch.randperm(
            self.num_interactions, generator=generator
        )[:self.random_keep_count]
        return indices.to(self.ui_edge_index.device)

    @torch.no_grad()
    def _sample_dynamic_random_indices(self):
        return torch.randperm(
            self.num_interactions,
            device=self.ui_edge_index.device,
        )[:self.random_keep_count]

    @torch.no_grad()
    def _sample_hard_train_indices(self, logits):
        uniform = torch.rand_like(logits).clamp_(1e-8, 1.0 - 1e-8)
        gumbel = -torch.log(-torch.log(uniform))
        scores = logits / self.hard_mask_temperature + gumbel
        return torch.topk(
            scores, self.hard_keep_count, sorted=False
        ).indices

    @torch.no_grad()
    def _select_hard_eval_indices(self, logits):
        return torch.topk(
            logits, self.hard_keep_count, sorted=False
        ).indices

    def pre_epoch_processing(self):
        self.mask_relation_epoch += 1
        self._mask_relation_rng.seed(
            self.mask_relation_seed + self.mask_relation_epoch
        )
        self._resample_freedom_adjacency()
        if self.mask_graph_mode == 'hard':
            self.hard_train_indices = self._sample_hard_train_indices(
                self.mask_logits
            )
        elif self.mask_graph_mode == 'local_prunning':
            self.local_pruned_adj = self._sample_local_pruned_adjacency()
        elif self.mask_graph_mode == 'random_dynamic':
            self.random_train_indices = (
                self._sample_dynamic_random_indices()
            )

    def post_epoch_processing(self):
        if self.mask_graph_mode == 'hard':
            self.hard_eval_indices = self._select_hard_eval_indices(
                self.mask_logits
            )

    def _current_hard_indices(self):
        if self.training:
            if self.hard_train_indices.numel() == 0:
                self.hard_train_indices = self._sample_hard_train_indices(
                    self.mask_logits
                )
            return self.hard_train_indices
        if self.hard_eval_indices.numel() == 0:
            self.hard_eval_indices = self._select_hard_eval_indices(
                self.mask_logits
            )
        return self.hard_eval_indices

    def _hard_masked_ui_adjacency(self, probabilities):
        kept = self._current_hard_indices()
        kept_undirected = torch.cat(
            (kept, kept + self.num_interactions), dim=0
        )
        edge_index = self.ui_edge_index[:, kept_undirected]
        selected_soft = probabilities[kept]
        straight_through = (
            torch.ones_like(selected_soft)
            + selected_soft
            - selected_soft.detach()
        )
        undirected_mask = torch.cat(
            (straight_through, straight_through), dim=0
        )
        if self.mask_degree_mode == 'full':
            return self._ui_adjacency_from_weights(
                self.full_norm_edge_weights[kept_undirected]
                * undirected_mask,
                edge_index,
            )
        return self._normalized_ui_adjacency(
            undirected_mask, edge_index
        )

    def _random_masked_ui_adjacency(self):
        kept = (
            self.random_train_indices
            if self.training
            else self.random_eval_indices
        )
        kept_undirected = torch.cat(
            (kept, kept + self.num_interactions), dim=0
        )
        edge_index = self.ui_edge_index[:, kept_undirected]
        if self.mask_degree_mode == 'full':
            return self._ui_adjacency_from_weights(
                self.full_norm_edge_weights[kept_undirected], edge_index
            )
        edge_weights = torch.ones(
            edge_index.size(1),
            dtype=self.full_norm_edge_weights.dtype,
            device=edge_index.device,
        )
        return self._normalized_ui_adjacency(edge_weights, edge_index)

    def _masked_ui_adjacency(self):
        if self.mask_graph_mode == 'double_full':
            return self.norm_adj, None
        if self.mask_graph_mode in {'random_fixed', 'random_dynamic'}:
            return self._random_masked_ui_adjacency(), None
        if self.mask_graph_mode == 'svd':
            return self.svd_adj, None
        if self.mask_graph_mode == 'local_prunning':
            if not self.training:
                return self.norm_adj, None
            if self.local_pruned_adj is None:
                self.local_pruned_adj = (
                    self._sample_local_pruned_adjacency()
                )
            return self.local_pruned_adj, None

        probabilities = torch.sigmoid(self.mask_logits)
        if self.mask_graph_mode == 'hard':
            return (
                self._hard_masked_ui_adjacency(probabilities),
                probabilities,
            )

        undirected_mask = torch.cat(
            (probabilities, probabilities), dim=0
        )
        if self.mask_degree_mode == 'full':
            adjacency = self._ui_adjacency_from_weights(
                self.full_norm_edge_weights * undirected_mask
            )
        else:
            adjacency = self._normalized_ui_adjacency(undirected_mask)
        return adjacency, probabilities

    @staticmethod
    def _memory_safe_sparse_mm(adjacency, embeddings):
        adjacency = adjacency.coalesce()
        return _ObservedEdgeSparseMM.apply(
            adjacency.indices(),
            adjacency.values(),
            tuple(adjacency.shape),
            embeddings,
        )

    def _propagate_ui_graph(self, adjacency, initial_embeddings):
        adjacency = adjacency.coalesce()
        differentiable_edges = adjacency.values().requires_grad
        embeddings = [initial_embeddings]
        current = initial_embeddings
        for _ in range(self.n_ui_layers):
            if differentiable_edges:
                current = self._memory_safe_sparse_mm(adjacency, current)
            else:
                current = torch.sparse.mm(adjacency, current)
            embeddings.append(current)
        return torch.stack(embeddings, dim=1).mean(dim=1)

    def _original_user_table(self):
        if self.item_input_mode != 'multimodal_concat':
            return self.user_embedding.weight
        return torch.cat(
            (
                self.concat_user_image_embedding.weight,
                self.concat_user_text_embedding.weight,
            ),
            dim=-1,
        )

    def _masked_user_table(self):
        if self.item_input_mode == 'multimodal_concat':
            if self.user_embedding_mode == 'shared':
                return self._original_user_table()
            return torch.cat(
                (
                    self.masked_user_image_embedding.weight,
                    self.masked_user_text_embedding.weight,
                ),
                dim=-1,
            )
        if self.masked_user_embedding is None:
            return self.user_embedding.weight
        return self.masked_user_embedding.weight

    def _masked_item_table(self, original_item_table):
        if self.item_input_mode != 'id':
            return original_item_table
        if self.masked_item_id_embedding is None:
            return original_item_table
        return self.masked_item_id_embedding.weight

    def _ui_fusion_modules(self, node_type):
        if self.ui_gate_mode == 'shared':
            return self.fusion_gate, self.fusion_projection
        if node_type == 'user':
            return self.user_fusion_gate, self.user_concat_projection
        return self.item_fusion_gate, self.item_concat_projection

    def _fuse_ui_pair(self, original, masked, node_type):
        if self.ui_fusion_mode == 'gloria_concat':
            return torch.cat((original, masked), dim=-1), None
        gate_module, projection = self._ui_fusion_modules(node_type)
        gate = torch.sigmoid(
            gate_module(torch.cat((original, masked), dim=-1))
        )
        gated_original = gate * original
        gated_masked = (1.0 - gate) * masked
        if self.ui_fusion_mode == 'gated_concat':
            fused = projection(
                torch.cat((gated_original, gated_masked), dim=-1)
            )
        else:
            fused = gated_original + gated_masked
        return fused, gate

    def _fuse_mm_items(self, original_items, masked_items, item_gate):
        if self.mm_gate_mode == 'reuse_ui_item':
            mm_gate = item_gate
        else:
            mm_gate = torch.sigmoid(
                self.mm_fusion_gate(
                    torch.cat((original_items, masked_items), dim=-1)
                )
            )
        fused = (
            mm_gate * original_items
            + (1.0 - mm_gate) * masked_items
        )
        return fused, mm_gate

    def _encode(self):
        masked_user_table = self._masked_user_table()
        original_item_table = self._original_item_table()
        masked_item_table = self._masked_item_table(original_item_table)
        masked_initial = torch.cat(
            (masked_user_table, masked_item_table), dim=0
        )
        masked_adjacency, probabilities = self._masked_ui_adjacency()
        masked_embeddings = self._propagate_ui_graph(
            masked_adjacency, masked_initial
        )
        masked_users, masked_ui_items = torch.split(
            masked_embeddings, (self.n_users, self.n_items), dim=0
        )

        if self.ui_branch_mode == 'masked_only':
            masked_mm_items = self._propagate_mm_graph(masked_item_table)
            return {
                'users': masked_users,
                'items': masked_ui_items + masked_mm_items,
                'fused_ui_items': masked_ui_items,
                'mm_items': masked_mm_items,
                'full_users': None,
                'full_items': None,
                'masked_users': masked_users,
                'masked_items': masked_ui_items,
                'full_mm_items': None,
                'masked_mm_items': masked_mm_items,
                'user_gate': None,
                'item_gate': None,
                'mm_gate': None,
                'mask': probabilities,
            }

        original_initial = torch.cat(
            (self._original_user_table(), original_item_table),
            dim=0,
        )
        original_embeddings = self._propagate_ui_graph(
            self._freedom_ui_adjacency(), original_initial
        )
        original_users, original_ui_items = torch.split(
            original_embeddings, (self.n_users, self.n_items), dim=0
        )

        fused_users, user_gate = self._fuse_ui_pair(
            original_users, masked_users, 'user'
        )
        fused_ui_items, item_gate = self._fuse_ui_pair(
            original_ui_items, masked_ui_items, 'item'
        )

        if self.ui_fusion_mode == 'gloria_concat':
            original_mm_items = self._propagate_mm_graph(
                original_item_table
            )
            masked_mm_items = self._propagate_mm_graph(masked_item_table)
            fused_mm_items = torch.cat(
                (original_mm_items, masked_mm_items), dim=-1
            )
            mm_gate = None
        else:
            original_mm_items = self._propagate_mm_graph(
                original_item_table
            )
            if self.item_embedding_mode == 'shared':
                masked_mm_items = original_mm_items
                fused_mm_items = original_mm_items
                mm_gate = None
            else:
                masked_mm_items = self._propagate_mm_graph(
                    self.masked_item_id_embedding.weight
                )
                fused_mm_items, mm_gate = self._fuse_mm_items(
                    original_mm_items, masked_mm_items, item_gate
                )

        return {
            'users': fused_users,
            'items': fused_ui_items + fused_mm_items,
            'fused_ui_items': fused_ui_items,
            'mm_items': fused_mm_items,
            'full_users': original_users,
            'full_items': original_ui_items,
            'masked_users': masked_users,
            'masked_items': masked_ui_items,
            'full_mm_items': original_mm_items,
            'masked_mm_items': masked_mm_items,
            'user_gate': user_gate,
            'item_gate': item_gate,
            'mm_gate': mm_gate,
            'mask': probabilities,
        }

    def forward(self):
        representations = self._encode()
        return representations['users'], representations['items']

    def info_nce(self, first_view, second_view):
        first_view = F.normalize(first_view, dim=1)
        second_view = F.normalize(second_view, dim=1)
        logits = torch.matmul(first_view, second_view.transpose(0, 1))
        logits = logits / self.cl_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )

    def _contrastive_loss(self, representations, users, positive_items):
        if self.cl_weight == 0.0 or self.ui_branch_mode != 'dual':
            return representations['users'].new_zeros(())
        unique_users = torch.unique(users)
        unique_items = torch.unique(positive_items)
        user_loss = self.info_nce(
            representations['full_users'][unique_users],
            representations['masked_users'][unique_users],
        )
        item_loss = self.info_nce(
            representations['full_items'][unique_items],
            representations['masked_items'][unique_items],
        )
        return 0.5 * (user_loss + item_loss)

    def _auxiliary_losses(
        self, all_users, users, positive_items, negative_items
    ):
        zero = all_users.new_zeros(())
        visual_loss = zero
        text_loss = zero
        if self.image_embedding is not None:
            image_features = self.image_aux_projection(
                self.image_trs(self.image_embedding.weight)
            )
            visual_loss = self.bpr_loss(
                all_users[users],
                image_features[positive_items],
                image_features[negative_items],
            )
        if self.text_embedding is not None:
            text_features = self.text_aux_projection(
                self.text_trs(self.text_embedding.weight)
            )
            text_loss = self.bpr_loss(
                all_users[users],
                text_features[positive_items],
                text_features[negative_items],
            )
        return visual_loss, text_loss

    def _branch_bpr_losses(
        self, representations, users, positive_items, negative_items
    ):
        zero = representations['users'].new_zeros(())
        if self.aux_bpr_mode == 'none' or self.aux_bpr_weight == 0.0:
            return zero, zero, zero

        original_items = (
            representations['full_items']
            + representations['full_mm_items']
        )
        masked_items = (
            representations['masked_items']
            + representations['masked_mm_items']
        )
        original_loss = self.bpr_loss(
            representations['full_users'][users],
            original_items[positive_items],
            original_items[negative_items],
        )
        masked_loss = self.bpr_loss(
            representations['masked_users'][users],
            masked_items[positive_items],
            masked_items[negative_items],
        )
        branch_loss = 0.5 * (original_loss + masked_loss)
        return branch_loss, original_loss, masked_loss

    def _mask_regularization(self, probabilities, reference_loss):
        if probabilities is None:
            zero = reference_loss.new_zeros(())
            return zero, zero, zero, reference_loss.new_ones(())
        mask_mean = probabilities.mean()
        budget_loss = (mask_mean - self.mask_keep_ratio).pow(2)
        binary_loss = (
            probabilities * (1.0 - probabilities)
        ).mean()
        regularization = (
            budget_loss + self.mask_binary_weight * binary_loss
        )
        return regularization, budget_loss, binary_loss, mask_mean

    def _sample_relation_users(self, users):
        unique_users = torch.unique(users.detach()).cpu().tolist()
        sample_count = max(
            1, int(math.ceil(
                len(unique_users) * self.mask_relation_user_ratio
            ))
        )
        if self.mask_relation_max_users > 0:
            sample_count = min(sample_count, self.mask_relation_max_users)
        if sample_count < len(unique_users):
            unique_users = self._mask_relation_rng.sample(
                unique_users, sample_count
            )
        return unique_users

    def _sample_relation_pairs(self, history_size):
        total_pairs = history_size * (history_size - 1) // 2
        target_count = min(
            total_pairs, self.mask_relation_pairs_per_user
        )
        if target_count == total_pairs:
            return [
                (left, right)
                for left in range(history_size)
                for right in range(left + 1, history_size)
            ]

        pairs = set()
        while len(pairs) < target_count:
            left = self._mask_relation_rng.randrange(history_size)
            right = self._mask_relation_rng.randrange(history_size - 1)
            if right >= left:
                right += 1
            if left > right:
                left, right = right, left
            pairs.add((left, right))
        return list(pairs)

    def _mask_relation_loss(self, representations, users, reference_loss):
        """Rank a user's edge logits by masked-GCN history relevance.

        Relevance is detached deliberately: it acts as a per-step teacher, so
        this auxiliary objective updates the edge mask rather than changing
        item representations to make its own targets easier.
        """
        zero = reference_loss.new_zeros(())
        if (
            self.mask_relation_mode == 'none'
            or self.mask_relation_weight == 0.0
        ):
            return zero, 0, zero

        masked_items = representations['masked_items'].detach()
        user_losses = []
        accepted_pairs = 0
        relevance_gaps = []
        for user_id in self._sample_relation_users(users):
            start = int(self.mask_relation_history_ptr[user_id].item())
            end = int(self.mask_relation_history_ptr[user_id + 1].item())
            if end - start < self.mask_relation_min_history:
                continue

            edge_ids = self.mask_relation_history_order[start:end]
            item_ids = self.mask_relation_forward_items[edge_ids]
            history_items = masked_items.index_select(0, item_ids)
            preference = history_items.mean(dim=0, keepdim=True)
            relevance = F.cosine_similarity(
                history_items,
                preference.expand_as(history_items),
                dim=-1,
            ).detach()

            pair_losses = []
            for left, right in self._sample_relation_pairs(end - start):
                relevance_gap = relevance[left] - relevance[right]
                if (
                    relevance_gap.abs().item()
                    <= self.mask_relation_min_relevance_gap
                ):
                    continue
                direction = relevance_gap.sign()
                mask_gap = (
                    self.mask_logits[edge_ids[left]]
                    - self.mask_logits[edge_ids[right]]
                )
                pair_losses.append(F.softplus(
                    -self.mask_relation_temperature
                    * direction
                    * mask_gap
                ))
                relevance_gaps.append(relevance_gap.abs())
                accepted_pairs += 1
            if pair_losses:
                # Give each user equal weight regardless of history length.
                user_losses.append(torch.stack(pair_losses).mean())

        if not user_losses:
            return zero, 0, zero
        relation_loss = torch.stack(user_losses).mean()
        mean_gap = torch.stack(relevance_gaps).mean()
        if self.mask_relation_warmup_epochs > 0:
            epoch = max(self.mask_relation_epoch, 0)
            warmup = min(
                1.0,
                float(epoch + 1) / self.mask_relation_warmup_epochs,
            )
            relation_loss = relation_loss * warmup
        return relation_loss, accepted_pairs, mean_gap

    def calculate_loss(self, interaction):
        users, positive_items, negative_items = interaction[:3]
        representations = self._encode()
        all_users = representations['users']
        all_items = representations['items']

        ranking_loss = self.bpr_loss(
            all_users[users],
            all_items[positive_items],
            all_items[negative_items],
        )
        visual_loss, text_loss = self._auxiliary_losses(
            all_users, users, positive_items, negative_items
        )
        contrastive_loss = self._contrastive_loss(
            representations, users, positive_items
        )
        (
            branch_bpr_loss,
            original_branch_bpr_loss,
            masked_branch_bpr_loss,
        ) = self._branch_bpr_losses(
            representations, users, positive_items, negative_items
        )
        (
            mask_regularization,
            budget_loss,
            binary_loss,
            mask_mean,
        ) = self._mask_regularization(
            representations['mask'], ranking_loss
        )
        (
            mask_relation_loss,
            mask_relation_pairs,
            mask_relation_gap,
        ) = self._mask_relation_loss(
            representations, users, ranking_loss
        )

        total_loss = (
            ranking_loss
            + self.aux_bpr_weight * branch_bpr_loss
            + self.reg_weight * (visual_loss + text_loss)
            + self.cl_weight * contrastive_loss
            + self.mask_weight * mask_regularization
            + self.mask_relation_weight * mask_relation_loss
        )
        self.latest_loss_components = {
            'bpr': ranking_loss.detach(),
            'aux_bpr': branch_bpr_loss.detach(),
            'original_branch_bpr': original_branch_bpr_loss.detach(),
            'masked_branch_bpr': masked_branch_bpr_loss.detach(),
            'visual_bpr': visual_loss.detach(),
            'text_bpr': text_loss.detach(),
            'contrastive': contrastive_loss.detach(),
            'mask': mask_regularization.detach(),
            'mask_budget': budget_loss.detach(),
            'mask_binary': binary_loss.detach(),
            'mask_mean': mask_mean.detach(),
            'mask_relation': mask_relation_loss.detach(),
            'mask_relation_pairs': mask_relation_pairs,
            'mask_relation_mean_gap': mask_relation_gap.detach(),
        }
        return total_loss

    def full_sort_predict(self, interaction):
        users, items = self.forward()
        return torch.matmul(
            users[interaction[0]], items.transpose(0, 1)
        )

    @torch.no_grad()
    def get_analysis_artifacts(self):
        was_training = self.training
        self.eval()
        representations = self._encode()
        forward_edges = self.ui_edge_index[:, :self.num_interactions]

        masks = {}
        if self.mask_logits is not None:
            probabilities = torch.sigmoid(self.mask_logits)
            selected_indices = self._select_hard_eval_indices(
                self.mask_logits
            )
            selected = torch.zeros_like(probabilities, dtype=torch.bool)
            selected[selected_indices] = True
            masks['masked_branch'] = {
                'logits': self.mask_logits.detach().cpu(),
                'probabilities': probabilities.detach().cpu(),
                'selected_at_keep_ratio': selected.cpu(),
            }
        if self.mask_graph_mode in {'random_fixed', 'random_dynamic'}:
            train_selected = torch.zeros(
                self.num_interactions, dtype=torch.bool
            )
            eval_selected = torch.zeros(
                self.num_interactions, dtype=torch.bool
            )
            train_selected[self.random_train_indices.cpu()] = True
            eval_selected[self.random_eval_indices.cpu()] = True
            masks['random_branch'] = {
                'train_selected': train_selected,
                'selected_at_keep_ratio': eval_selected,
            }

        embedding_tables = {
            name + '.weight': module.weight.detach().cpu()
            for name, module in self.named_modules()
            if isinstance(module, nn.Embedding)
            and module.weight.requires_grad
        }
        exported_representations = {
            name: value.detach().cpu()
            for name, value in representations.items()
            if torch.is_tensor(value) and name != 'mask'
        }
        latest_losses = {
            name: (
                value.detach().cpu()
                if torch.is_tensor(value)
                else value
            )
            for name, value in self.latest_loss_components.items()
        }
        artifacts = {
            'metadata': {
                'model': self.__class__.__name__,
                'architecture': 'freedom_residual_dual_ui',
                'mask_graph_mode': self.mask_graph_mode,
                'mask_degree_mode': self.mask_degree_mode,
                'ui_branch_mode': self.ui_branch_mode,
                'ui_fusion_mode': self.ui_fusion_mode,
                'ui_gate_mode': self.ui_gate_mode,
                'mm_gate_mode': self.mm_gate_mode,
                'gate_init_mode': self.gate_init_mode,
                'gate_initial_original_weight': (
                    self.gate_initial_original_weight
                ),
                'user_embedding_mode': self.user_embedding_mode,
                'item_embedding_mode': self.item_embedding_mode,
                'item_input_mode': self.item_input_mode,
                'hybrid_mm_weight': (
                    float(torch.sigmoid(self.hybrid_mm_logit).item())
                    if self.hybrid_mm_logit is not None
                    else self.hybrid_mm_weight
                ),
                'mask_keep_ratio': self.mask_keep_ratio,
                'random_mask_seed': self.random_mask_seed,
                'dropout': self.dropout,
                'embedding_dim': self.embedding_dim,
                'branch_embedding_dim': self.branch_embedding_dim,
                'final_embedding_dim': self.final_embedding_dim,
                'cl_weight': self.cl_weight,
                'cl_temperature': self.cl_temperature,
                'aux_bpr_mode': self.aux_bpr_mode,
                'aux_bpr_weight': self.aux_bpr_weight,
                'mask_relation_mode': self.mask_relation_mode,
                'mask_relation_weight': self.mask_relation_weight,
                'mask_relation_temperature': (
                    self.mask_relation_temperature
                ),
                'mask_relation_epoch': self.mask_relation_epoch,
                'num_users': self.n_users,
                'num_items': self.n_items,
                'num_interactions': self.num_interactions,
            },
            'ui_edges': {
                'user_ids': forward_edges[0].detach().cpu(),
                'item_ids': (
                    forward_edges[1] - self.n_users
                ).detach().cpu(),
            },
            'masks': masks,
            'embedding_tables': embedding_tables,
            'representations': exported_representations,
            'latest_loss_components': latest_losses,
        }
        if was_training:
            self.train()
        return artifacts
