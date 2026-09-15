r"""
Memory-efficient PGL with two-branch masked user-item graph learning.

The two branches share the same initial node embeddings. The second branch
can use a soft mask, a hard sparse mask, an SVD graph, a locally pruned graph,
or the complete graph as an ablation. Local pruning follows the original PGL:
the subgraph is used for training and the complete graph for inference. Masked
modes can use either full-graph degrees or degrees recomputed from the masked
weights. Branch outputs use either a gated sum or a gated concatenation
projected back to the branch dimension before adding the multimodal item-item
representation.

Sparse propagation uses an edge-only custom backward whenever the adjacency
values require gradients. This avoids allocating a dense
``(n_users + n_items) ** 2`` adjacency gradient while preserving the native
``torch.sparse.mm`` forward and its gradients on observed edges.
"""

import math
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds

from common.abstract_recommender import GeneralRecommender


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
            output_gradients = grad_output.index_select(0, row)
            source_embeddings = embeddings.index_select(0, col)
            grad_values = (output_gradients * source_embeddings).sum(dim=1)

        grad_embeddings = None
        if ctx.needs_input_grad[3]:
            transpose_indices = torch.stack((col, row), dim=0)
            transpose_size = (
                ctx.adjacency_size[1], ctx.adjacency_size[0]
            )
            transpose_adjacency = torch.sparse_coo_tensor(
                transpose_indices,
                values,
                transpose_size,
                dtype=values.dtype,
                device=values.device,
            ).coalesce()
            grad_embeddings = torch.sparse.mm(
                transpose_adjacency, grad_output
            )

        return None, grad_values, None, grad_embeddings


def _config_value(config, key, default):
    value = config[key]
    return default if value is None else value


class PGL_MASKED(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PGL_MASKED, self).__init__(config, dataset)

        self.embedding_dim = _config_value(config, 'embedding_size', 64)
        self.feat_embed_dim = _config_value(config, 'feat_embed_dim', 64)
        self.knn_k = _config_value(config, 'knn_k', 10)
        self.n_mm_layers = _config_value(config, 'n_mm_layers', 1)
        self.n_ui_layers = _config_value(config, 'n_ui_layers', 2)
        self.mm_image_weight = _config_value(config, 'mm_image_weight', 0.1)

        self.cl_weight = _config_value(config, 'cl_weight', 0.05)
        self.cl_temperature = _config_value(config, 'cl_temperature', 0.2)
        self.cl_dropout = _config_value(config, 'dropout', 0.2)
        self.cl_mode = str(
            _config_value(config, 'cl_mode', 'auto')
        ).lower()
        self.mask_weight = _config_value(config, 'mask_weight', 0.1)
        self.mask_keep_ratio = _config_value(config, 'mask_keep_ratio', 0.3)
        self.mask_binary_weight = _config_value(
            config, 'mask_binary_weight', 0.1
        )
        self.mask_degree_mode = str(
            _config_value(config, 'mask_degree_mode', 'masked')
        ).lower()
        self.mask_graph_mode = str(
            _config_value(config, 'mask_graph_mode', 'soft')
        ).lower()
        self.hard_mask_temperature = _config_value(
            config, 'hard_mask_temperature', 1.0
        )
        self.random_mask_seed = int(
            _config_value(config, 'random_mask_seed', 999)
        )
        self.user_embedding_mode = str(
            _config_value(config, 'user_embedding_mode', 'shared')
        ).lower()
        self.ui_branch_mode = str(
            _config_value(config, 'ui_branch_mode', 'dual')
        ).lower()
        self.ui_fusion_mode = str(
            _config_value(config, 'ui_fusion_mode', 'gated_sum')
        ).lower()

        if not 0.0 < self.mask_keep_ratio < 1.0:
            raise ValueError('mask_keep_ratio must be between 0 and 1.')
        if self.cl_temperature <= 0.0:
            raise ValueError('cl_temperature must be positive.')
        if not 0.0 <= self.cl_dropout < 1.0:
            raise ValueError('dropout must be in the interval [0, 1).')
        if self.cl_mode not in {
            'auto', 'branch', 'dropout', 'branch_and_dropout'
        }:
            raise ValueError(
                "cl_mode must be 'auto', 'branch', 'dropout', or "
                "'branch_and_dropout'."
            )
        if self.knn_k <= 0:
            raise ValueError('knn_k must be positive.')
        if self.mask_degree_mode not in {'full', 'masked'}:
            raise ValueError(
                "mask_degree_mode must be either 'full' or 'masked'."
            )
        if self.mask_graph_mode not in {
            'soft', 'hard', 'double_full', 'svd', 'local_prunning',
            'random_fixed', 'random_dynamic'
        }:
            raise ValueError(
                "mask_graph_mode must be 'soft', 'hard', 'double_full', "
                "'svd', 'local_prunning', 'random_fixed', or "
                "'random_dynamic'."
            )
        if self.hard_mask_temperature <= 0.0:
            raise ValueError('hard_mask_temperature must be positive.')
        if self.user_embedding_mode not in {'shared', 'separate'}:
            raise ValueError(
                "user_embedding_mode must be either 'shared' or 'separate'."
            )
        if self.ui_branch_mode not in {'dual', 'masked_only'}:
            raise ValueError(
                "ui_branch_mode must be 'dual' or 'masked_only'."
            )
        if (
            self.cl_weight > 0.0
            and self.cl_mode in {'branch', 'branch_and_dropout'}
            and self.ui_branch_mode != 'dual'
        ):
            raise ValueError(
                "Branch CL modes require ui_branch_mode 'dual'."
            )
        if self.ui_fusion_mode not in {'gated_sum', 'gated_concat'}:
            raise ValueError(
                "ui_fusion_mode must be 'gated_sum' or 'gated_concat'."
            )
        if (
            self.ui_fusion_mode == 'gated_concat'
            and self.ui_branch_mode != 'dual'
        ):
            raise ValueError(
                "ui_fusion_mode='gated_concat' requires "
                "ui_branch_mode='dual'."
            )
        if self.v_feat is None or self.t_feat is None:
            raise ValueError(
                'PGL_MASKED requires both image_feat.npy and text_feat.npy.'
            )

        self.n_nodes = self.n_users + self.n_items
        self.ui_embedding_dim = 2 * self.embedding_dim
        self.mm_embedding_dim = 2 * self.feat_embed_dim
        self.final_embedding_dim = self.ui_embedding_dim

        # Use a binary, duplicate-free interaction matrix so one learnable
        # logit always corresponds to exactly one undirected interaction.
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        interaction_matrix = interaction_matrix.tocsr()
        interaction_matrix.eliminate_zeros()
        interaction_matrix.data.fill(1.0)
        self.interaction_matrix = interaction_matrix.tocoo()
        if self.interaction_matrix.nnz == 0:
            raise ValueError('PGL_MASKED requires at least one interaction.')

        self._build_ui_graph()

        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_text.weight)
        nn.init.xavier_uniform_(self.user_image.weight)

        if (
            self.user_embedding_mode == 'separate'
            and self.ui_branch_mode == 'dual'
        ):
            self.second_user_text = nn.Embedding(
                self.n_users, self.embedding_dim
            )
            self.second_user_image = nn.Embedding(
                self.n_users, self.embedding_dim
            )
            nn.init.xavier_uniform_(self.second_user_text.weight)
            nn.init.xavier_uniform_(self.second_user_image.weight)
        else:
            self.second_user_text = None
            self.second_user_image = None

        self.image_embedding = nn.Embedding.from_pretrained(
            self.v_feat, freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            self.t_feat, freeze=False
        )
        self.image_trs = nn.Linear(
            self.v_feat.shape[1], self.feat_embed_dim
        )
        self.text_trs = nn.Linear(
            self.t_feat.shape[1], self.feat_embed_dim
        )

        if self.mm_embedding_dim == self.final_embedding_dim:
            self.item_ui_projection = nn.Identity()
            self.mm_output_projection = nn.Identity()
        else:
            self.item_ui_projection = nn.Linear(
                self.mm_embedding_dim, self.ui_embedding_dim
            )
            self.mm_output_projection = nn.Linear(
                self.mm_embedding_dim, self.final_embedding_dim
            )

        self.fusion_gate = nn.Linear(
            2 * self.ui_embedding_dim, self.ui_embedding_dim
        )
        nn.init.xavier_uniform_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        if self.ui_fusion_mode == 'gated_concat':
            self.fusion_projection = nn.Linear(
                2 * self.ui_embedding_dim, self.ui_embedding_dim
            )
            nn.init.xavier_uniform_(self.fusion_projection.weight)
            nn.init.zeros_(self.fusion_projection.bias)
        else:
            self.fusion_projection = None
        self.cl_dropout_layer = nn.Dropout(self.cl_dropout)

        self._build_or_load_mm_graph(config)
        self.latest_loss_components = {}

    def _build_ui_graph(self):
        users = torch.from_numpy(
            self.interaction_matrix.row.astype(np.int64, copy=False)
        )
        items = torch.from_numpy(
            self.interaction_matrix.col.astype(np.int64, copy=False)
        ) + self.n_users

        forward_edges = torch.stack((users, items), dim=0)
        reverse_edges = torch.stack((items, users), dim=0)
        edge_index = torch.cat((forward_edges, reverse_edges), dim=1)
        self.register_buffer('ui_edge_index', edge_index)

        self.num_interactions = self.interaction_matrix.nnz
        initial_logit = math.log(
            self.mask_keep_ratio / (1.0 - self.mask_keep_ratio)
        )
        if self.mask_graph_mode in {
            'double_full', 'svd', 'local_prunning',
            'random_fixed', 'random_dynamic'
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
        full_edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32)
        full_norm_edge_weights = self._normalized_ui_edge_weights(
            full_edge_weights
        )
        self.register_buffer(
            'full_norm_edge_weights', full_norm_edge_weights
        )
        norm_adj = self._ui_adjacency_from_weights(full_norm_edge_weights)
        self.register_buffer('norm_adj', norm_adj)

        self.register_buffer('svd_adj', None)
        self.register_buffer('local_pruned_adj', None)
        if self.mask_graph_mode == 'svd':
            self.svd_adj = self._svd_subgraph_extraction(norm_adj)
        elif self.mask_graph_mode == 'local_prunning':
            self.local_pruned_adj = self._sample_local_pruned_adjacency()

    def _svd_subgraph_extraction(self, adjacency):
        """Build the global SVD subgraph used by the original PGL."""
        adjacency = adjacency.coalesce().cpu()
        indices = adjacency.indices().numpy()
        values = adjacency.values().numpy().astype(np.float64, copy=False)
        scipy_adjacency = sp.coo_matrix(
            (values, (indices[0], indices[1])),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsc()

        rank = min(self.embedding_dim, self.n_nodes - 1)
        left_vectors, singular_values, right_vectors = svds(
            scipy_adjacency, k=rank
        )
        # ``sparsesvd``, used by the original implementation, returns values
        # from largest to smallest. Restore that ordering for SciPy's svds.
        order = np.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        left_vectors = left_vectors[:, order]
        right_vectors = right_vectors[order, :]

        num_top_bottom = max(1, int(0.25 * rank))
        top_values = singular_values[:num_top_bottom]
        bottom_values = singular_values[-num_top_bottom:]
        product_values = top_values * bottom_values
        product_matrix = (
            left_vectors[:, :num_top_bottom]
            @ np.diag(product_values)
            @ right_vectors[:num_top_bottom, :]
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
        ).to(self.ui_edge_index.device)
        return torch.sparse_coo_tensor(
            svd_indices,
            svd_values,
            (self.n_nodes, self.n_nodes),
            device=self.ui_edge_index.device,
        ).coalesce()

    @property
    def local_keep_count(self):
        # The original PGL keeps 30% of edges. mask_keep_ratio defaults to
        # 0.3 and makes the same local-pruning method configurable here.
        return max(
            1,
            min(
                self.num_interactions,
                int(self.num_interactions * self.mask_keep_ratio),
            ),
        )

    @torch.no_grad()
    def _sample_local_pruned_adjacency(self):
        """Degree-weighted local edge pruning from the original PGL."""
        sampling_weights = self.full_norm_edge_weights[
            :self.num_interactions
        ]
        kept_interactions = torch.multinomial(
            sampling_weights,
            self.local_keep_count,
            replacement=False,
        )
        reverse_interactions = kept_interactions + self.num_interactions
        kept_undirected = torch.cat(
            (kept_interactions, reverse_interactions), dim=0
        )
        edge_index = self.ui_edge_index[:, kept_undirected]
        edge_weights = torch.ones(
            edge_index.size(1),
            dtype=self.full_norm_edge_weights.dtype,
            device=edge_index.device,
        )
        return self._normalized_ui_adjacency(edge_weights, edge_index)

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
        degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
        degree_inv_sqrt = torch.where(
            degree > 0,
            degree_inv_sqrt,
            torch.zeros_like(degree_inv_sqrt),
        )
        normalized_weights = (
            degree_inv_sqrt[row] * edge_weights * degree_inv_sqrt[col]
        )
        return normalized_weights

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
        normalized_weights = self._normalized_ui_edge_weights(
            edge_weights, edge_index
        )
        return self._ui_adjacency_from_weights(
            normalized_weights, edge_index
        )

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
        # Match hard top-k exactly so random and learnable masks are comparable.
        return self.hard_keep_count

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
    def _sample_hard_train_indices(self, mask_logits):
        uniform_noise = torch.rand_like(mask_logits).clamp_(
            1e-8, 1.0 - 1e-8
        )
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        selection_scores = (
            mask_logits / self.hard_mask_temperature + gumbel_noise
        )
        return torch.topk(
            selection_scores,
            self.hard_keep_count,
            sorted=False,
        ).indices

    @torch.no_grad()
    def _select_hard_eval_indices(self, mask_logits):
        return torch.topk(
            mask_logits,
            self.hard_keep_count,
            sorted=False,
        ).indices

    def pre_epoch_processing(self):
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

    def _hard_masked_ui_adjacency(self, interaction_mask):
        kept_interactions = self._current_hard_indices()
        reverse_interactions = kept_interactions + self.num_interactions
        kept_undirected = torch.cat(
            (kept_interactions, reverse_interactions), dim=0
        )
        hard_edge_index = self.ui_edge_index[:, kept_undirected]

        selected_soft_mask = interaction_mask[kept_interactions]
        selected_hard_mask = (
            torch.ones_like(selected_soft_mask)
            + selected_soft_mask
            - selected_soft_mask.detach()
        )
        hard_undirected_mask = torch.cat(
            (selected_hard_mask, selected_hard_mask), dim=0
        )

        if self.mask_degree_mode == 'full':
            masked_edge_weights = (
                self.full_norm_edge_weights[kept_undirected]
                * hard_undirected_mask
            )
            return self._ui_adjacency_from_weights(
                masked_edge_weights, hard_edge_index
            )

        return self._normalized_ui_adjacency(
            hard_undirected_mask, hard_edge_index
        )

    def _random_masked_ui_adjacency(self):
        kept_interactions = (
            self.random_train_indices
            if self.training
            else self.random_eval_indices
        )
        reverse_interactions = kept_interactions + self.num_interactions
        kept_undirected = torch.cat(
            (kept_interactions, reverse_interactions), dim=0
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

        if self.mask_graph_mode in {'svd', 'local_prunning'}:
            if self.mask_graph_mode == 'svd':
                return self.svd_adj, None
            if not self.training:
                return self.norm_adj, None
            if self.local_pruned_adj is None:
                self.local_pruned_adj = (
                    self._sample_local_pruned_adjacency()
                )
            return self.local_pruned_adj, None

        interaction_mask = torch.sigmoid(self.mask_logits)
        if self.mask_graph_mode == 'hard':
            masked_adj = self._hard_masked_ui_adjacency(interaction_mask)
            return masked_adj, interaction_mask

        undirected_mask = torch.cat(
            (interaction_mask, interaction_mask), dim=0
        )
        if self.mask_degree_mode == 'full':
            masked_edge_weights = (
                self.full_norm_edge_weights * undirected_mask
            )
            masked_adj = self._ui_adjacency_from_weights(
                masked_edge_weights
            )
        else:
            masked_adj = self._normalized_ui_adjacency(undirected_mask)
        return masked_adj, interaction_mask

    def _build_or_load_mm_graph(self, config):
        dataset_path = os.path.abspath(
            os.path.join(config['data_path'], config['dataset'])
        )
        cache_name = 'mm_adj_freedomdsp_{}_{}.pt'.format(
            self.knn_k, int(10 * self.mm_image_weight)
        )
        mm_adj_file = os.path.join(dataset_path, cache_name)

        if os.path.exists(mm_adj_file):
            mm_adj = torch.load(mm_adj_file, map_location=self.device)
            if tuple(mm_adj.shape) != (self.n_items, self.n_items):
                raise ValueError(
                    'Cached multimodal graph has shape {}, expected {}.'.format(
                        tuple(mm_adj.shape), (self.n_items, self.n_items)
                    )
                )
        else:
            with torch.no_grad():
                image_adj = self.get_knn_adj_mat(
                    self.image_embedding.weight.detach()
                )
                text_adj = self.get_knn_adj_mat(
                    self.text_embedding.weight.detach()
                )
                mm_adj = (
                    self.mm_image_weight * image_adj
                    + (1.0 - self.mm_image_weight) * text_adj
                ).coalesce()
            torch.save(mm_adj.cpu(), mm_adj_file)
            mm_adj = mm_adj.to(self.device)

        self.register_buffer('mm_adj', mm_adj.coalesce())

    def get_knn_adj_mat(self, mm_embeddings):
        if self.n_items == 0:
            raise ValueError('Cannot build an item graph without items.')

        topk = min(self.knn_k, self.n_items)
        context_norm = F.normalize(mm_embeddings, p=2, dim=-1, eps=1e-12)
        similarity = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_indices = torch.topk(similarity, topk, dim=-1)
        del similarity

        rows = torch.arange(
            knn_indices.size(0), device=mm_embeddings.device
        ).unsqueeze(1).expand(-1, topk)
        indices = torch.stack(
            (rows.reshape(-1), knn_indices.reshape(-1)), dim=0
        )
        return self._normalized_item_adjacency(
            indices, torch.Size((self.n_items, self.n_items))
        )

    @staticmethod
    def _normalized_item_adjacency(indices, size):
        values = torch.ones(
            indices.size(1), dtype=torch.float32, device=indices.device
        )
        adjacency = torch.sparse_coo_tensor(
            indices, values, size, device=indices.device
        ).coalesce()
        row_sum = torch.sparse.sum(adjacency, dim=1).to_dense()
        degree_inv_sqrt = row_sum.clamp_min(1e-12).pow(-0.5)
        row, col = adjacency.indices()
        normalized_values = (
            degree_inv_sqrt[row]
            * adjacency.values()
            * degree_inv_sqrt[col]
        )
        return torch.sparse_coo_tensor(
            adjacency.indices(),
            normalized_values,
            size,
            device=indices.device,
        ).coalesce()

    def _initial_node_embeddings(self):
        image_features = F.normalize(
            self.image_trs(self.image_embedding.weight), dim=-1
        )
        text_features = F.normalize(
            self.text_trs(self.text_embedding.weight), dim=-1
        )
        multimodal_items = torch.cat(
            (image_features, text_features), dim=1
        )

        user_embeddings = torch.cat(
            (self.user_image.weight, self.user_text.weight), dim=1
        )
        if self.second_user_image is not None:
            second_user_embeddings = torch.cat(
                (
                    self.second_user_image.weight,
                    self.second_user_text.weight,
                ),
                dim=1,
            )
        else:
            second_user_embeddings = user_embeddings

        ui_item_embeddings = self.item_ui_projection(multimodal_items)
        full_initial_embeddings = torch.cat(
            (user_embeddings, ui_item_embeddings), dim=0
        )
        second_initial_embeddings = torch.cat(
            (second_user_embeddings, ui_item_embeddings), dim=0
        )
        return (
            full_initial_embeddings,
            second_initial_embeddings,
            multimodal_items,
        )

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
        """Propagate without dense gradients for learnable edge weights."""
        adjacency = adjacency.coalesce()
        differentiable_adjacency = adjacency.requires_grad

        embeddings = [initial_embeddings]
        current_embeddings = initial_embeddings
        for _ in range(self.n_ui_layers):
            if differentiable_adjacency:
                current_embeddings = self._memory_safe_sparse_mm(
                    adjacency, current_embeddings
                )
            else:
                current_embeddings = torch.sparse.mm(
                    adjacency, current_embeddings
                )
            embeddings.append(current_embeddings)
        return torch.stack(embeddings, dim=1).mean(dim=1)

    def _propagate_mm_graph(self, item_embeddings):
        propagated_items = item_embeddings
        for _ in range(self.n_mm_layers):
            propagated_items = torch.sparse.mm(
                self.mm_adj, propagated_items
            )
        return self.mm_output_projection(propagated_items)

    def _fuse_ui_branches(self, full_embeddings, masked_embeddings):
        branch_embeddings = torch.cat(
            (full_embeddings, masked_embeddings), dim=-1
        )
        gate = torch.sigmoid(self.fusion_gate(branch_embeddings))
        gated_full = gate * full_embeddings
        gated_masked = (1.0 - gate) * masked_embeddings
        if self.ui_fusion_mode == 'gated_concat':
            # gated_branches = torch.cat(
            #     (gated_full, gated_masked), dim=-1
            # )
            gated_branches = torch.cat(
                            (full_embeddings, masked_embeddings), dim=-1
                        )
            fused_embeddings = self.fusion_projection(gated_branches)
        else:
            fused_embeddings = gated_full + gated_masked
        return fused_embeddings, gate

    def _encode(self):
        (
            full_initial_embeddings,
            second_initial_embeddings,
            multimodal_items,
        ) = self._initial_node_embeddings()

        masked_adj, interaction_mask = self._masked_ui_adjacency()
        masked_embeddings = self._propagate_ui_graph(
            masked_adj, second_initial_embeddings
        )

        if self.ui_branch_mode == 'masked_only':
            masked_users, masked_items = torch.split(
                masked_embeddings, [self.n_users, self.n_items], dim=0
            )
            mm_items = self._propagate_mm_graph(multimodal_items)
            return {
                'users': masked_users,
                'items': masked_items + mm_items,
                'full_users': None,
                'full_items': None,
                'masked_users': masked_users,
                'masked_items': masked_items,
                'mask': interaction_mask,
            }

        full_embeddings = self._propagate_ui_graph(
            self.norm_adj, full_initial_embeddings
        )

        fused_embeddings, _ = self._fuse_ui_branches(
            full_embeddings, masked_embeddings
        )

        full_users, full_items = torch.split(
            full_embeddings, [self.n_users, self.n_items], dim=0
        )
        masked_users, masked_items = torch.split(
            masked_embeddings, [self.n_users, self.n_items], dim=0
        )
        fused_users, fused_items = torch.split(
            fused_embeddings, [self.n_users, self.n_items], dim=0
        )

        mm_items = self._propagate_mm_graph(multimodal_items)
        final_items = fused_items + mm_items

        return {
            'users': fused_users,
            'items': final_items,
            'full_users': full_users,
            'full_items': full_items,
            'masked_users': masked_users,
            'masked_items': masked_items,
            'mask': interaction_mask,
        }

    def forward(self):
        representations = self._encode()
        return representations['users'], representations['items']

    @staticmethod
    def bpr_loss(users, positive_items, negative_items):
        positive_scores = torch.sum(users * positive_items, dim=1)
        negative_scores = torch.sum(users * negative_items, dim=1)
        return -F.logsigmoid(positive_scores - negative_scores).mean()

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

    def pgl_info_nce(self, first_view, second_view):
        """One-direction InfoNCE used by the original PGL implementation."""
        first_view = F.normalize(first_view, dim=1)
        second_view = F.normalize(second_view, dim=1)
        logits = torch.matmul(first_view, second_view.transpose(0, 1))
        logits = logits / self.cl_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _dropout_contrastive_loss(
        self, user_embeddings, positive_embeddings
    ):
        """Original PGL CL on two dropout views of fused representations."""
        user_loss = self.pgl_info_nce(
            self.cl_dropout_layer(user_embeddings),
            self.cl_dropout_layer(user_embeddings),
        )
        item_loss = self.pgl_info_nce(
            self.cl_dropout_layer(positive_embeddings),
            self.cl_dropout_layer(positive_embeddings),
        )
        return 0.5 * (user_loss + item_loss)

    def _branch_contrastive_loss(
        self, representations, users, positive_items
    ):
        """Symmetric CL between the full and masked pre-fusion branches."""
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

    def _contrastive_losses(
        self,
        representations,
        users,
        positive_items,
        user_embeddings,
        positive_embeddings,
        reference_loss,
    ):
        zero = reference_loss.new_zeros(())
        if self.cl_weight == 0.0:
            return zero, zero, zero

        effective_mode = self.cl_mode
        if effective_mode == 'auto':
            effective_mode = (
                'dropout'
                if representations['full_users'] is None
                else 'branch'
            )

        branch_loss = zero
        dropout_loss = zero
        active_losses = []
        if effective_mode in {'branch', 'branch_and_dropout'}:
            branch_loss = self._branch_contrastive_loss(
                representations, users, positive_items
            )
            active_losses.append(branch_loss)
        if effective_mode in {'dropout', 'branch_and_dropout'}:
            dropout_loss = self._dropout_contrastive_loss(
                user_embeddings, positive_embeddings
            )
            active_losses.append(dropout_loss)

        contrastive_loss = torch.stack(active_losses).mean()
        return contrastive_loss, branch_loss, dropout_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        positive_items = interaction[1]
        negative_items = interaction[2]
        representations = self._encode()

        user_embeddings = representations['users'][users]
        positive_embeddings = representations['items'][positive_items]
        negative_embeddings = representations['items'][negative_items]
        ranking_loss = self.bpr_loss(
            user_embeddings, positive_embeddings, negative_embeddings
        )

        (
            contrastive_loss,
            branch_contrastive_loss,
            dropout_contrastive_loss,
        ) = self._contrastive_losses(
            representations,
            users,
            positive_items,
            user_embeddings,
            positive_embeddings,
            ranking_loss,
        )

        interaction_mask = representations['mask']
        if interaction_mask is None:
            mask_loss = ranking_loss.new_zeros(())
            mask_mean = ranking_loss.new_ones(())
        else:
            mask_mean = interaction_mask.mean()
            budget_loss = (
                mask_mean - self.mask_keep_ratio
            ).pow(2)
            binary_loss = (
                interaction_mask * (1.0 - interaction_mask)
            ).mean()
            mask_loss = (
                budget_loss + self.mask_binary_weight * binary_loss
            )

        total_loss = (
            ranking_loss
            + self.cl_weight * contrastive_loss
            + self.mask_weight * mask_loss
        )
        self.latest_loss_components = {
            'bpr': ranking_loss.detach(),
            'contrastive': contrastive_loss.detach(),
            'branch_contrastive': branch_contrastive_loss.detach(),
            'dropout_contrastive': dropout_contrastive_loss.detach(),
            'mask': mask_loss.detach(),
            'mask_mean': mask_mean.detach(),
        }
        return total_loss

    def full_sort_predict(self, interaction):
        user_embeddings, item_embeddings = self.forward()
        batch_user_embeddings = user_embeddings[interaction[0]]
        return torch.matmul(
            batch_user_embeddings, item_embeddings.transpose(0, 1)
        )

    @torch.no_grad()
    def get_analysis_artifacts(self):
        """Export masks and learned embeddings aligned with U-I edge IDs."""
        was_training = self.training
        self.eval()
        representations = self._encode()

        forward_edges = self.ui_edge_index[:, :self.num_interactions]
        edge_users = forward_edges[0].detach().cpu()
        edge_items = (
            forward_edges[1] - self.n_users
        ).detach().cpu()

        masks = {}
        mask_entries = []
        if self.mask_logits is not None:
            mask_entries.append(('masked_branch', self.mask_logits))

        for branch_name, logits in mask_entries:
            probabilities = torch.sigmoid(logits)
            topk_indices = self._select_hard_eval_indices(logits)
            topk_selected = torch.zeros_like(
                probabilities, dtype=torch.bool
            )
            topk_selected[topk_indices] = True
            masks[branch_name] = {
                'logits': logits.detach().cpu(),
                'probabilities': probabilities.detach().cpu(),
                'selected_at_keep_ratio': topk_selected.detach().cpu(),
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

        embedding_tables = {}
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Embedding) and module.weight.requires_grad:
                embedding_tables[module_name + '.weight'] = (
                    module.weight.detach().cpu()
                )

        exported_representations = {
            name: tensor.detach().cpu()
            for name, tensor in representations.items()
            if torch.is_tensor(tensor) and name != 'mask'
        }

        artifacts = {
            'metadata': {
                'model': self.__class__.__name__,
                'mask_graph_mode': self.mask_graph_mode,
                'mask_degree_mode': self.mask_degree_mode,
                'ui_branch_mode': self.ui_branch_mode,
                'ui_fusion_mode': self.ui_fusion_mode,
                'user_embedding_mode': self.user_embedding_mode,
                'cl_mode': self.cl_mode,
                'cl_weight': self.cl_weight,
                'cl_temperature': self.cl_temperature,
                'cl_dropout': self.cl_dropout,
                'mask_keep_ratio': self.mask_keep_ratio,
                'random_mask_seed': self.random_mask_seed,
                'num_users': self.n_users,
                'num_items': self.n_items,
                'num_interactions': self.num_interactions,
            },
            'ui_edges': {
                'user_ids': edge_users,
                'item_ids': edge_items,
            },
            'masks': masks,
            'embedding_tables': embedding_tables,
            'representations': exported_representations,
        }

        if was_training:
            self.train()
        return artifacts
