"""FREEDOM with an independently masked user-item graph branch.

The FREEDOM branch keeps its original degree-sensitive epoch dropout.  The
second branch is controlled by a learned/static mask and never receives that
dropout.  There is deliberately no contrastive-learning objective.
"""

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse.linalg import svds

from models.freedom import FREEDOM, _config_value


class _ObservedEdgeSparseMM(torch.autograd.Function):
    """Sparse multiplication whose adjacency gradient is edge-only."""

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
    """Dual-branch FREEDOM with configurable user-item graph masking."""

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
        self.user_embedding_mode = str(
            _config_value(config, 'user_embedding_mode', 'separate')
        ).lower()
        self.item_embedding_mode = str(
            _config_value(config, 'item_embedding_mode', 'separate')
        ).lower()
        self.ui_branch_mode = str(
            _config_value(config, 'ui_branch_mode', 'dual')
        ).lower()
        self.ui_fusion_mode = str(
            _config_value(config, 'ui_fusion_mode', 'gated_concat')
        ).lower()
        self._validate_mask_config()

        if self.user_embedding_mode == 'separate':
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

        if self.ui_branch_mode == 'dual':
            self.fusion_gate = nn.Linear(
                2 * self.embedding_dim, self.embedding_dim
            )
            nn.init.xavier_uniform_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)
        else:
            self.fusion_gate = None

        if (
            self.ui_branch_mode == 'dual'
            and self.ui_fusion_mode == 'gated_concat'
        ):
            self.final_embedding_dim = 2 * self.embedding_dim
        else:
            self.final_embedding_dim = self.embedding_dim

        # Auxiliary feature BPR must use the same dimension as final scores.
        if self.image_embedding is not None:
            self.image_aux_projection = self._make_aux_projection(
                self.final_embedding_dim
            )
        if self.text_embedding is not None:
            self.text_aux_projection = self._make_aux_projection(
                self.final_embedding_dim
            )

        self._initialize_mask_graph()
        self.latest_loss_components = {}

    def _validate_mask_config(self):
        if not 0.0 < self.mask_keep_ratio < 1.0:
            raise ValueError('mask_keep_ratio must be between 0 and 1.')
        if self.mask_weight < 0.0 or self.mask_binary_weight < 0.0:
            raise ValueError('Mask loss weights cannot be negative.')
        if self.mask_degree_mode not in {'full', 'masked'}:
            raise ValueError(
                "mask_degree_mode must be either 'full' or 'masked'."
            )
        if self.mask_graph_mode not in {
            'soft', 'hard', 'double_full', 'svd', 'local_prunning'
        }:
            raise ValueError(
                "mask_graph_mode must be 'soft', 'hard', 'double_full', "
                "'svd', or 'local_prunning'."
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
        if self.ui_branch_mode not in {'dual', 'masked_only'}:
            raise ValueError(
                "ui_branch_mode must be 'dual' or 'masked_only'."
            )
        if self.ui_fusion_mode not in {'gated_sum', 'gated_concat'}:
            raise ValueError(
                "ui_fusion_mode must be 'gated_sum' or 'gated_concat'."
            )
        if (
            self.ui_branch_mode == 'masked_only'
            and self.ui_fusion_mode == 'gated_concat'
        ):
            raise ValueError(
                "ui_fusion_mode='gated_concat' requires ui_branch_mode='dual'."
            )

    def _initialize_mask_graph(self):
        initial_logit = math.log(
            self.mask_keep_ratio / (1.0 - self.mask_keep_ratio)
        )
        if self.mask_graph_mode in {
            'double_full', 'svd', 'local_prunning'
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
        self.register_buffer('svd_adj', None)
        self.register_buffer('local_pruned_adj', None)
        if self.mask_graph_mode == 'svd':
            self.svd_adj = self._svd_subgraph_extraction(self.norm_adj)
        elif self.mask_graph_mode == 'local_prunning':
            self.local_pruned_adj = self._sample_local_pruned_adjacency()

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
        return self._normalized_ui_adjacency(
            edge_weights, edge_index
        )

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
        # These two samplers intentionally write disjoint graph state.
        self._resample_freedom_adjacency()
        if self.mask_graph_mode == 'hard':
            self.hard_train_indices = self._sample_hard_train_indices(
                self.mask_logits
            )
        elif self.mask_graph_mode == 'local_prunning':
            self.local_pruned_adj = self._sample_local_pruned_adjacency()

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

    def _masked_ui_adjacency(self):
        if self.mask_graph_mode == 'double_full':
            return self.norm_adj, None
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

    def _masked_initial_embeddings(self):
        users = (
            self.user_embedding.weight
            if self.masked_user_embedding is None
            else self.masked_user_embedding.weight
        )
        items = (
            self.item_id_embedding.weight
            if self.masked_item_id_embedding is None
            else self.masked_item_id_embedding.weight
        )
        return torch.cat((users, items), dim=0)

    def _fuse_ui_branches(self, full_embeddings, masked_embeddings):
        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat((full_embeddings, masked_embeddings), dim=-1)
            )
        )
        gated_full = gate * full_embeddings
        gated_masked = (1.0 - gate) * masked_embeddings
        if self.ui_fusion_mode == 'gated_concat':
            return torch.cat((gated_full, gated_masked), dim=-1), gate
        return gated_full + gated_masked, gate

    def _encode(self):
        masked_adjacency, probabilities = self._masked_ui_adjacency()
        masked_embeddings = self._propagate_ui_graph(
            masked_adjacency, self._masked_initial_embeddings()
        )
        masked_users, masked_items = torch.split(
            masked_embeddings, (self.n_users, self.n_items), dim=0
        )

        if self.ui_branch_mode == 'masked_only':
            final_items = self._propagate_mm_graph(masked_items)
            return {
                'users': masked_users,
                'items': final_items,
                'fused_items': masked_items,
                'full_users': None,
                'full_items': None,
                'masked_users': masked_users,
                'masked_items': masked_items,
                'mask': probabilities,
            }

        full_initial = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight),
            dim=0,
        )
        full_embeddings = self._propagate_ui_graph(
            self._freedom_ui_adjacency(), full_initial
        )
        full_users, full_items = torch.split(
            full_embeddings, (self.n_users, self.n_items), dim=0
        )

        fused_embeddings, _ = self._fuse_ui_branches(
            full_embeddings, masked_embeddings
        )
        fused_users, fused_items = torch.split(
            fused_embeddings, (self.n_users, self.n_items), dim=0
        )
        final_items = self._propagate_mm_graph(fused_items)
        return {
            'users': fused_users,
            'items': final_items,
            'fused_items': fused_items,
            'full_users': full_users,
            'full_items': full_items,
            'masked_users': masked_users,
            'masked_items': masked_items,
            'mask': probabilities,
        }

    def forward(self):
        representations = self._encode()
        return representations['users'], representations['items']

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
        (
            mask_regularization,
            budget_loss,
            binary_loss,
            mask_mean,
        ) = self._mask_regularization(
            representations['mask'], ranking_loss
        )
        total_loss = (
            ranking_loss
            + self.reg_weight * (visual_loss + text_loss)
            + self.mask_weight * mask_regularization
        )
        self.latest_loss_components = {
            'bpr': ranking_loss.detach(),
            'visual_bpr': visual_loss.detach(),
            'text_bpr': text_loss.detach(),
            'mask': mask_regularization.detach(),
            'mask_budget': budget_loss.detach(),
            'mask_binary': binary_loss.detach(),
            'mask_mean': mask_mean.detach(),
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
                'mask_graph_mode': self.mask_graph_mode,
                'mask_degree_mode': self.mask_degree_mode,
                'ui_branch_mode': self.ui_branch_mode,
                'ui_fusion_mode': self.ui_fusion_mode,
                'user_embedding_mode': self.user_embedding_mode,
                'item_embedding_mode': self.item_embedding_mode,
                'mask_keep_ratio': self.mask_keep_ratio,
                'dropout': self.dropout,
                'embedding_dim': self.embedding_dim,
                'final_embedding_dim': self.final_embedding_dim,
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
