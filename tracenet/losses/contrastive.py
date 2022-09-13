"""
adapted from https://github.com/kreshuklab/spoco
"""

import math

import torch
from torch import nn as nn


class ContrastiveLoss(nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'
    """

    def __init__(self, delta_var, delta_dist, kernel_threshold=None, norm='fro', alpha=1., beta=1., gamma=0.001,
                 instance_term_weight=1., unlabeled_push_weight=1., ignore_label=None, bg_push=False,
                 hinge_pull=True, instance_loss=None, aux_loss_ignore_zero=True):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.instance_term_weight = instance_term_weight
        self.unlabeled_push_weight = unlabeled_push_weight
        self.ignore_label = ignore_label
        self.bg_push = bg_push
        self.hinge_pull = hinge_pull
        self.instance_loss = instance_loss

        self.aux_loss_ignore_zero = aux_loss_ignore_zero
        self.dist_to_mask = Gaussian(delta_var=delta_var,
                                     pmaps_threshold=kernel_threshold if kernel_threshold is not None
                                     else delta_var)
        self.clustered_masks = []
        self.gt_masks = []
        self.clear_masks()

    def clear_masks(self):
        self.clustered_masks = []
        self.gt_masks = []

    def _compute_variance_term(self, cluster_means, embeddings, target, instance_counts, ignore_zero_label):
        """
        Computes the variance term, i.e. intra-cluster pull force that draws embeddings towards the mean embedding

        C - number of clusters (instances)
        E - embedding dimension
        SPATIAL - volume shape, i.e. DxHxW for 3D/ HxW for 2D

        Args:
            cluster_means: mean embedding of each instance, tensor (CxE)
            embeddings: embeddings vectors per instance, tensor (ExSPATIAL)
            target: label tensor (1xSPATIAL); each label is represented as one-hot vector
            instance_counts: number of voxels per instance
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """

        assert target.dim() in (2, 3)
        n_instances = cluster_means.shape[0]

        # compute the spatial mean and instance fields by scattering with the
        # target tensor
        cluster_means_spatial = cluster_means[target]
        instance_sizes_spatial = instance_counts[target]

        # permute the embedding dimension to axis 0
        if target.dim() == 2:
            cluster_means_spatial = cluster_means_spatial.permute(2, 0, 1)
        else:
            cluster_means_spatial = cluster_means_spatial.permute(3, 0, 1, 2)

        # compute the distance to cluster means
        dist_to_mean = torch.norm(embeddings - cluster_means_spatial, self.norm, dim=0)

        if ignore_zero_label:
            # zero out distances corresponding to 0-label cluster, so that it does not contribute to the loss
            dist_mask = torch.ones_like(dist_to_mean)
            dist_mask[target == 0] = 0
            dist_to_mean = dist_to_mean * dist_mask
            # decrease number of instances
            n_instances -= 1
            # if there is only 0-label in the target return 0
            if n_instances == 0:
                return 0.

        if self.hinge_pull:
            # zero out distances less than delta_var (hinge)
            dist_to_mean = torch.clamp(dist_to_mean - self.delta_var, min=0)

        dist_to_mean = dist_to_mean ** 2
        # normalize the variance by instance sizes and number of instances and sum it up
        variance_term = torch.sum(dist_to_mean / instance_sizes_spatial) / n_instances
        return variance_term

    def _compute_background_push(self, cluster_means, embeddings, target):
        assert target.dim() in (2, 3)
        n_instances = cluster_means.shape[0]

        # permute embedding dimension at the end
        if target.dim() == 2:
            embeddings = embeddings.permute(1, 2, 0)
        else:
            embeddings = embeddings.permute(1, 2, 3, 0)

        # decrease number of instances `C` since we're ignoring 0-label
        n_instances -= 1
        # if there is only 0-label in the target return 0
        if n_instances == 0:
            return 0.

        background_mask = target == 0
        n_background = background_mask.sum()
        background_push = 0.
        # skip embedding corresponding to the background pixels
        for cluster_mean in cluster_means[1:]:
            # compute distances between embeddings and a given cluster_mean
            dist_to_mean = torch.norm(embeddings - cluster_mean, self.norm, dim=-1)
            # apply background mask and compute hinge
            dist_hinged = torch.clamp((self.delta_dist - dist_to_mean) * background_mask, min=0) ** 2
            background_push += torch.sum(dist_hinged) / n_background

        # normalize by the number of instances
        return background_push / n_instances

    def _compute_distance_term(self, cluster_means, ignore_zero_label):
        """
        Compute the distance term, i.e an inter-cluster push-force that pushes clusters away from each other,
        increasing the distance between cluster centers

        Args:
            cluster_means: mean embedding of each instance, tensor (CxE)
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """
        C = cluster_means.size(0)
        if C == 1:
            # just one cluster in the batch, so distance term does not contribute to the loss
            return 0.

        # expand cluster_means tensor in order to compute the pair-wise distance between cluster means
        # CxE -> CxCxE
        cluster_means = cluster_means.unsqueeze(0)
        shape = list(cluster_means.size())
        shape[0] = C

        # cm_matrix1 is CxCxE
        cm_matrix1 = cluster_means.expand(shape)
        # transpose the cluster_means matrix in order to compute pair-wise distances
        cm_matrix2 = cm_matrix1.permute(1, 0, 2)
        # compute pair-wise distances between cluster means, result is a CxC tensor
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=2)

        # create matrix for the repulsion distance (i.e. cluster centers further apart than 2 * delta_dist
        # are not longer repulsed)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        repulsion_dist = repulsion_dist.to(cluster_means.device)

        if ignore_zero_label:
            if C == 2:
                # just two cluster instances, including one which is ignored,
                # i.e. distance term does not contribute to the loss
                return 0.
            # set the distance to 0-label to be greater than 2*delta_dist, so that it does not contribute to the loss
            # because of the hinge at 2*delta_dist

            # find minimum dist
            d_min = torch.min(dist_matrix[dist_matrix > 0]).item()
            # dist_multiplier = 2 * delta_dist / d_min + unlabeled_push_weight
            dist_multiplier = 2 * self.delta_dist / d_min + 1e-3
            # create distance mask
            dist_mask = torch.ones_like(dist_matrix)
            dist_mask[0, 1:] = dist_multiplier
            dist_mask[1:, 0] = dist_multiplier

            # mask the dist_matrix
            dist_matrix = dist_matrix * dist_mask
            # decrease number of instances
            C -= 1

        # zero out distances grater than 2*delta_dist (hinge)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2
        # sum all of the hinged pair-wise distances
        dist_sum = torch.sum(hinged_dist)
        # normalized by the number of paris and return
        distance_term = dist_sum / (C * (C - 1))
        return distance_term

    def _compute_regularizer_term(self, cluster_means):
        """
        Computes the regularizer term, i.e. a small pull-force that draws all clusters towards origin to keep
        the network activations bounded
        """
        # compute the norm of the mean embeddings
        norms = torch.norm(cluster_means, p=self.norm, dim=1)
        # return the average norm per batch
        return torch.sum(norms) / cluster_means.size(0)

    def _should_ignore(self, target):
        # set default values
        ignore_zero_label = False
        single_target = target

        if self.ignore_label is not None:
            assert target.dim() == 4, "Expects target to be 2xDxHxW when ignore_label is set"
            # get relabeled target
            single_target = target[0]
            # get original target and ignore 0-label only if 0-label was present in the original target
            original = target[1]
            ignore_zero_label = self.ignore_label in original

        return ignore_zero_label, single_target

    def create_instance_pmaps_and_masks(self, embeddings, anchors, target):
        """
        Given the feature space and the anchor embeddings returns the 'soft' masks (one for every anchor)
        together with ground truth binary masks extracted from the target.

        Both: 'soft' masks and ground truth masks are stacked along a new channel dimension.

        Args:
            embeddings (torch.Tensor): ExSpatial image embeddings (E - emb dim)
            anchors (torch.Tensor): CxE anchor points in the embedding space (C - number of anchors)
            target (torch.Tensor): (partial) ground truth segmentation

        Returns:
            (soft_masks, gt_masks): tuple of two tensors of shape CxSpatial
        """
        inst_pmaps = []
        inst_masks = []

        # permute embedding dimension
        if target.dim() == 2:
            embeddings = embeddings.permute(1, 2, 0)
        else:
            embeddings = embeddings.permute(1, 2, 3, 0)

        for i, anchor_emb in enumerate(anchors):
            if i == 0 and self.aux_loss_ignore_zero:
                # ignore 0-label
                continue
            # compute distance map
            distance_map = torch.linalg.norm(embeddings - anchor_emb, self.norm, dim=-1)
            # convert distance map to instance pmaps and save
            inst_pmaps.append(self.dist_to_mask(distance_map).unsqueeze(0))
            # create real mask and save
            assert i in target
            inst_masks.append((target == i).float().unsqueeze(0))

        if not inst_masks:
            # no masks have been extracted from the image
            return None, None

        # stack along batch dimension
        inst_pmaps = torch.stack(inst_pmaps)
        inst_masks = torch.stack(inst_masks)

        return inst_pmaps, inst_masks

    def instance_based_loss(self, embeddings, cluster_means, target):
        """
        Computes auxiliary loss based on embeddings and a given list of target instances together with
        their mean embeddings

        Args:
            embeddings (torch.tensor): pixel embeddings (ExSPATIAL)
            cluster_means (torch.tensor): mean embeddings per instance (CxExSINGLETON_SPATIAL)
            target (torch.tensor): ground truth instance segmentation (SPATIAL)
        """
        if self.instance_loss is None:
            return 0.
        else:
            assert embeddings.size()[1:] == target.size()
            # extract soft and ground truth masks from the feature space
            instance_pmaps, instance_masks = self.create_instance_pmaps_and_masks(embeddings,
                                                                                  cluster_means, target)
            self.clustered_masks.append(instance_pmaps.detach())
            self.gt_masks.append(instance_masks.detach())
            # compute instance-based loss
            if instance_masks is None:
                return 0.
            return self.instance_loss(instance_pmaps, instance_masks).mean()

    def forward(self, input_, target):
        """
        Args:
             input_ (torch.tensor): embeddings predicted by the network (NxExDxHxW) (E - embedding dims)
                                    expects float32 tensor
             target (torch.tensor): ground truth instance segmentation (NxDxHxW)
                                    expects int64 tensor
                                    if self.ignore_zero_label is True then expects target of shape Nx2xDxHxW where
                                    relabeled version is in target[:,0,...] and the original labeling is in
                                    target[:,1,...]

        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
        """

        n_batches = input_.shape[0]
        # compute the loss per each instance in the batch separately
        # and sum it up in the per_instance variable
        per_instance_loss = 0.
        for single_input, single_target in zip(input_, target):
            # check if the target contain ignore_label; ignore_label is going to be mapped to the 0-label
            # so we just need to ignore 0-label in the pull and push forces
            ignore_zero_label, single_target = self._should_ignore(single_target)
            contains_bg = 0 in single_target
            if self.bg_push and contains_bg:
                ignore_zero_label = True

            instance_ids, instance_counts = torch.unique(single_target, return_counts=True)

            # compare spatial dimensions
            assert single_input.size()[1:] == single_target.size()

            # compute mean embeddings (output is of shape CxE)
            cluster_means = compute_cluster_means(single_input, single_target, instance_ids.size(0))

            # compute variance term, i.e. pull force
            variance_term = self._compute_variance_term(cluster_means, single_input, single_target, instance_counts,
                                                        ignore_zero_label)

            # compute background push force, i.e. push force between the mean cluster embeddings and embeddings of
            # background pixels
            # compute only ignore_zero_label is True, i.e. a given patch contains background label
            unlabeled_push = 0.
            if self.bg_push and contains_bg:
                unlabeled_push = self._compute_background_push(cluster_means, single_input, single_target)

            # compute the instance-based loss
            instance_loss = self.instance_based_loss(single_input, cluster_means, single_target)

            # compute distance term, i.e. push force
            distance_term = self._compute_distance_term(cluster_means, ignore_zero_label)

            # compute regularization term
            regularization_term = self._compute_regularizer_term(cluster_means)

            # compute total loss and sum it up
            loss = self.alpha * variance_term + \
                   self.beta * distance_term + \
                   self.gamma * regularization_term + \
                   self.instance_term_weight * instance_loss + \
                   self.unlabeled_push_weight * unlabeled_push

            per_instance_loss += loss

        # reduce across the batch dimension
        return per_instance_loss.div(n_batches)


# kernel function used to convert the distance map (i.e. `||embeddings - anchor_embedding||`) into an instance mask
class Gaussian(nn.Module):
    def __init__(self, delta_var, pmaps_threshold):
        super().__init__()
        self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

    def forward(self, dist_map):
        return torch.exp(- dist_map * dist_map / self.two_sigma)


def compute_cluster_means(embeddings, target, n_instances):
    """
    Computes mean embeddings per instance, embeddings withing a given instance and the number of voxels per instance.

    C - number of instances
    E - embedding dimension
    SPATIAL - volume shape, i.e. DxHxW for 3D/ HxW for 2D

    Args:
        embeddings: tensor of pixel embeddings, shape: ExSPATIAL
        target: one-hot encoded target instances, shape: CxSPATIAL
    """
    target = expand_as_one_hot(target.unsqueeze(0), n_instances).squeeze(0)
    target = target.unsqueeze(1)
    spatial_ndim = embeddings.dim() - 1
    dim_arg = (2, 3) if spatial_ndim == 2 else (2, 3, 4)

    embedding_dim = embeddings.size(0)

    # get number of pixels in each cluster; output: Cx1
    num_pixels = torch.sum(target, dim=dim_arg)

    # expand target: Cx1xSPATIAL -> CxExSPATIAL
    shape = list(target.size())
    shape[1] = embedding_dim
    target = target.expand(shape)

    # expand input_: ExSPATIAL -> 1xExSPATIAL
    embeddings = embeddings.unsqueeze(0)

    # sum embeddings in each instance (multiply first via broadcasting); embeddings_per_instance shape CxExSPATIAL
    embeddings_per_instance = embeddings * target
    # num's shape: CxEx1(SPATIAL)
    num = torch.sum(embeddings_per_instance, dim=dim_arg)

    # compute mean embeddings per instance CxE
    mean_embeddings = num / num_pixels

    return mean_embeddings


def expand_as_one_hot(input_, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input_ (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        output torch.Tensor of size (NxCxSPATIAL)
    """
    assert input_.dim() > 2

    # expand the input tensor to Nx1xSPATIAL before scattering
    input_ = input_.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input_.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input_.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input_ = input_.clone()
        input_[input_ == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input_.device).scatter_(1, input_, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input_.device).scatter_(1, input_, 1)
