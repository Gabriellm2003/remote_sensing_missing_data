import torch


def weighted_soft_margin_triple_loss(distance_matrix, batch_size, loss_weight=10):
	pair_n = batch_size * (batch_size - 1.0)

	pos_dist = torch.diag(distance_matrix)
	triplet_g2s = pos_dist - distance_matrix
	loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_g2s * loss_weight)))/pair_n

	triplet_s2g = torch.unsqueeze(pos_dist, 1) - distance_matrix
	loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_s2g * loss_weight)))/pair_n
	
	loss = (loss_s2g + loss_g2s)/2.0
	
	return loss