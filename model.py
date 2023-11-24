import math
import random
from random import random
# from random import seed
import numpy as np
from numpy.random import seed
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus, normalize
from torch.autograd import Variable, Function

seed(825)

class FastCNN(nn.Module):
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        super(FastCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(200, channel, kernel_size=kernel),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                )
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_out = []
        for module in self.fast_cnn:
            module_out = module(x).squeeze()

            # prints shape with < 2 dimensions before error
            # print('Module out shape' +  str(module_out.shape))

            # potential solution
            if len(module_out.shape) < 2:
                module_out = torch.unsqueeze(module_out, 1)

            x_out.append(module_out)
        x_out = torch.cat(x_out, dim=1)
        return x_out


class EncodingPart(nn.Module):
    def __init__(
        self,
        cnn_channel=32,
        cnn_kernel_size=(1, 2, 4, 8),
        shared_image_dim=128,
        shared_text_dim=128
    ):
        super(EncodingPart, self).__init__()
        self.shared_text_encoding = FastCNN(
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_encoding = self.shared_text_encoding(text)
        # print('text.shape' + str(text.shape))
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class SimilarityModule(nn.Module):
    def __init__(self, shared_dim=128, sim_dim=64):
        super(SimilarityModule, self).__init__()
        self.encoding = EncodingPart()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text, image):
        text_encoding, image_encoding = self.encoding(text, image)
        text_aligned = self.text_aligner(text_encoding)
        image_aligned = self.image_aligner(image_encoding)
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, image_aligned, pred_similarity

    def cal_conloss(self, text, image,label, con_loss):
        text_encoding, image_encoding = self.encoding(text, image)
        features = torch.cat([text_encoding.unsqueeze(1), image_encoding.unsqueeze(1)], dim=1)
        loss_con1 = con_loss(image_encoding.unsqueeze(1), label)
        loss_con2 = con_loss(text_encoding.unsqueeze(1), label)
        loss_intra = loss_con1 + loss_con2
        loss_con = con_loss(features, label)
        return loss_con + loss_intra

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  
        return Independent(Normal(loc=mu, scale=sigma), 1)


class ConsistencyMeasurement(nn.Module):
    def __init__(self):
        super(ConsistencyMeasurement, self).__init__()
        self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = torch.sigmoid(skl)
        return skl


class UnimodalDetection(nn.Module):
        def __init__(self, shared_dim=128, prime_dim = 16):
            super(UnimodalDetection, self).__init__()
            self.text_uni = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU()
            )
            self.image_uni = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU()
            )

        def forward(self, text_encoding, image_encoding):
            text_prime = self.text_uni(text_encoding)
            image_prime = self.image_uni(image_encoding)
            return text_prime, image_prime


class CrossModule4Batch(nn.Module):
    def __init__(self, text_in_dim=64, image_in_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )
        self.gconv1 = nn.Sequential(
            nn.Linear(self.corre_dim, 64),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )
        self.gconv2 = nn.Sequential(
            nn.Linear(64, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image, in_aff, out_aff):
        # text = self.attention_1(text,image,text)
        # correlation_out = self.attention_2(image, text,image)
        text_in = text.unsqueeze(2)
        image_in = image.unsqueeze(1)
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)

        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.c_specific_2(correlation_p)

        correlation_out = self.gconv1(torch.sparse.mm(in_aff, correlation_out))
        correlation_out = self.gconv2(torch.sparse.mm(out_aff, correlation_out))
        return correlation_out



class DetectionModule(nn.Module):
    def __init__(self, feature_dim=64+16+16, h_dim=64):
        super(DetectionModule, self).__init__()
        self.encoding = EncodingPart()
        self.ambiguity_module = ConsistencyMeasurement()
        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule4Batch()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(h_dim, 2)
        )

    def forward(self, text_raw, image_raw, text, image, in_aff, out_aff):
        # text_encoding, image_encoding = self.encoding_module(text, image)
        skl = self.ambiguity_module(text, image)
        text_prime, image_prime = self.encoding(text_raw, image_raw)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)
        correlation = self.cross_module(text, image, in_aff, out_aff)
        weight_uni = (1-skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        text_final = weight_uni * text_prime
        img_final = weight_uni * image_prime
        corre_final = weight_corre * correlation
        final_corre = torch.cat([text_final, img_final, corre_final], 1)
        pre_label = self.classifier_corre(final_corre)

        return pre_label


from typing import Any, Optional, Tuple
#%%
class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()
    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)
    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

def grad_reverse(x, lambd=1.0):
    lam = torch.tensor(lambd)
    return GradReverse.apply(x,lam)

class GradReverseLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(GradReverseLayer,self).__init__()
        self.lambd = lambd
    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x,lam)