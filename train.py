import copy
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
from tqdm import tqdm
from dataset import FeatureDataset
from dataset_test import FeatureDataset1
from losses import SupConLoss
from model import SimilarityModule, DetectionModule
from numpy.random import seed

DEVICE = "cuda:0"
NUM_WORKER = 1
BATCH_SIZE = 128
LR = 0.001  # 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 100
TEMP = 0.05


def prepare_data(text, image, label):
    nr_index = [i for i, l in enumerate(label) if l == 1]
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)
    return fixed_text, matched_image, unmatched_image


def euclidean_dist(x, y):
    """
    Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
    Returns:
    dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def normalize(affnty):
    col_sum = affnty.sum(axis=1)[:, np.newaxis]
    row_sum = affnty.sum(axis=0)

    out_affnty = affnty / col_sum
    in_affnty = (affnty / row_sum).t()

    out_affnty = Variable(torch.Tensor(out_affnty)).cuda()
    in_affnty = Variable(torch.Tensor(in_affnty)).cuda()

    return in_affnty, out_affnty


def train():
    # ---  Load Config  ---
    device = torch.device(DEVICE)
    num_workers = NUM_WORKER
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH

    # ---  Load Data  ---
    dataset_dir = 'data/twitter'
    train_set = FeatureDataset(
        "{}/train_text_with_label.npz".format(dataset_dir),
        "{}/train_image_with_label.npz".format(dataset_dir)
    )
    test_set = FeatureDataset(
        "{}/test_text_with_label.npz".format(dataset_dir),
        "{}/test_image_with_label.npz".format(dataset_dir)
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # ---  Build Model & Trainer  ---
    similarity_module = SimilarityModule()
    similarity_module.to(device)
    detection_module = DetectionModule()
    detection_module.to(device)
    loss_func_similarity = torch.nn.CosineEmbeddingLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_contrastive = SupConLoss(temperature=TEMP)
    optim_task_similarity = torch.optim.Adam(
        similarity_module.parameters(), lr=lr, weight_decay=l2
    )
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=lr, weight_decay=l2
    )

    # ---  Model Training  ---
    best_acc = 0
    best_confusion_matrix = [[]]
    best_f1 = 0
    best_p = 0
    best_r = 0
    for epoch in range(num_epoch):

        similarity_module.train()
        detection_module.train()
        corrects_pre_detection = 0
        loss_detection_total = 0
        similarity_count = 0
        detection_count = 0

        for i, (text, image, label) in tqdm(enumerate(train_loader)):
            batch_size = text.shape[0]
            text = text.to(device)
            image = image.to(device)
            label = label.to(device)

            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)

            # ---  TASK2 Detection  ---
            text_aligned, image_aligned, _ = similarity_module(text, image)
            loss_con = similarity_module.cal_conloss(text, image, label, loss_func_contrastive)
            optim_task_similarity.zero_grad()
            loss_con.backward(retain_graph=True)
            optim_task_similarity.step()

            # 邻接矩阵
            text_aligned, image_aligned, _ = similarity_module(text, image)
            sim_origin = label.float().unsqueeze(0).mm(label.float().unsqueeze(0).t())
            adj = (sim_origin > 0).float()
            C = torch.cat((2 * text_aligned, 0.3 * image_aligned), 1)
            dis_C = euclidean_dist(C, C)
            A_C = torch.exp(-dis_C / 4)
            C = C.mm(C.t()) * A_C
            C = C * adj
            in_aff, out_aff = normalize(C.type(torch.FloatTensor))
            pre_detection = detection_module(text, image, text_aligned, image_aligned, in_aff, out_aff)
            loss_detection = loss_func_detection(pre_detection, label)

            optim_task_detection.zero_grad()
            loss_detection = loss_detection
            loss_detection.backward()
            optim_task_detection.step()

            pre_label_detection = pre_detection.argmax(1)
            corrects_pre_detection += pre_label_detection.eq(label.view_as(pre_label_detection)).sum().item()

            # ---  Record  ---
            loss_detection_total += loss_detection.item() * text.shape[0]
            similarity_count += (2 * fixed_text.shape[0] * 2)
            detection_count += text.shape[0]

        loss_detection_train = loss_detection_total / detection_count
        acc_detection_train = corrects_pre_detection / detection_count

        # ---  Test  ---
        acc_similarity_test, acc_detection_test, loss_similarity_test, loss_detection_test, cm_similarity, cm_detection = test(
        similarity_module, detection_module, test_loader)

        if acc_detection_test >= best_acc:
            best_acc = acc_detection_test
            best_confusion_matrix = cm_detection
            best_p = best_confusion_matrix.diagonal() / np.sum(best_confusion_matrix, axis=1)
            best_r = best_confusion_matrix.diagonal() / np.sum(best_confusion_matrix, axis=0)
            best_f1 = 2*np.multiply(best_p,best_r)/(best_p+best_r)
        print('---  TASK2 Detection  ---')
        print(
            "EPOCH = %d \n acc_detection_train = %.3f \n acc_detection_test = %.3f \n  best_acc = %.3f \n loss_detection_train = %.3f \n loss_detection_test = %.3f \n" %
            (epoch + 1, acc_detection_train, acc_detection_test, best_acc, loss_detection_train, loss_detection_test)
        )
        print('best_p = {} \n best_r = {} \n best_f1 = {} \n'.format(best_p, best_r, best_f1))
        print('best_confusion_matrix is\n {}\n'.format(best_confusion_matrix))

        print('---  TASK1 Similarity Confusion Matrix  ---')
        print('{}\n'.format(cm_similarity))

        print('---  TASK2 Detection Confusion Matrix  ---')
        print('{}\n'.format(cm_detection))


def test(similarity_module, detection_module, test_loader):
    similarity_module.eval()
    detection_module.eval()

    device = torch.device(DEVICE)
    loss_func_detection = torch.nn.CrossEntropyLoss()

    similarity_count = 0
    detection_count = 0
    loss_similarity_total = 0
    loss_detection_total = 0
    similarity_label_all = []
    detection_label_all = []
    similarity_pre_label_all = []
    detection_pre_label_all = []

    with torch.no_grad():
        for i, (text, image, label) in enumerate(test_loader):
            batch_size = text.shape[0]
            text = text.to(device)
            image = image.to(device)
            label = label.to(device)

            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)

            text_aligned, image_aligned, _ = similarity_module(text, image)

            sim_origin = label.float().unsqueeze(1).mm(label.float().unsqueeze(1).t())
            adj = (sim_origin > 0).float()
            C = torch.cat((2 * text_aligned, 0.3 * image_aligned), 1)
            dis_C = euclidean_dist(C, C)
            A_C = torch.exp(-dis_C / 4)
            C = C.mm(C.t()) * A_C
            C = C * adj

            in_aff, out_aff = normalize(C.type(torch.FloatTensor))
            pre_detection = detection_module(text, image, text_aligned, image_aligned, in_aff, out_aff)
            loss_detection = loss_func_detection(pre_detection, label)
            pre_label_detection = pre_detection.argmax(1)

            # ---  Record  ---

            loss_detection_total += loss_detection.item() * text.shape[0]
            similarity_count += (fixed_text.shape[0] * 2)
            detection_count += text.shape[0]

            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

        loss_similarity_test = loss_similarity_total / similarity_count
        loss_detection_test = loss_detection_total / detection_count

        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        acc_similarity_test = accuracy_score(similarity_pre_label_all, similarity_label_all)
        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        cm_similarity = confusion_matrix(similarity_pre_label_all, similarity_label_all)
        cm_detection = confusion_matrix(detection_pre_label_all, detection_label_all)

    return acc_similarity_test, acc_detection_test, loss_similarity_test, loss_detection_test, cm_similarity, cm_detection


if __name__ == "__main__":
    seed(825)
    train()
