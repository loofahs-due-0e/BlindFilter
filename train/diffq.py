import numpy as np
import torch
import torch.nn as nn
import time

# constants
num_ham = 16545
num_spam = 17170
num_data = num_ham + num_spam
num_token = 30522

# hyper-parameters
l2_lambda = 0.

#                diffq_lambda  kurtosis_lambda
# noise_only     0.56          0.0
# kurtosis_only  0.0           1.0
# both           6.8           1.0
diffq_lambda = 6.8
diffq_minbit = 0.1
diffq_maxbit = 2.
diffq_initbit = 1.28

kurtosis_lambda = 1.

train_data = 0.1  # fraction of dataset for training, [0, 1)

scale = 100.

bitwidth_init_from_prev = False
save_result = True

batch_size = 128
epoch = 100
lr_0 = 5e-4

result_accum_repeat = 1

device = 'cuda'  # or 'cpu'


def _hardtanh(x):
    return torch.clamp(x, -1, 1)


def _tanh_ste(x):
    t = torch.tanh(x)
    return (_hardtanh(x) - t).detach() + t


def _num_bits(x, ste):
    """
        ste    : boolean, whether to apply STE or not
        return : tensor of shape [diffq_groups], learned number of bits
                    in range of [min_bits, max_bits]
    """
    t = None
    if ste:
        t = _tanh_ste(x)
    else:
        t = _hardtanh(x)
    return (diffq_maxbit * (1. + t) + diffq_minbit * (1. - t)) / 2.


def rand_like(x):
    return torch.rand_like(x) - 0.5  # uniform
    # return torch.randn_like(x) * 0.5  # gaussian


def num_bit_to_logit(x):
    return (2 * x - diffq_maxbit - diffq_minbit) / (diffq_maxbit - diffq_minbit)


def kurtosis(x, target=1.8):
    k = torch.mean(((x - torch.mean(x)) / torch.std(x)) ** 4.)
    return (k - target) ** 2.


if __name__ == "__main__":
    # load data
    data = np.zeros([num_data, num_token])
    label = np.zeros([num_data])

    print('loading dataset...')
    f = open('./enron/spam_index', 'r')
    f2 = open('./enron/ham_index', 'r')

    lines = f.readlines()
    lines2 = f2.readlines()

    i = 0
    for line in lines:
        used_tokens = line.rstrip('\n').split(' ')
        for t in used_tokens:
            data[i][int(t)] = 1.
        label[i] = 0.  # spam
        i += 1

    for line in lines2:
        used_tokens = line.rstrip('\n').split(' ')
        for t in used_tokens:
            data[i][int(t)] = 1.
        label[i] = 1.  # ham
        i += 1

    f.close()
    f2.close()

    print('loading completed, shuffling...')
    random_index = [i for i in range(num_ham + num_spam)]
    np.random.shuffle(random_index)
    data = data[random_index]
    label = label[random_index]

    num_train_data = round((num_ham + num_spam) * train_data)
    num_val_data = num_data - num_train_data
    train_data = data[:num_train_data]
    val_data = data[num_train_data:]
    train_label = label[:num_train_data]
    val_label = label[num_train_data:]

    # load model
    filename = '7_0.957'
    f = open('./' + filename + '.bin', 'r')
    lines = f.readlines()
    params = nn.Parameter(
        torch.tensor(
            [float(l) for l in lines],
            dtype=torch.float32, device=device))
    f.close()

    threshold = 0.957

    assert len(params) == num_token

    # diffQ
    bitlogit_mask = [1] * num_token
    bitlogit_invalid = 0
    bitlogit_arr = None
    if bitwidth_init_from_prev:
        f = open('bitwidth_orig.txt', 'r')
        bitlogit_arr = [num_bit_to_logit(float(l)) for l in f.readlines()]
        f.close()
        for i in range(len(bitlogit_arr)):
            if bitlogit_arr[i] > 0.1:
                bitlogit_arr[i] = num_bit_to_logit(diffq_initbit)
        assert len(bitlogit_arr) == num_token
    else:
        bitlogit_arr = [num_bit_to_logit(diffq_initbit)] * num_token
        min_logit = num_bit_to_logit(diffq_minbit)
        for i in range(num_token):
            if params[i] == 0.036668803637087244:
                bitlogit_arr[i] = min_logit
                bitlogit_mask[i] = 0
                bitlogit_invalid += 1
    bitwidth_logit = nn.Parameter(
        torch.tensor(bitlogit_arr, dtype=torch.float32, device=device))

    # evaluate on whole dataset
    print('evaluating...')
    with torch.no_grad():
        filter_result = torch.inner(
            params.clone().detach(),
            torch.tensor(data, dtype=torch.float32, device=device)) -\
            (np.log(1./threshold) - 1.)
        binary_result = filter_result > 0.
        num_correct_ham = torch.inner(
            binary_result.cpu().long(),
            torch.tensor(label, dtype=torch.long, device='cpu')).item()
        num_correct_spam = torch.inner(
            1 - binary_result.cpu().long(),
            torch.tensor(1.-label, dtype=torch.long, device='cpu')).item()
        num_testingHam = np.sum(label)
        num_testingSpam = np.sum(1.-label)

        TP = num_correct_ham
        FP = num_testingHam - num_correct_ham
        FN = num_testingSpam - num_correct_spam

        print(
            'baseline accuracy = {}'.format(
                torch.sum(
                    binary_result == torch.tensor(label, device=device)).item() /
                num_data))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2*(recall * precision) / (recall + precision)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1_score}')

    if save_result:
        baseline_result = []
        f = open('baseline.txt', 'w')
        with torch.no_grad():
            for _ in range(result_accum_repeat):
                b_result = np.inner(params.cpu().clone(
                ).detach(), data) - (np.log(1./threshold) - 1.)
                baseline_result.append('\n'.join(str(f) for f in b_result))
        f.write('\n'.join(f for f in baseline_result))
        f.close()

    # train
    print('start training...')
    start = time.clock_gettime(0)
    optimizer = torch.optim.Adam([params, bitwidth_logit], lr=lr_0)
    loss = nn.BCELoss()
    k_loss = []
    bce_loss = []
    diffq_loss = []
    for e in range(epoch):
        print('epoch {} / {}'.format(e+1, epoch))
        num_steps = num_train_data // batch_size
        residue = num_train_data % batch_size
        for s in range(num_steps + (residue > 0)):
            optimizer.zero_grad()
            # activation : [batch_size, num_token]
            off = batch_size if s != num_steps else residue
            start_i = s * batch_size
            act = torch.tensor(train_data[start_i: start_i + off],
                               dtype=torch.float32, device=device)
            l = torch.tensor(train_label[start_i: start_i + off],
                             dtype=torch.float32, device=device)

            # weight : [num_token]
            # noise : [num_token]
            noise = torch.abs(params) /\
                (torch.pow(2., _num_bits(bitwidth_logit, ste=True)) - 1) *\
                rand_like(params)
            w = params + noise

            # [batch_size]
            filter_result = torch.inner(act, w) -\
                (np.log(1./threshold) - 1.)
            result = torch.sigmoid(filter_result / scale)

            k = kurtosis(filter_result)
            k_loss.append(k.clone().detach().cpu())
            l2 = torch.linalg.norm(params, dim=0, ord=2)
            bce = loss(result, l)
            bce_loss.append(bce.clone().detach().cpu())
            diffq_l = torch.mean(_num_bits(bitwidth_logit, ste=True))
            diffq_loss.append(diffq_l.clone().detach().cpu())

            output = loss(result, l) + kurtosis_lambda * k +\
                l2_lambda * l2 + diffq_lambda * diffq_l

            output.backward()
            optimizer.step()

        with torch.no_grad():
            num_steps = num_val_data // batch_size
            residue = num_val_data % batch_size
            accurate = 0
            for s in range(num_steps + (residue > 0)):
                off = batch_size if s != num_steps else residue
                start_i = s * batch_size
                act = torch.tensor(val_data[start_i: start_i + off],
                                   dtype=torch.float32, device=device)
                l = torch.tensor(val_label[start_i: start_i + off],
                                 dtype=torch.float32, device=device)

                noise = torch.abs(params) /\
                    (torch.pow(2., _num_bits(bitwidth_logit, ste=True)) - 1) *\
                    rand_like(params)
                w = params + noise

                filter_result = torch.inner(act, w) -\
                    (np.log(1./threshold) - 1.)
                result = torch.sigmoid(filter_result / scale)

                binary_result = result >= 0.5

                accurate += torch.sum(binary_result == l).item()

        print('    val accuracy = {}'.format(accurate / num_val_data))
        print(
            'average bitwidth = {}'.format(
                np.inner(_num_bits(bitwidth_logit, ste=False).cpu().detach().numpy(),
                         bitlogit_mask) / (30522 - bitlogit_invalid)
            ))

    # final evaluation on whole dataset
    num_test = 256
    act = torch.tensor(data, dtype=torch.float32, device=device)
    precision = np.zeros([num_test])
    recall = np.zeros([num_test])
    acc = np.zeros([num_test])
    for i in range(num_test):
        noise = torch.abs(params) /\
            (torch.pow(2., _num_bits(bitwidth_logit, ste=True)) - 1) *\
            rand_like(params)
        w = params + noise
        filter_result = torch.inner(act, w) - (np.log(1./threshold) - 1.)
        br = filter_result >= 0.
        t_label = torch.tensor(label, dtype=bool, device=device)

        fp = torch.sum(t_label & ~br).item()  # label 1, binary result 0
        fn = torch.sum(~t_label & br).item()  # label 0, binary result 1
        tp = torch.sum(~t_label & ~br).item()  # label 0, binary result 0
        tn = torch.sum(t_label & br).item()  # label 1, binary result 1
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        acc[i] = (tp + tn) / (tp + tn + fp + fn)

    print('precision : {}({})'.format(
        precision.mean(), precision.std()))
    print('recall : {}({})'.format(recall.mean(), recall.std()))
    f1 = 2 / (1 / precision + 1 / recall)
    print('f1 : {}({})'.format(f1.mean(), f1.std()))
    print('accuracy: {}({})'.format(acc.mean(), acc.std()))

    print('training ended in {} s'.format(int(time.clock_gettime(0) - start)))
    bw = '\n'.join(str(b)
                   for b in _num_bits(bitwidth_logit,
                                      ste=False).cpu().detach().numpy())
    f = open('bitwidth.txt', 'w')
    f.write(bw)
    f.close()

    f = open('param.txt', 'w')
    f.write('\n'.join([str(p) for p in params.cpu().detach().numpy()]))
    f.close()

    f = open(filename + '.csv', 'w')
    buffer = []
    noise_range = abs(params) /\
        (torch.pow(2., _num_bits(bitwidth_logit, ste=False)) - 1.)
    for i in range(len(params)):
        buffer.append(str(params[i].item()) + ',' + str(noise_range[i].item()))
    f.write('\n'.join(buffer))
    f.close()

    if save_result:
        result_noise = []
        result = []
        f = open('result_noise.txt', 'w')
        with torch.no_grad():
            for _ in range(result_accum_repeat):
                filter_result = np.inner(params.cpu().clone(
                ).detach(), data) - (np.log(1./threshold) - 1.)
                result.append('\n'.join(str(f) for f in filter_result))
            for _ in range(result_accum_repeat):
                for i in range(len(data)):
                    w = params +\
                        abs(params) / (torch.pow(2.,
                                                 _num_bits(bitwidth_logit, ste=True)) - 1) *\
                        rand_like(params)
                    filter_noise_result = np.inner(
                        w.cpu().clone().detach(), data[i]) - (np.log(1./threshold) - 1.)
                    result_noise.append(str(filter_noise_result))

        f.write('\n'.join(f for f in result_noise))
        f.close()

        f = open('result.txt', 'w')
        f.write('\n'.join(f for f in result))
        f.close()
