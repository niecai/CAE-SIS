import numpy as np
import random
import math


def encryptionInSMIESIS(data, share_matrix):
    le = len(data)
    K = np.random.randint(0, 2, (192, ))

    a = [0] * 4
    for i in range(48):
        for j in range(4):
            a[j] += K[j * 48 + i]
            a[j] /= 2

    c = a[0] * a[1] * a[2] * a[3] * math.pow(2, 48)
    r1 = (c + a[0]) % 0.4 + 3.6
    r2 = (c + a[1]) % 0.4 + 3.6
    c1 = [(c + a[2]) % 1]
    c2 = [(c + a[3]) % 1]

    for i in range(0, le - 1):
        if c1[i] < 0.5:
            c1.append(r1 * c1[i] / 2)
        else:
            c1.append(r1 * (1 - c1[i]) / 2)

        if c2[i] < 0.5:
            c2.append(r2 * c2[i] / 2)
        else:
            c2.append(r2 * (1 - c2[i]) / 2)

    E1 = [0] * le
    E2 = [0] * le
    E1[0] = data[0]
    multiple = 10000000000000
    for i in range(1, le):
        E1[i] = (data[i] + int(c1[i] * multiple) + data[i - 1]) % 256
    E2[-1] = E1[-1]
    for i in range(le - 2, -1, -1):
        E2[i] = (E1[i] + int(c2[i] * multiple) + E1[i + 1]) % 256

    sum_E2 = sum(E2)
    Ks = np.copy(K)
    i = 191
    while sum_E2:
        Ks[i] = K[i] ^ (sum_E2 & 1)
        sum_E2 >>= 1
        i -= 1

    bit_replace = [1] * 8
    for i in range(1, 8):
        bit_replace[i] = bit_replace[i - 1] << 1

    E = []
    for i in range(24):
        E.append(0)
        for j in range(8):
            E[-1] += bit_replace[j] * Ks[i * 8 + j]
    E += E2

    e_l = len(E)
    n, d = share_matrix.shape
    D = [d // 256, d % 256]
    share = []

    for k in range(n):
        B = []
        for i in range(math.ceil(d / 8)):
            b = 0
            for j in range(min(d - 8 * i, 8)):
                b += bit_replace[j] * share_matrix[k][j]
            B.append(b)

        H = []
        for i in range(math.ceil(e_l / d)):
            for j in range(min(e_l - d * i, d)):
                if share_matrix[k][j]:
                    H.append(E[d * i + j])

        share.append([D, B, H])

    return share


def decryptionInSMIESIS(data_list):
    bit_replace = [1] * 8
    for i in range(1, 8):
        bit_replace[i] = bit_replace[i - 1] << 1

    share_matrix = []
    d = data_list[0][0][0] * 256 + data_list[0][0][1]
    for _, B, _ in data_list:
        row = []
        i = 0
        j = 0
        while i < d:
            if B[i // 8] & bit_replace[j]:
                row.append(1)
            else:
                row.append(0)
            i += 1
            j = (j + 1) % 8
        share_matrix.append(row)

    reconstructionE = []
    max_len = 0
    for i, e in enumerate(data_list):
        _, _, H = e
        row = share_matrix[i]
        l_h = len(H)
        i = 0
        j = 0
        E = []

        while i < l_h:
            if row[j]:
                E.append(H[i])
                i += 1
            else:
                E.append(0)
            j = (j + 1) % d
        reconstructionE.append(E)
        max_len = max(max_len, len(E))
    for E in reconstructionE:
        while len(E) < max_len:
            E.append(0)

    Ed = reconstructionE[0]
    for i in range(1, len(reconstructionE)):
        for j in range(max_len):
            Ed[j] |= reconstructionE[i][j]

    Ks, E2 = Ed[:24], Ed[24:]
    K = []
    for i in range(24):
        for j in range(8):
            if Ks[i] & bit_replace[j]:
                K.append(1)
            else:
                K.append(0)
    sum_e2 = sum(E2)
    data_len = len(E2)
    i = 191
    now = 1
    while now <= sum_e2:
        if sum_e2 & now:
            K[i] ^= 1
        now <<= 1
        i -= 1

    a = [0] * 4
    for i in range(48):
        for j in range(4):
            a[j] += K[j * 48 + i]
            a[j] /= 2

    c = a[0] * a[1] * a[2] * a[3] * math.pow(2, 48)
    r1 = (c + a[0]) % 0.4 + 3.6
    r2 = (c + a[1]) % 0.4 + 3.6
    c1 = [(c + a[2]) % 1]
    c2 = [(c + a[3]) % 1]

    for i in range(0, data_len - 1):
        if c1[i] < 0.5:
            c1.append(r1 * c1[i] / 2)
        else:
            c1.append(r1 * (1 - c1[i]) / 2)

        if c2[i] < 0.5:
            c2.append(r2 * c2[i] / 2)
        else:
            c2.append(r2 * (1 - c2[i]) / 2)

    multiple = 10000000000000
    E1 = [0] * data_len
    E1[-1] = E2[-1]
    for i in range(data_len - 2, -1, -1):
        E1[i] = ((E2[i] - E1[i + 1] - int(c2[i] * multiple)) % 256 + 256) % 256
    res = [0] * data_len
    res[0] = E1[0]
    for i in range(1, data_len):
        res[i] = ((E1[i] - res[i - 1] - int(c1[i] * multiple)) % 256 + 256) % 256
    print("testtesttest")
    return np.asarray(res)


def encryptionInAE(data, n, choose_matrix, block_len, share_matrix, data_len, feature):
    share_list = []
    data.dtype = 'uint32'
    choose_matrix = np.asarray(choose_matrix, dtype=np.uint32) * np.iinfo('uint32').max

    # Splitting on element attributes
    dip = math.ceil(32 / block_len)
    bit_choose = np.repeat(share_matrix, dip, axis=0)
    bit_choose = bit_choose.reshape(n, dip * block_len)
    bit_choose = bit_choose[:, :32]

    bit_replace = [1] * 32
    for i in range(1, 32):
        bit_replace[i] = bit_replace[i - 1] << 1
    bit_replace = np.asarray(bit_replace, dtype='uint32')[:, np.newaxis]
    bit_replace = (np.matmul(bit_choose, bit_replace))

    replace_choose = np.random.randint(0, 2, (data_len,), dtype=np.bool)
    choose_matrix[:, replace_choose] = bit_replace

    def get_random_list(l):
        res = list(range(l))
        for i in range(l - 1, 0, -1):
            x = random.randint(0, i - 1)
            res[x], res[i] = res[i], res[x]
        return res

    for i in range(feature):
        choose_matrix[:, [i]] = bit_replace[get_random_list(n), :]

    for i in range(n):
        print('generation {} share.'.format(i))
        share_list.append(np.bitwise_and(data, choose_matrix[i]))
    return share_list


def initialMatrixGeneration(k, direc_generation=False, n=None):
    matrix = []
    le = n if direc_generation else (2 * k - 2)
    now = [1] * le

    def findColumn(index, x):
        if x == k - 1:
            matrix.append(now.copy())
            return
        if index == le:
            return
        now[index] = 0
        findColumn(index + 1, x + 1)
        now[index] = 1
        findColumn(index + 1, x)

    findColumn(0, 0)
    shuffle_index = list(range(le))
    random.shuffle(shuffle_index)
    return np.asarray(matrix, dtype=np.bool).T[shuffle_index, :]


def matrixExpansion(s0, matrix, k):
    n, m = s0.shape
    now_n, now_m = matrix.shape
    block_len = now_n // (k - 1)
    replace = np.reshape(matrix, (k - 1, block_len, now_m))
    res = np.ones((now_n * 2, now_m * m), dtype=int)
    for i in range(m):
        l = 0
        for j in range(n):
            if s0[j, i]:
                res[(j * block_len):((j + 1) * block_len), (i * now_m):((i + 1) * now_m)] = replace[l]
                l += 1
    return res


def decomposition(data, k, n, direc_generation=False, encryption_method=None, feature=None):
    if encryption_method == 'AE' or encryption_method == 'SMIESIS':
        # The generation method of this paper
        if direc_generation:
            share_matrix = initialMatrixGeneration(k, direc_generation=direc_generation, n=n)
        # The original generation method in Bao's paper
        else:
            share_matrix = initialMatrixGeneration(k)
            first_matrix = np.copy(share_matrix)
            for _ in range(max(math.ceil(math.log2(n / (2 * k - 2))), 0)):
                share_matrix = matrixExpansion(first_matrix, share_matrix, k)
            choose = list(range(share_matrix.shape[0]))
            random.shuffle(choose)
            share_matrix = share_matrix[choose[:n], :]

        block_len = share_matrix.shape[1]
        data_len = len(data)
        print("data_len:{}".format(data_len))
        dip = math.ceil(data_len / block_len)
        choose_matrix = np.repeat(share_matrix, dip, axis=0)
        choose_matrix = choose_matrix.reshape(n, dip * block_len)
        choose_matrix = choose_matrix[:, :data_len]
        if encryption_method == 'AE':
            share_list = encryptionInAE(data, n, choose_matrix, block_len, share_matrix, data_len, feature)
        else:
            share_list = encryptionInSMIESIS(data, share_matrix=share_matrix)
    else:
        # Additional methods to be added
        share_list = []
    return share_list


def reconstruction(data_list, encryption_method=None):
    if encryption_method == 'AE':
        res = data_list[0]
        for i in range(1, len(data_list)):
            res = np.bitwise_or(res, data_list[i])
        res.dtype = 'float32'
    elif encryption_method == 'SMIESIS':
        res = decryptionInSMIESIS(data_list)
    else:
        # Additional methods to be added
        res = None
    return res
