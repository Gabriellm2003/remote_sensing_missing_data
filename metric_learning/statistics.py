import os
import numpy as np
import statistics


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def generate_retrieval_list(dist_array, topK, img_ids, img_ids2, output_file):

    for i in range(dist_array.shape[0]):
        position = np.argpartition(dist_array[i, :], topK)
        indexes = position[:topK]
        output_file.write(str(img_ids2[i]).replace('b\'', '') + ' : ')
        my_dict = {}
        for j in range(topK):
            my_dict[indexes[j]] = dist_array[i, indexes[j]]
        sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
        counter = 0
        for k, v in sorted_dict.items():
            if (counter != topK-1):
                output_file.write(str(img_ids[k]).replace('b\'', '') + ',')
            else:
                output_file.write(str(img_ids[k]).replace('b\'', ''))
            counter += 1
        output_file.write('\n')
        




def extract_ranking(distance_matrix, top_k, img_ids, img_ids2, query_type):
    labels_dict = {'airport': 0, 'bridge': 1, 'church': 2, 'forest': 3, 'lake': 4, 
                   'park': 5, 'river': 6, 'skyscraper': 7, 'stadium': 8, 'statue': 9, 'tower': 10,
                   'apartment':0, 'apartament': 0, 'hospital': 1, 'house': 1, 'industrial': 2, 
                   'parking_lot': 3, 'religious': 4, 'school': 5, 'store': 6, 'vacant_lot': 8}
    pred_array = np.zeros((img_ids.shape[0], top_k))
    if (query_type == 'ground'):
        for i in range(distance_matrix.shape[0]):
            position = np.argpartition(distance_matrix[i, :], top_k)
            indexes = position[:top_k]
            classe = labels_dict[str(img_ids2[i]).split('___')[0].replace("b'", "")]
            my_dict = {}
            for j in range(top_k):
                my_dict[indexes[j]] = distance_matrix[i, indexes[j]]
            sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
            counter = 0
            for k, v in sorted_dict.items():
                if (labels_dict[str(img_ids[k]).split('___')[0].replace("b'", "")] == classe):
                    pred_array[i][counter] = 1
                else:
                    pred_array[i][counter] = 0
                counter += 1
        return pred_array


    if (query_type == 'aerial'):
        for i in range(distance_matrix.shape[0]):
            position = np.argpartition(distance_matrix[:, i], top_k)
            indexes = position[:top_k]
            classe = labels_dict[str(img_ids[i]).split('___')[0].replace("b'", "")]
            my_dict = {}
            for j in range(top_k):
                my_dict[indexes[j]] = distance_matrix[indexes[j], i]
            sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
            counter = 0
            for k, v in sorted_dict.items():
                if (labels_dict[str(img_ids2[k]).split('___')[0].replace("b'", "")] == classe):
                    pred_array[i][counter] = 1
                else:
                    pred_array[i][counter] = 0
                counter += 1
        return pred_array


        



def calculate_mAP(distance_matrix, ids_aerial, ids_ground):
    rankings = [1, 2, 3, 4, 5, 10, 50, 100]
    
    print ("Mean Average Precision - Ground as query:")
    for i in rankings:
        pred_array = extract_ranking(distance_matrix, i, ids_aerial, ids_ground, 'ground')
        m_ap = mean_average_precision(pred_array)
        print ("TOP " + str(i) + ": " + str(m_ap))

    print ("Mean Average Precision - Aerial as query:")
    for i in rankings:
        pred_array = extract_ranking(distance_matrix, i, ids_aerial, ids_ground, 'aerial')
        m_ap = mean_average_precision(pred_array)
        print ("TOP " + str(i) + ": " + str(m_ap))

def calculate_mAP_test(distance_matrix, ids_aerial, ids_ground, query, out_file):
    rankings = range(1,101)
    mAPs = []
    
    if (query == 'ground'):
    	for i in rankings:
        	pred_array = extract_ranking(distance_matrix, i, ids_aerial, ids_ground, 'ground')
        	m_ap = mean_average_precision(pred_array)
        	mAPs.append(m_ap)
    else:
    	for i in rankings:
        	pred_array = extract_ranking(distance_matrix, i, ids_aerial, ids_ground, 'aerial')
        	m_ap = mean_average_precision(pred_array)
        	mAPs.append(m_ap)
    out_file.write(str(mAPs))

def generate_retrieval_list(dist_array, topK, img_ids, img_ids2, output_file, query):
    
    if (query == 'ground'):
        for i in range(dist_array.shape[0]):
            position = np.argpartition(dist_array[i, :], topK)
            indexes = position[:topK]
            output_file.write(str(img_ids2[i]).replace('b\'', '') + ' : ')
            my_dict = {}
            for j in range(topK):
                my_dict[indexes[j]] = dist_array[i, indexes[j]]
            sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
            counter = 0
            for k, v in sorted_dict.items():
                if (counter != topK-1):
                    output_file.write(str(img_ids[k]).replace('b\'', '') + ',')
                else:
                    output_file.write(str(img_ids[k]).replace('b\'', ''))
                counter += 1
            output_file.write('\n')
    else:
        for i in range(dist_array.shape[1]):
            position = np.argpartition(dist_array[:, i], topK)
            indexes = position[:topK]
            output_file.write(str(img_ids[i]).replace('b\'', '') + ' : ')
            my_dict = {}
            for j in range(topK):
                my_dict[indexes[j]] = dist_array[indexes[j], i]
            sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
            counter = 0
            for k, v in sorted_dict.items():
                if (counter != topK-1):
                    output_file.write(str(img_ids2[k]).replace('b\'', '') + ',')
                else:
                    output_file.write(str(img_ids2[k]).replace('b\'', ''))
                counter += 1
            output_file.write('\n')
