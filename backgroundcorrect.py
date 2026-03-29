#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Adapted from doi: 10.1093/jncics/pkac080 
https://github.com/pfnet-research/head_model'''
import argparse
import os
import numpy as np
import pandas as pd
import math
import glob
import os
import six


class IDConverter(object):

    def __init__(self):
        self._id2name = []
        self._name2id = {}

    def to_name(self, id_):
        if len(self._id2name) <= id_:
            raise ValueError('Invalid index')
        return self._id2name[id_]

    def to_id(self, name):
        if name not in self._name2id:
            self._name2id[name] = len(self._name2id)
            self._id2name.append(name)
        return self._name2id[name]

    @property
    def id2name(self):
        return self._id2name

    @property
    def name2id(self):
        return self._name2id

    @property
    def unique_num(self):
        return len(self._id2name)

class UniqueNameGenerator(object):

    def __init__(self):
        self.counts = {}

    def make_unique(self, name):
        name = str(name)
        if name in self.counts:
            cnt = self.counts[name]
            self.counts[name] += 1
            return name + '_' + str(cnt)
        else:
            self.counts[name] = 1
            return name

def convert_feature_vectors(fvs, col_num,
                            forbid_empty_entry=True):
    """Converts feature vectors to numpy ndarray.

    Args:
        fvs (list of dictionaries): list of feature vectors
            fvs[sample_id][feature_id] = value
        col_num: Feature dimension of converted feature vectors.
        forbid_empty_entry (bool): If ``True``, the resulting
        array should not have ``numpy.nan``

    Returns:
        numpy.ndarray: Shape is (row_num, col_num)
            where row_num = len(fvs).
            ret[sample_id][feature_id] = value
            If some feature value is not existent in `fvs`,
            the corresponding element in the output is
            filled with ``numpy.nan``.

    """

    row_num = len(fvs)
    ret = np.full((row_num, col_num), np.nan, dtype=np.float32)
    for i, fv in enumerate(fvs):
        for k, v in six.iteritems(fv):
            if not isinstance(k, int):
                raise ValueError('keys of dictionaries in '
                                 'fvs must be integer.')
            if k >= col_num:
                raise ValueError('keys of dictionaries in '
                                 'fvs must be smaller than col_num.')
            ret[i][k] = v
    if forbid_empty_entry and np.isnan(ret).any():
        raise ValueError('Non-existent feature found')
    return ret


def _create_feature_vector(df, feature_id_converter):
    fv = {}
    mirna_names = df['G_Name']
    values = df['635nm']

    gen = UniqueNameGenerator()
    for name, value in six.moves.zip(mirna_names, values):
        name = gen.make_unique(name)
        fid = feature_id_converter.to_id(name)
        if fid in fv:
            print('%s is duplicated' % name)
        fv[fid] = value
    return fv


def _is_older_than_v21(df):
    return 'hsa-miR-9500' in list(df['G_Name'])


def fetch_dir(dir_name, feature_id_converter, filters=()):
    feature_vectors = []
    instance_names = []

    for fname in glob.glob('%s/*.txt' % dir_name):
        if not (os.path.isfile(fname)):
            print(str(fname) + ' is not a measurement file.')
            continue

        try:
            df = pd.read_csv(fname, header=6, delimiter='\t')
        except Exception:
            print('Failed to parse: ' + str(fname))
            continue

        if not _is_older_than_v21(df):
            continue

        for _filter in filters:
            df = _filter(df)

        fv = _create_feature_vector(df, feature_id_converter)
        feature_vectors.append(fv)
        instance_names.append(fname)

    return feature_vectors, instance_names


def assert_equal(a, b):
    assert a == b, '%d != %d' % (a, b)

def backgroundcorrect(df):
    df_column_normalized = df.apply(samplewise_preprocess, axis=1)
    return df_column_normalized

def use_hsa_mirna_only(df):
    return df.filter(regex=r"^hsa-") #df[df['G_Name'].str.startswith('hsa-')]


def compute_control_statistics(col):
    neg_con_indices = col.index.str.startswith('Negative Control 2')
    num_neg_con = sum(neg_con_indices)
    if num_neg_con < 3:
        # At least two negative controls are removed.
        # So, we need at least three controls to compute
        # mean and std.
        msg = 'col must have at least three negative control.'
        raise ValueError(msg)

    cutoff_num = max(num_neg_con // 20, 1)
    neg_con = col[neg_con_indices].sort_values()
    neg_con = neg_con[cutoff_num:-cutoff_num]
    mean = neg_con.mean()
    std = neg_con.std(ddof=0)
    return mean, std


def samplewise_preprocess(col):
    mean, std = compute_control_statistics(col)
    col = col.loc[col.index.map(lambda x: x.startswith('hsa-'))]

    col -= mean
    present = col > 2 * std
    col[present] = col[present].apply(math.log2)

    # In the original paper [1], they used the commented processing
    # instead of the uncommented one.
    # Following Kawauchi-san@Toray,
    # we change the default values filling to the absent probes.
    # [1] https://onlinelibrary.wiley.com/doi/full/10.1111/cas.12880
    col[~present] = 0.1
    col[col < 0] = 0.1
    # min_val = col[present].min()
    # col[~present] = min_val - 0.1
    # col[col < 0] = min_val - 0.1
    return col




def fetch(root_dir, dataset, filters):
    feature_vectors = []
    instance_names = []
    labels = []

    feature_id_converter = IDConverter()
    label_id_converter = IDConverter()

    for dir_name, label_name in dir_names:
        dir_name = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_name):
            print(dir_name + ' not found.')
            continue

        print('Processing ' + dir_name)

        feature_vectors_, instance_names_ = fetch_dir(
            dir_name, feature_id_converter, filters)
        lid = label_id_converter.to_id(label_name)
        labels_ = [lid] * len(feature_vectors_)

        assert_equal(len(feature_vectors_), len(instance_names_))
        assert_equal(len(feature_vectors_), len(labels_))

        feature_vectors += feature_vectors_
        instance_names += instance_names_
        labels += labels_

    # Convert data
    feature_names = feature_id_converter.id2name
    label_names = label_id_converter.id2name
    feature_vectors = convert_feature_vectors(
        feature_vectors, feature_id_converter.unique_num, True)
    instance_names = np.array(instance_names)
    labels = np.array(labels)

    return feature_names, label_names, feature_vectors, instance_names, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessor')
    parser.add_argument('--in-dir', '-i', type=str, default='input')

    args = parser.parse_args()


    dir_names = [('BC/benign_breast_cancer/v21', 'BC_benign'),
                ('BC/breast_cancer/v21', 'BC'),
                ('BL/bladder_cancer/v21', 'BL'),
                ('BT/biliary_tract_cancer/v21', 'BT'),
                ('CC/colorectal_cancer/v21', 'CC'),
                ('EC/esophageal_cancer/v21', 'EC'),
                ('GC/gastric_cancer/v21', 'GC'),
                ('GL/benign_skull/v21', 'GL_benign'),
                ('GL/glioma/v21', 'GL'),
                ('HC/v21', 'HC'),
                ('LK/v21', 'LK'),
                ('OV/benign_ovarian_cancer/v21', 'OV_benign'),
                ('OV/ovarian_cancer/v21', 'OV'),
                ('PC/v21', 'PC'),
                ('PR/benign_prostate_cancer/v21', 'PR_benign'),
                ('PR/prostate_cancer/v21', 'PR'),
                ('SA/v21/benign_primary', 'SA_benign'),
                ('SA/v21/malignant_primary', 'SA'),
                ('VOL/NCGG/v21', 'VOL'),
                ('VOL/minoru/XA/v21', 'VOL'),
                ('VOL/minoru/XB/v21', 'VOL')]


    d = fetch(args.in_dir, dir_names, [])
    # print('d: ',d)
    feature_names = d[0]
    # print('feature names: ',type(feature_names),len(feature_names), feature_names[0])
    label_names = d[1]
    feature_vectors = d[2]
    # print('feature vectors: ',type(feature_vectors), feature_vectors.shape, feature_vectors[0])
    instance_names = d[3]
    # print('instance_names: ',type(instance_names),instance_names.shape, instance_names[0])
    labels = d[4]
    # print('types: ',type(feature_names),type(label_names),type(feature_vectors),type(labels))

    ids = np.array([os.path.splitext(os.path.basename(path))[0] for path in instance_names])
    # print('ids: ', ids[0])

    import os

    mapping_file = "rename_mapping.txt"
    file_mapping_dict = {}

    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            file_mapping_dict = {
                line.split(':')[0].strip(): line.split(':')[1].strip() 
                for line in f if ':' in line
            }
    v_get = np.vectorize(file_mapping_dict.get)
    ids = v_get(ids)

    df = pd.DataFrame(feature_vectors,columns=feature_names).set_index(ids)
    df = backgroundcorrect(df)
    df = use_hsa_mirna_only(df)
    # print('df shape: ',df.shape)
    df.insert(0, 'label', labels)
    df.to_csv('data/backgroundcorrected_idx.csv')
    print('background correction done')

