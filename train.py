#!/usr/bin/env python3
import os
import sys
import time
import argparse
import tempfile
import shutil
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CASTLE_BACKEND'] = 'pytorch'
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import uuid

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.utils import register_keras_serializable

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Environment uses GPU :)")
else:
    print("WARNING - Environment is not using GPU, only CPU!")

def setup_gpu_worker():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except ValueError:
                    pass 
                
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
                    )
                except ValueError:
                    pass 
        except RuntimeError as e:
            print(f"Failed to set GPU memory limit: {e}")

def get_encoded_labels(labels):
    maxval = np.max(labels) + 1
    return np.eye(maxval)[labels]


def warm_start_prototypes(model_wrapper, training_set):
    extractor = model_wrapper.extractor
    mahalanobis_layer = None
    for layer in model_wrapper.model.layers:
        if 'mahalanobis' in layer.name.lower() or hasattr(layer, 'centers'):
            mahalanobis_layer = layer
            break
            
    if mahalanobis_layer is None:
        print('mahalanobis_layer is None')
        return

    all_embeddings = []
    all_labels = []
    
    for x_batch, y_batch in training_set:
        feats = extractor(x_batch, training=False)
        all_embeddings.append(feats.numpy())
        
        if len(y_batch.shape) > 1:
            lbls = np.argmax(y_batch, axis=1)
        else:
            lbls = y_batch
        all_labels.append(lbls)
        
    X_train_emb = np.vstack(all_embeddings)
    y_train_lbl = np.concatenate(all_labels)
    
    num_classes = mahalanobis_layer.num_classes
    embedding_dim = mahalanobis_layer.embedding_dim
    
    new_centers = np.zeros((num_classes, embedding_dim), dtype=np.float32)
    
    # claculate the means for all classes
    for c in range(num_classes):
        class_samples = X_train_emb[y_train_lbl == c]
        
        if len(class_samples) > 0:
            mean_vec = np.mean(class_samples, axis=0)
            new_centers[c] = mean_vec
        else:
            print(f"  [Warning] Class {c} has no samples in training set!")

    # inject weights into model
    current_sigma = mahalanobis_layer.inverse_sigma.numpy()
    mahalanobis_layer.set_weights([new_centers, current_sigma])
    print('prototypes initialized to training data centers')

@register_keras_serializable()
class MCGaussianNoise(layers.GaussianNoise):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class MahalanobisOutput(layers.Layer):
    def __init__(self, num_classes, embedding_dim, **kwargs):
        super(MahalanobisOutput, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.centers = self.add_weight(
            name='prototypes',
            shape=(self.num_classes, self.embedding_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # initialize inverse covariance to 1.0 (Euclidean assumption start)
        self.inverse_sigma = self.add_weight(
            name='inverse_sigma',
            shape=(self.num_classes, self.embedding_dim),
            initializer='ones',
            trainable=True
        )
        super(MahalanobisOutput, self).build(input_shape)

    def call(self, inputs):
        # inputs: (Batch, Dim)
        x = tf.expand_dims(inputs, 1)          # (Batch, 1, Dim)
        mu = tf.expand_dims(self.centers, 0)   # (1, Classes, Dim)
        sigma = tf.expand_dims(self.inverse_sigma, 0) # (1, Classes, Dim)
        
        squared_diff = tf.square(x - mu)
        
        sigma_clean = tf.math.softplus(sigma) + 1e-5
        
        weighted_dist = tf.multiply(squared_diff, sigma_clean)
        mahalanobis_sq = tf.reduce_sum(weighted_dist, axis=2)
        
        scaler = tf.math.rsqrt(tf.cast(self.embedding_dim, tf.float32))
        
        return tf.nn.softmax(-mahalanobis_sq * scaler)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
        })
        return config
    


class ClassificationModel:
    def __init__(self, args, img_w, training_set, validation_set, test_set, classnum, loaded, fold=0):
        tf.keras.utils.set_random_seed(args.seed)
        
        self.dropout = getattr(args, 'dropout', 0.3)
        self.lr = args.lr
        self.comb = args.comb
        self.image_width = img_w
        self.model = None
        self.loaded = loaded
        self.classnum = classnum
        self.channel = 1
        self.dense0 = args.dense0
        self.dense1 = args.dense1
        self.dense2 = args.dense2
        self.dense3 = args.dense3
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:6] # random 6-character string
        self.i = f'{args.data}_{timestamp}_{unique_id}_fold{fold}_{args.baseline}'
        self.fold = fold
        self.epochs = args.epochs
        self.batch_size = args.batch
        self.baseline = args.baseline
        self.dataname = args.data
        print(f'MODEL USED: {self.baseline}')
        if not loaded:
            self.model = self.build_model(training_set, validation_set)
        else:
            self.model = tf.keras.models.load_model(f'model/{self.i}_classifier_trained.keras')

    def define_callbacks(self):
        import os 
        from tensorflow.keras.callbacks import EarlyStopping
        

        early_stopping = EarlyStopping(patience=8, min_delta=0.001, restore_best_weights=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,       
            patience=8,       
            min_lr=1e-6,
            verbose=0
        )

        return [early_stopping, reduce_lr]

    def predict(self, dataset):
        return self.model.predict(dataset)

    def build_model(self, training_set, validation_set):
        from sklearn.utils import class_weight
        
        data_input = Input(shape=(self.image_width, self.channel))
        data_features = layers.Flatten()(data_input)
        data_features = MCGaussianNoise(0.2)(data_features)

        data_features = layers.Dense(self.dense0, activation='relu')(data_features)
        data_features = layers.BatchNormalization()(data_features)
        data_features = MCDropout(self.dropout)(data_features)

        data_features = layers.Dense(self.dense1, activation='relu')(data_features)
        data_features = layers.BatchNormalization()(data_features)
        data_features = MCDropout(self.dropout)(data_features)
        
        data_features = layers.Dense(self.dense2, activation='relu')(data_features)
        data_features = layers.BatchNormalization()(data_features)
        data_features = MCDropout(self.dropout)(data_features)

        if self.baseline == 'MLP':
            print('MLP LAYER')
            data_features = layers.Dense(self.dense3, activation='relu', name='embedding')(data_features)
            self.extractor = Model(inputs=data_input, outputs=data_features)

            cancer = layers.Dense(self.classnum, activation='softmax')(data_features)
            
        else:
            print('USE MAHALANOBIS')
            data_features = layers.Dense(self.dense3, activation=None, name='embedding')(data_features)
            self.extractor = Model(inputs=data_input, outputs=data_features)

            cancer = MahalanobisOutput(num_classes=self.classnum, embedding_dim=self.dense3)(data_features)

        model = Model(inputs=[data_input], outputs=[cancer])    

        y_weight = np.concatenate([np.argmax(y, axis=1) for x, y in training_set])
        unique_classes = np.unique(y_weight)
        class_weights_array = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_weight)
        class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights_array)}

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), 
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
            metrics=[
                tf.keras.metrics.F1Score(average='macro'),
                'accuracy', 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall()
            ],
            run_eagerly=False
        )

        self.model = model
        if self.baseline == 'OURS':
            warm_start_prototypes(self, training_set)
        
        model.fit(
            training_set,          
            epochs=self.epochs, 
            shuffle=True,          
            validation_data=validation_set,
            callbacks=self.define_callbacks(),
            class_weight=class_weight_dict,
            verbose=0
        )
        
        return model



def predict_with_mc_sampling(model, dataset, n_aug=20):
    all_passes = []
    
    for i in range(n_aug):
        pass_predictions = []
        for batch_x, _ in dataset:
            pred = model(batch_x, training=False) 
            pass_predictions.append(pred.numpy())
            
        all_passes.append(np.vstack(pass_predictions))
            
    all_passes_np = np.array(all_passes)
    variance = np.var(all_passes_np, axis=0)
    # print(f"\n  [DEBUG] TTA Mean Variance: {np.mean(variance):.6f}")
    expected_p = np.mean(all_passes_np, axis=0)
    # predictive entropy 
    epsilon = 1e-12 
    predictive_entropy = -np.sum(expected_p * np.log(expected_p + epsilon), axis=1)
    
    # print(f"\n  [DEBUG] Mean Predictive Entropy: {np.mean(predictive_entropy):.6f}")

    return expected_p, predictive_entropy

def select_top_features(X_train, y_train, k):
    model_tree = RandomForestClassifier(
        n_estimators=150, 
        max_depth=None, 
        min_samples_split=5, 
        min_samples_leaf=1, 
        class_weight='balanced_subsample',
        n_jobs=-1, 
        random_state=42
    )
    sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select=k, step=0.05, verbose=0)
    sel_rfe_tree.fit(X_train, y_train)
    
    selected_indices = sel_rfe_tree.get_support(indices=True)
    sorted_indices = sorted(selected_indices)
    return sorted_indices

def run_baseline(args, X_train, y_train, X_test, y_test, best_params=None, ood_datasets=None, outer_fold=0):
    setup_gpu_worker()
    if best_params is None: best_params = {}
    print(f'RUN BASELINE: {args.baseline}')
    
    model = None
    y_pred = None
    probs = None
    f1 = 0.0
    ood_probs = {}
    ood_entropies = {}
    entropy = None
    selected_features = None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if args.baseline == 'LR':
        from sklearn.linear_model import LogisticRegression
        from tensorflow.keras.utils import to_categorical
        
        n_features = int(best_params.get("n_features", 100))
        c_val = best_params.get("C", 1.0)
        cw_val = best_params.get("class_weight", None)
        if cw_val == 'None': cw_val = None
        
        selected_features = select_top_features(X_train, y_train, k=n_features) 
        X_train_sel, X_test_sel = X_train[:, selected_features], X_test[:, selected_features]

        from imblearn.over_sampling import BorderlineSMOTE
        sm = BorderlineSMOTE(random_state=args.seed, k_neighbors=3) 
        X_train_sel, y_train = sm.fit_resample(X_train_sel, y_train)

        model = LogisticRegression(random_state=42, class_weight=cw_val, max_iter=1000, C=c_val).fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        probs = model.predict_proba(X_test_sel)
        
        num_classes = np.max(y_test) + 1  
        metric = tf.keras.metrics.F1Score(average='macro')
        metric.update_state(to_categorical(y_test, num_classes), to_categorical(y_pred, num_classes))
        f1 = metric.result()
        
        if ood_datasets:
            for name, data in ood_datasets.items():
                ood_probs[name] = model.predict_proba(data[:, selected_features])
                
    elif args.baseline == 'RF':
        n_features = int(best_params.get("n_features", 100))
        n_est = int(best_params.get("n_estimators", 200))
        max_depth = best_params.get("max_depth", None)
        min_samples_split = int(best_params.get("min_samples_split", 2))
        
        selected_features = select_top_features(X_train, y_train, k=n_features) 
        X_train_sel, X_test_sel = X_train[:, selected_features], X_test[:, selected_features]

        from imblearn.over_sampling import BorderlineSMOTE
        sm = BorderlineSMOTE(random_state=args.seed, k_neighbors=3) 
        X_train_sel, y_train = sm.fit_resample(X_train_sel, y_train)

        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth, min_samples_split=min_samples_split,
            class_weight='balanced', random_state=args.seed, n_jobs=-1 
        )
        model.fit(X_train_sel, y_train)

        probs = model.predict_proba(X_test_sel)
        y_pred = model.predict(X_test_sel)

        if ood_datasets:
            for name, data in ood_datasets.items():
                ood_probs[name] = model.predict_proba(data[:, selected_features])

    elif args.baseline == 'MLP':
        from tensorflow.keras.utils import to_categorical
        
        args.filter_k = int(best_params.get("n_features", 150))
        args.dense0 = int(best_params.get("dense0", 128))
        args.dense1 = int(best_params.get("dense1", 64))
        args.dense2 = int(best_params.get("dense2", 32))
        args.dense3 = int(best_params.get("dense3", 16))
        args.dropout = float(best_params.get("dropout", 0.3))
        args.lr = float(best_params.get("lr", 0.001))
        
        selected_features = select_top_features(X_train, y_train, k=args.filter_k) 
        X_train_sel, X_test_sel = X_train[:, selected_features], X_test[:, selected_features]

        from imblearn.over_sampling import BorderlineSMOTE
        sm = BorderlineSMOTE(random_state=args.seed, k_neighbors=3) 
        X_train_sel, y_train = sm.fit_resample(X_train_sel, y_train)

        num_classes_global = int(np.max(y_train) + 1)
        y_train_encoded = to_categorical(y_train, num_classes=num_classes_global)
        y_test_encoded = to_categorical(y_test, num_classes=num_classes_global)

        BATCH_SIZE = args.batch
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_sel, y_train_encoded)).shuffle(len(X_train_sel)).batch(BATCH_SIZE)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_sel, y_test_encoded)).batch(BATCH_SIZE)

        model = ClassificationModel(
            args, X_train_sel.shape[1], train_ds, test_ds, test_ds,
            classnum=num_classes_global, loaded=False, fold=outer_fold
        )
       
        probs_no_tta = model.model.predict(test_ds, verbose=0)
        y_pred_no_tta = np.argmax(probs_no_tta, axis=1) if probs_no_tta.ndim > 1 else (probs_no_tta > 0.5).astype(int)
        y_true_eval = y_test if np.ndim(y_test) == 1 else np.argmax(y_test, axis=1)

        probs, entropy = predict_with_mc_sampling(model.model, test_ds, n_aug=20)
        y_pred = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > 0.5).astype(int)
        
        if ood_datasets:
            for name, data in ood_datasets.items():
                data_sel = data[:, selected_features]
                data_ds = tf.data.Dataset.from_tensor_slices((data_sel, np.zeros(len(data_sel)))).batch(BATCH_SIZE)
                ood_probs[name], entropy_ood = predict_with_mc_sampling(model.model, data_ds, n_aug=20)
                ood_entropies[name] = entropy_ood


    elif args.baseline == 'XGB':
        import xgboost as xgb
        from tensorflow.keras.utils import to_categorical
        
        n_features = int(best_params.get("n_features", 100))
        max_depth = int(best_params.get("max_depth", 6))
        lr = best_params.get("learning_rate", 0.3)
        n_est = int(best_params.get("n_estimators", 100))
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        selected_features = select_top_features(X_train_split, y_train_split, k=n_features) 
        
        X_train_sel = X_train_split[:, selected_features]
        X_test_sel = X_test[:, selected_features]
        X_val_sel = X_val[:, selected_features]

        from imblearn.over_sampling import BorderlineSMOTE
        sm = BorderlineSMOTE(random_state=args.seed, k_neighbors=3) 
        X_train_sel, y_train_split = sm.fit_resample(X_train_sel, y_train_split)

        model = xgb.XGBClassifier(tree_method="hist", device="cuda", early_stopping_rounds=3, max_depth=max_depth, learning_rate=lr, n_estimators=n_est)
        model.fit(X_train_sel, y_train_split, eval_set=[(X_val_sel, y_val)], verbose=False)
        
        y_pred = model.predict(X_test_sel)
        probs = model.predict_proba(X_test_sel)
        
        num_classes = np.max(y_train) + 1  
        metric = tf.keras.metrics.F1Score(average='macro')
        metric.update_state(to_categorical(y_test, num_classes), to_categorical(y_pred, num_classes))
        f1 = metric.result()
        
        if ood_datasets:
            for name, data in ood_datasets.items():
                ood_probs[name] = model.predict_proba(data[:, selected_features])

    elif args.baseline == '1D':
        from tensorflow.keras.models import Sequential
        
        f_size = int(best_params.get("filters", 32))
        k_size = int(best_params.get("kernel_size", 3))
        d_units = int(best_params.get("dense_units", 128))
        lr = best_params.get("lr", 0.001)
    
        
        from imblearn.over_sampling import BorderlineSMOTE
        sm = BorderlineSMOTE(random_state=args.seed, k_neighbors=3) 
        X_train, y_train = sm.fit_resample(X_train, y_train)

        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], 1)
        
        y_train_enc = get_encoded_labels(y_train)
        y_test_enc = get_encoded_labels(y_test)
        num_classes = len(y_train_enc[0])

        model = Sequential()
        model.add(layers.Conv2D(f_size, kernel_size=(1, k_size), strides=(1, 1), input_shape=(1, X_train.shape[2], 1)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(1, 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(d_units, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['categorical_accuracy', tf.keras.metrics.F1Score(average='macro')])
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=3, verbose=0)]
        model.fit(X_train, y_train_enc, batch_size=32, epochs=200, verbose=0, callbacks=callbacks, validation_split=0.1, shuffle=True)
        
        probs = model.predict(X_test)
        y_pred = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > 0.5).astype(int)
        
        if ood_datasets:
            for name, data in ood_datasets.items():
                data_reshaped = data.reshape(data.shape[0], 1, data.shape[1], 1)
                ood_probs[name] = model.predict(data_reshaped, verbose=0)

    elif args.baseline == 'HEAD':
        try:
            np.sctypes
        except AttributeError:
            np.sctypes = {
                'int': [np.int8, np.int16, np.int32, np.int64],
                'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                'float': [np.float16, np.float32, np.float64],
                'complex': [np.complex64, np.complex128],
                'others': [bool, object, bytes, str, np.void]
            }
        import head.project as project 
        import sklearn.impute
        import sklearn.neighbors
        import sklearn.svm
        import xgboost as xgb
        import head.features.knn 
        import head.models.chainer_file 
        from head.data import io as head_io 
        
        n_features = int(best_params.get("n_features", 100)) 
        l2_xgb_depth = int(best_params.get("l2_xgb_depth", 5))
        l2_xgb_lr = float(best_params.get("l2_xgb_lr", 0.1))

        unique_id = f"{int(time.time())}_{np.random.randint(0, 10000)}"
        work_dir = f"./temp_ensemble_work/{unique_id}"
        os.makedirs(work_dir, exist_ok=True)

        num_classes = len(np.unique(y_train))
        label_file_path = os.path.join(work_dir, "label_names.txt")
        with open(label_file_path, "w") as f:
            for i in range(num_classes): f.write(f"{i}\n")

        importancescores = []
        for rs in range(5):
            clf = xgb.XGBClassifier(random_state=rs, n_jobs=1).fit(X_train, y_train)
            importancescores.append(np.array(clf.feature_importances_))
        importancescores = np.mean(np.array(importancescores), axis=0)

        top_indices = np.argsort(importancescores)[::-1][:n_features]
        selected_features = top_indices

        X_train = X_train[:, top_indices]
        X_test = X_test[:, top_indices]

        X_eval_list = [X_test] 
        lengths = [len(X_test)]
        dataset_names = []
        
        if ood_datasets:
            for name, data in ood_datasets.items():
                data_sel = data[:, top_indices] 
                X_eval_list.append(data_sel)
                lengths.append(len(data_sel))
                dataset_names.append(name)
                
        X_eval_combined = np.vstack(X_eval_list)

        pl = project.pipeline(
            working_dir=work_dir, seed=42, n_folds=3, 
            X_train_raw=X_train, X_test_raw=X_eval_combined, 
            y_train_raw=y_train, label_name_path=label_file_path 
        )

        pl.transform(output_name='imputed', input_names=['raw'], unsupervised=True, estimator=sklearn.impute.SimpleImputer(strategy='median'))
        pl.transform(output_name='normalized', input_names=['imputed'], unsupervised=True, estimator=sklearn.preprocessing.Normalizer())
        pl.transform(output_name='tsne', input_names=['imputed'], unsupervised=True, estimator=sklearn.manifold.TSNE(3))

        pl.predict_proba(output_name='lev1_random-forest', input_names=['imputed'], validate_size=2, estimator=sklearn.ensemble.RandomForestClassifier(n_jobs=4))
        pl.predict_proba(output_name='lev1_logistic-regression', input_names=['imputed'], validate_size=2, estimator=sklearn.linear_model.LogisticRegression())
        pl.predict_proba(output_name='lev1_extra-tree', input_names=['imputed'], validate_size=2, estimator=sklearn.ensemble.ExtraTreesClassifier(n_jobs=4))
        pl.decision_function(output_name='lev1_linear-svc', input_names=['imputed'], validate_size=2, estimator=sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=50), version=1)
        
        KS = [2, 4, 8, 16, 32, 64, 128, 256]
        for k in KS: pl.predict_proba(output_name='lev1_knn_k={}'.format(k), input_names=['imputed'], validate_size=2, estimator=sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, n_jobs=4))
        pl.transform(output_name='lev1_knn_distances', input_names=['imputed'], validate_size=2, estimator=head.features.knn.KNNDistanceFeature(ks=[1, 2, 4]))
        pl.transform(output_name='lev1_knn_distances_tsne', input_names=['tsne'], validate_size=2, estimator=head.features.knn.KNNDistanceFeature(ks=[1]))
        
        pl.predict_proba(output_name='lev1_xgboost', input_names=['raw', 'tsne'], validate_size=2, estimator=xgb.XGBClassifier(objective='multi:softproba', learning_rate=0.05, max_depth=5, n_estimators=1000, nthread=4, subsample=0.5, colsample_bytree=1.0))
        pl.predict_proba(output_name='lev1_mlp3', input_names=['imputed', 'tsne'], validate_size=2, estimator=head.models.chainer_file.ChainerClassifier(head.models.chainer_file.MLP3, gpu=args.gpu, n_epoch=100, n_out=len(pl.label_names)))
        pl.predict_proba(output_name='lev1_mlp4', input_names=['imputed', 'tsne'], validate_size=2, estimator=head.models.chainer_file.ChainerClassifier(head.models.chainer_file.MLP4, gpu=args.gpu, n_epoch=200, n_out=len(pl.label_names)))

        LEVEL1_PREDICTIONS = ['lev1_random-forest', 'lev1_logistic-regression', 'lev1_extra-tree', 'lev1_linear-svc', 'lev1_xgboost', 'lev1_mlp3', 'lev1_mlp4'] + ['lev1_knn_k={}'.format(k) for k in KS]
        LEVEL1_FEATURES = ['tsne', 'lev1_knn_distances', 'lev1_knn_distances_tsne']

        pl.predict_proba(output_name='lev2_logistic-regression', input_names=LEVEL1_PREDICTIONS, validate_size=1, estimator=sklearn.linear_model.LogisticRegression(), version=1)
        pl.predict_proba(output_name='lev2_xgboost2', input_names=(LEVEL1_PREDICTIONS + LEVEL1_FEATURES), validate_size=1, estimator=xgb.XGBClassifier(objective='multi:softmax', learning_rate=l2_xgb_lr, max_depth=l2_xgb_depth, n_estimators=1000, nthread=4, subsample=0.9, colsample_bytree=0.7), version=1)
        pl.predict_proba(output_name='lev2_mlp4', input_names=(['imputed'] + LEVEL1_PREDICTIONS + LEVEL1_FEATURES), validate_size=1, estimator=head.models.chainer_file.ChainerClassifier(head.models.chainer_file.MLP4, gpu=args.gpu, n_epoch=200, n_out=len(pl.label_names)), version=1)
        pl.predict(output_name='lev2_linear-svc', input_names=['imputed'], validate_size=1, estimator=sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=50)) 
        
        import glob
        try:
            search_pattern = f"{work_dir}/**/lev2_xgboost2/**/test.h5"
            found_files = glob.glob(search_pattern, recursive=True)
            if not found_files:
                search_pattern_npy = f"{work_dir}/**/lev2_xgboost2/**/test.npy"
                found_files = glob.glob(search_pattern_npy, recursive=True)
            if not found_files: raise FileNotFoundError(f"Could not find valid output file in {work_dir}")
                
            target_file = found_files[0]
            probs_combined, info = head_io.load(target_file)
            probs = probs_combined[:lengths[0]]
            y_pred = np.argmax(probs, axis=1)
            
            current_idx = lengths[0]
            if ood_datasets:
                for i, name in enumerate(dataset_names):
                    size = lengths[i+1]
                    ood_probs[name] = probs_combined[current_idx : current_idx + size]
                    current_idx += size
        except Exception as e:
            raise e
        shutil.rmtree(work_dir)

    elif args.baseline == 'TABNET':
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch
        
        n_features = int(best_params.get("n_features", 100))
        n_d = int(best_params.get("n_d", 16)) 
        n_a = int(best_params.get("n_a", 16)) 
        lr = best_params.get("lr", 0.02)
        
        selected_features = select_top_features(X_train, y_train, k=n_features) 
        X_train_sel = X_train[:, selected_features]
        X_test_sel = X_test[:, selected_features]

        from imblearn.over_sampling import BorderlineSMOTE
        sm = BorderlineSMOTE(random_state=args.seed, k_neighbors=3) 
        X_train_sel, y_train = sm.fit_resample(X_train_sel, y_train)

        model = TabNetClassifier(
            n_d=n_d, n_a=n_a, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=lr),
            scheduler_params={"step_size": 10, "gamma": 0.9}, scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax', verbose=0
        )

        model.fit(
            X_train=X_train_sel, y_train=y_train, eval_set=[(X_train_sel, y_train), (X_test_sel, y_test)],
            eval_name=['train', 'val'], eval_metric=['balanced_accuracy'], max_epochs=args.epochs,
            patience=15, batch_size=args.batch, virtual_batch_size=args.batch // 2, drop_last=False
        )

        probs = model.predict_proba(X_test_sel)
        y_pred = model.predict(X_test_sel)

        if ood_datasets:
            for name, data in ood_datasets.items():
                data_sel = data[:, selected_features]
                ood_probs[name] = model.predict_proba(data_sel)
                
    elif args.baseline == 'OURS':
        from tensorflow.keras.utils import to_categorical
        
        args.filter_k = int(best_params.get("n_features", 150))
        args.dense0 = int(best_params.get("dense0", 128))
        args.dense1 = int(best_params.get("dense1", 64))
        args.dense2 = int(best_params.get("dense2", 32))
        args.dense3 = int(best_params.get("dense3", 16))
        args.dropout = float(best_params.get("dropout", 0.3))
        args.lr = float(best_params.get("lr", 0.001))
        
        selected_features = select_top_features(X_train, y_train, k=args.filter_k) 
        X_train, X_test = X_train[:, selected_features], X_test[:, selected_features]

        from imblearn.over_sampling import BorderlineSMOTE
        sm = BorderlineSMOTE(random_state=args.seed, k_neighbors=3) 
        X_train, y_train = sm.fit_resample(X_train, y_train)
        
        num_classes_global = int(np.max(y_train) + 1)
        y_train = to_categorical(y_train, num_classes=num_classes_global)
        y_test_cat = to_categorical(y_test, num_classes=num_classes_global)
        
        BATCH_SIZE = args.batch
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE, drop_remainder=True)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test_cat)).batch(BATCH_SIZE)
        
        model = ClassificationModel(
            args, X_train.shape[1], train_ds, test_ds, test_ds,
            classnum=num_classes_global, loaded=False, fold=outer_fold
        )
        
        probs_no_tta = model.model.predict(test_ds, verbose=0)
        y_pred_no_tta = np.argmax(probs_no_tta, axis=1) if probs_no_tta.ndim > 1 else (probs_no_tta > 0.5).astype(int)
        y_true_eval = y_test if np.ndim(y_test) == 1 else np.argmax(y_test, axis=1)
        
        
        probs, entropy = predict_with_mc_sampling(model.model, test_ds, n_aug=20)
        y_pred = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > 0.5).astype(int)
        
        if ood_datasets:
            for name, data in ood_datasets.items():
                data_sel = data[:, selected_features]
                data_ds = tf.data.Dataset.from_tensor_slices((data_sel, np.zeros(len(data_sel)))).batch(BATCH_SIZE)
                ood_probs[name], entropy_ood = predict_with_mc_sampling(model.model, data_ds, n_aug=20)
                ood_entropies[name] = entropy_ood

    y_true_indices = y_test if np.ndim(y_test) == 1 else np.argmax(y_test, axis=1)
    f1 = f1_score(y_true_indices, y_pred, average='macro')
    return y_pred, f1, probs, model, ood_probs, selected_features, entropy, ood_entropies

def ray_trainable(config, X_train, y_train, X_val, y_val, args):
    import gc
    try:
        _, f1_score_val, _, _, _, _, _, _ = run_baseline(
            args, X_train, y_train, X_val, y_val, best_params=config
        )
        tune.report({"f1": f1_score_val})
        tf.keras.backend.clear_session()
        gc.collect()
    except Exception as e:
        print(f"Trial Failed: {e}")
        raise e

def main(args):
    if args.comb == 2:
        print('RUNNING TASK 1 - Tissue-of-Origin Task (14 Classes)')
    if args.comb == 0:
        print('RUNNING TASK 2 - Cancer Task (19 Classes)')
    
    file_path = 'data/sorted.csv' 
    X = pd.read_csv(file_path)[:-2]
    indices = X.index.tolist()
    all_features = X.columns.tolist()[1:]
    y = X['label']
    X = X.drop(columns=['label']).values

    if args.comb == 2:
        mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 8, 11: 9, 12: 9, 13: 10, 14: 11, 15: 11, 16: 12, 17: 12, 18: 13}
        y = y.map(mapping)
        
    y = y.values

    if ray.is_initialized(): ray.shutdown()
    
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        num_physical_gpus = len(cuda_visible.split(","))
    else:
        num_physical_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 2))
    
    needs_gpu = args.baseline in ['OURS', '1D', 'XGB', 'TABNET', 'MLP', 'HEAD']
    use_gpu = needs_gpu and num_physical_gpus > 0
    
    ray_gpus = num_physical_gpus if use_gpu else 0
    tune_gpu_fraction = 0.5 if use_gpu else 0
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    
    if use_gpu:
        max_concurrent = min(int(ray_gpus / tune_gpu_fraction), int(num_cpus / 2))
    else:
        max_concurrent = int(num_cpus / 2)
        
    max_concurrent = max(1, max_concurrent) 
    print(f"Ray will run {max_concurrent} trials concurrently using {ray_gpus} GPUs and {num_cpus} CPUs.")

    unique_dir = tempfile.mkdtemp()
    tune_log_dir = os.path.join(unique_dir, "ray_results") 
    
    import subprocess
    try:
        ray.init(
            num_gpus=ray_gpus, 
            num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), 
            include_dashboard=False, 
            ignore_reinit_error=True,
            _temp_dir=unique_dir
        )
    except Exception as e:
        print(f"Ray Init Failed: {e}")
        return

    all_results = []
    all_prob_matrices = {}
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_fold = 0
    
    for train_idx, test_idx in outer_cv.split(X, y):
        outer_fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fold_test_ids = [indices[i] for i in test_idx]

        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=outer_fold)
        
        search_space = {}
        n_trials = 0
        if args.baseline == 'LR':
            search_space = {
                "n_features": tune.choice([50, 100, 150, 200]), 
                "C": tune.loguniform(1e-4, 1e2), 
                "class_weight": tune.choice([None, "balanced"])
            }
            n_trials = 25
            
        elif args.baseline == 'RF':
            search_space = {
                "n_features": tune.choice([50, 100, 150, 200]), 
                "n_estimators": tune.choice([100, 200, 500]), 
                "max_depth": tune.choice([10, 20, 30, None]), 
                "min_samples_split": tune.choice([2, 5, 10])
            }
            n_trials = 30
            
        elif args.baseline == 'MLP':
            # identical to OURS for a fair ablation study
            search_space = {
                "n_features": tune.choice([50, 100, 150, 200]),
                "dense0": tune.choice([128, 256, 512]), 
                "dense1": tune.choice([64, 128, 256]), 
                "dense2": tune.choice([32, 64, 128]), 
                "dense3": tune.choice([16, 32, 64]), 
                "dropout": tune.choice([0.2, 0.3, 0.4, 0.5]), 
                "lr": tune.loguniform(1e-4, 1e-2)
            }
            n_trials = 45
            
        elif args.baseline == 'XGB':
            search_space = {
                "n_features": tune.choice([50, 100, 150, 200]), 
                "max_depth": tune.randint(3, 10), 
                "learning_rate": tune.loguniform(0.01, 0.3), 
                "n_estimators": tune.randint(100, 500)
            }
            n_trials = 30 
            
        elif args.baseline == '1D':
            search_space = {
                "filters": tune.choice([16, 32, 64, 128]), 
                "kernel_size": tune.choice([3, 5, 7, 11]), 
                "dense_units": tune.choice([64, 128, 256, 512]), 
                "lr": tune.loguniform(1e-4, 1e-2)
            }
            n_trials = 30 
            
        elif args.baseline == 'HEAD':
            search_space = {
                "n_features": tune.choice([50, 100, 150, 200]), 
                "l2_xgb_depth": tune.randint(3, 8), 
                "l2_xgb_lr": tune.loguniform(0.01, 0.3)
            }
            n_trials = 25 
            
        elif args.baseline == 'TABNET':
            search_space = {
                "n_features": tune.choice([50, 100, 150, 200]), 
                "n_d": tune.choice([8, 16, 32, 64]), 
                "n_a": tune.choice([8, 16, 32, 64]), 
                "lr": tune.loguniform(1e-3, 1e-1)
            }
            n_trials = 30
            
        elif args.baseline == 'OURS':
            search_space = {
                "n_features": tune.choice([50, 100, 150, 200]),
                "dense0": tune.choice([128, 256, 512]), 
                "dense1": tune.choice([64, 128, 256]), 
                "dense2": tune.choice([32, 64, 128]), 
                "dense3": tune.choice([16, 32, 64]), 
                "dropout": tune.choice([0.2, 0.3, 0.4, 0.5]), 
                "lr": tune.loguniform(1e-4, 1e-2)
            }
            n_trials = 45
        
        best_score = -np.inf
        best_params = {}

        if n_trials > 0:
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                X_t, X_val_inner = X_train[inner_train_idx], X_train[inner_val_idx]
                y_t, y_val_inner = y_train[inner_train_idx], y_train[inner_val_idx]

                cpus_per_trial = 2

                algo = ConcurrencyLimiter(OptunaSearch(metric="f1", mode="max"), max_concurrent=max_concurrent)


                tuner = tune.Tuner(
                    tune.with_resources(
                        tune.with_parameters(ray_trainable, X_train=X_t, y_train=y_t, X_val=X_val_inner, y_val=y_val_inner, args=args),
                        resources={"cpu": cpus_per_trial, "gpu": tune_gpu_fraction}
                    ),
                    run_config=tune.RunConfig(storage_path=tune_log_dir),
                    tune_config=tune.TuneConfig(search_alg=algo, num_samples=n_trials, max_concurrent_trials=max_concurrent),
                    param_space=search_space,
                )
                
                results = tuner.fit()
                best_result = results.get_best_result(metric="f1", mode="max")
                
                if best_result.metrics['f1'] > best_score:
                    best_score = best_result.metrics['f1']
                    best_params = best_result.config

                print("\n" + "="*60)
                print(f"BEST HYPERPARAMETERS FOR FOLD {outer_fold} - best macro F1 score: {best_score}")
                print("="*60)
                for key, value in best_params.items():
                    print(f"  -> {key}: {value}")
                print("="*60 + "\n")
                

        ood_datasets = {
            "noise_1.0": np.random.normal(0, 1.0, size=X_test.shape),
            "noise_3.0": np.random.normal(0, 3.0, size=X_test.shape),
            "noise_5.0": np.random.normal(0, 5.0, size=X_test.shape),
            "covariate_shift": X_test * 5.0
        }
        
        preds_test, f1_test, preds_proba, model, ood_probs, selected_features, entropy, ood_entropies = run_baseline(
            args, X_train, y_train, X_test, y_test, best_params, ood_datasets=ood_datasets, outer_fold=outer_fold
        )

        if args.baseline == 'OURS' and model is not None:
            os.makedirs("result/saved_models", exist_ok=True)
            save_path = f"result/saved_models/{args.data}_comb{args.comb}_fold{outer_fold}_{args.baseline}.keras"
            try:
                model.model.save(save_path)
            except Exception as e:
                model.model.save_weights(save_path.replace('.keras', '.weights.h5'))
                
            if selected_features is not None:
                scaler = StandardScaler()
                all_prob_matrices[f"fold_{outer_fold}_X_train_processed"] = scaler.fit_transform(X_train)[:, selected_features]
                all_prob_matrices[f"fold_{outer_fold}_y_train_processed"] = y_train

        y_test = y_test[:len(preds_test)]
        
        fold_results = pd.DataFrame({
            'id': fold_test_ids[:len(preds_test)],
            'outer_fold': np.ones(shape=len(y_test)) * outer_fold,
            'y_true': y_test,
            'y_pred': preds_test,
            'y_proba': preds_proba.tolist()
        })

        for i in range(preds_proba.shape[1]): fold_results[f'prob_class_{i}'] = preds_proba[:, i]
        all_results.append(fold_results)

        all_prob_matrices.update({
            f"fold_{outer_fold}_y_true": y_test,
            f"fold_{outer_fold}_best_params": str(best_params),

            # Real ID Test Set
            f"fold_{outer_fold}_real_probs": preds_proba,
            f"fold_{outer_fold}_real_entropy": entropy,
            
            # OOD: Additive Noise 1.0
            f"fold_{outer_fold}_noise_1.0_probs": ood_probs.get("noise_1.0"),
            f"fold_{outer_fold}_noise_1.0_entropy": ood_entropies.get("noise_1.0"),
            
            # OOD: Additive Noise 3.0
            f"fold_{outer_fold}_noise_3.0_probs": ood_probs.get("noise_3.0"),
            f"fold_{outer_fold}_noise_3.0_entropy": ood_entropies.get("noise_3.0"),
            
            # OOD: Additive Noise 5.0
            f"fold_{outer_fold}_noise_5.0_probs": ood_probs.get("noise_5.0"),
            f"fold_{outer_fold}_noise_5.0_entropy": ood_entropies.get("noise_5.0"),
            
            # OOD: Covariate Shift
            f"fold_{outer_fold}_covariate_shift_probs": ood_probs.get("covariate_shift"),
            f"fold_{outer_fold}_covariate_shift_entropy": ood_entropies.get("covariate_shift")
        })
        if selected_features is not None: all_prob_matrices[f"fold_{outer_fold}_features"] = selected_features


    ray.shutdown()
    results_df = pd.concat(all_results, ignore_index=True)
    
    os.makedirs("result/baselines", exist_ok=True)
    results_df.to_csv(f"result/baselines/{args.data}_nestedcv_predictions_{args.comb}_{args.baseline}.csv", index=False)
    np.savez_compressed(f"result/baselines/{args.data}_OOD_probs_comb{args.comb}_{args.baseline}.npz", **all_prob_matrices)

    y_true_all, y_pred_all = results_df['y_true'].values, results_df['y_pred'].values
    
    if args.comb == 2:
        print('Results for Task 1 - Tissue-of-Origin Task (14 Classes)')
    if args.comb == 0:
        print('Results for Task 2 - Cancer Task (19 Classes)')
    print(f"OVERALL Accuracy: {accuracy_score(y_true_all, y_pred_all):.3f}, Macro-F1: {f1_score(y_true_all, y_pred_all, average='macro'):.3f}")

    try:
        shutil.rmtree(unique_dir, ignore_errors=True)
        shutil.rmtree(os.path.expanduser("~/ray_results"), ignore_errors=True)
    except Exception as e:
        print(f"Cleanup error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN settings')
    parser.add_argument('--filter_k', type=int, default=150) 
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--dense0', type=int, default=512) 
    parser.add_argument('--dense1', type=int, default=512) 
    parser.add_argument('--dense2', type=int, default=512) 
    parser.add_argument('--dense3', type=int, default=512) 
    parser.add_argument('--epochs', type=int, default=70) 
    parser.add_argument('--data', type=str, default='miRNA') 
    parser.add_argument('--smote', action='store_true')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--comb', type=int, default=0) 
    parser.add_argument('--baseline', type=str, default='OURS')
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
    os.makedirs("result", exist_ok=True)
    os.makedirs("result/baselines", exist_ok=True)
    print('Start ')
    tracemalloc.start()
    st = time.time()
    main(args)
    elapsed_time = time.time() - st
    print(f'Execution time full: {(elapsed_time / 60):.2f} minutes')