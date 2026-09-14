# TabPFN3BackboneLoader are TALENTLoaderWithPreprocessing adapted from 
# https://github.com/PriorLabs/TabPFN with specialization for TabPFN-3
# Licensed under Prior Labs License, see LICENSE.txt

import torch
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from paper.bench_utils import ModelLoader, BenchDataLoader, Eval
from paper.TabPFN_3.attention_wiring import ModifiedICLAttention

####################################
## Convenience class for benching ##
####################################

# includes manual kv-cache handling
# TabPFN-3 instantly deletes categorical_inds, so we do not pass it
# it also deletes task_type, although this is set from classifier as an attribute on the model on init.
# cache handling from InferenceEngineExplicitKVCache, but simplified for a single estimator
class TabPFN3BackboneLoader(ModelLoader):
    wrapper_classes = [ModifiedICLAttention]
    _model = None

    def __call__(self):
        # Init classifier to load weights and config
        clf = TabPFNClassifier(device="cuda")
        clf._initialize_model_variables()
        model = clf.model_
        model.to("cuda")
        model.eval()
        model.inference_row_chunk_size = 32768  # trying to speed up inference
        self._model = model
        return model
    
    def forward(self, model, x):
        x_train, y, _ = x
        with torch.no_grad(), torch.autocast("cuda"):
            _, cache = model(x_train, y, return_kv_cache=True, only_return_standard_out=True)
        return cache, y

    def post_forward(self, model, state, x):
        cache, y = state
        _, _, x_test = x
        with torch.no_grad(), torch.autocast("cuda"):
            out = model(x_test, y, kv_cache=cache, x_is_test_only=True, only_return_standard_out=True)
        return out[:, 0, :].cpu().numpy()

# the data preprocessing mirrors version 8.3.0 of the tabpfn-package
class TALENTLoaderWithPreprocessing(BenchDataLoader):
    def __init__(
            self, 
            dataset=None, 
            same_size_warmup=False, 
            big_run_length=100_000,
            big_run_features=25,
            max_test_samples=None,
        ):
        self.dataset = dataset
        self.y_test = None
        self.label_encoder = None
        self.data_warmup = False if same_size_warmup else True  # same_size_warmup=True for SVOO
        self.big_run_length = big_run_length
        self.big_run_features = big_run_features
        self.max_test_samples = max_test_samples

    def __call__(self):
        ###################
        ## Load datasets ##
        ###################
        
        import pickle
        from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
        from tabpfn.preprocessing.datamodel import FeatureModality
        from tabpfn.validation import ensure_compatible_predict_input_sklearn

        with open(f"/root/talent-large-datasets/talent_large/{self.dataset}.pkl", "rb") as f:
            rec = pickle.load(f)

        ######################
        ## Get clf instance ##
        ######################

        clf = TabPFNClassifier(device="cuda")
        clf.categorical_features_indices = rec["categorical_features_indices"]
        clf._initialize_model_variables()

        #####################################################
        ## Training data, mirror of TabPFNClassifier.fit() ##
        ## calling _initialize_model_variables()           ##
        #####################################################

        _, X_train, y_train = clf._initialize_dataset_preprocessing(
            X=rec["X_train"], y=rec["y_train"], random_state=0,
        )

        ##########################################################
        ## Test data, mirror of TabPFNClassifier._raw_predict() ##
        ##########################################################

        X_test = rec["X_test"]
        X_test = ensure_compatible_predict_input_sklearn(X_test, clf)
        X_test = fix_dtypes(
            X_test,
            cat_indices=clf.inferred_feature_schema_.indices_for(
                FeatureModality.CATEGORICAL
            ),
        )
        X_test = process_text_na_dataframe(
            X=X_test, 
            ord_encoder=clf.ordinal_encoder_,  # always set in _initialize_dataset_preprocessing
            # passthrough_inf in InferenceConfig defaults to False, same as process_text_na_dataframe
        )
        self.y_test = rec["y_test"]
        # optional subsampling here
        if self.max_test_samples is not None and len(X_test) > self.max_test_samples:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(X_test), self.max_test_samples, replace=False)
            X_test = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
            self.y_test = self.y_test[idx]

        #################################
        ## Stacking into torch tensors ##
        #################################

        # logic matching _build_cache and _call_model in InferenceEngineExplicitKVCache, simplified
        self.label_encoder = clf.label_encoder_  # we do not use this, instead we label-encode before running on modal
        # using default, float32. np.asarray is not doing anything here
        x_train = torch.tensor(np.asarray(X_train), dtype=torch.float32, device="cuda").unsqueeze(1)
        x_test = torch.tensor(np.asarray(X_test), dtype=torch.float32, device="cuda").unsqueeze(1)
        y = torch.tensor(np.asarray(y_train), dtype=torch.float32, device="cuda")
        print(
            f"Loading {self.dataset}: {x_train.shape[0]} train, "
            f"{x_test.shape[0]} test, {x_train.shape[2]} features"
        )
        return (x_train, y, x_test)

    def _make_random_data(self, n, c, n_classes):
        x = torch.randn(n, 1, c, dtype=torch.float32, device="cuda")
        y = torch.randint(0, n_classes, (n,), dtype=torch.float32, device="cuda")
        return (x, y, None)

    def warmup_data(self):
        return [
            self._make_random_data(1023, 100, 20),
            self._make_random_data(4095, 50, 10),
            self._make_random_data(self.big_run_length, self.big_run_features, 2),
        ]
    
# Eval
class Accuracy(Eval):
    name = "accuracy"
    def compute(out, reference, data_loader):
        preds = np.argmax(out, axis=1) if out.ndim > 1 else out
        acc = accuracy_score(data_loader.y_test, preds)
        return acc