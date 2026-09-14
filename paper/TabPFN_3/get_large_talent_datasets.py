# - download benchmark_dataset.zip from here: https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z
#     we take out the datasets with more than 100k samples total (train+test), 
#     similarily to what is done in the TabPFN-3 technical report (table 13).
#     however, it appears like only the classification datasets are actually present
# - put the data in 'paper/TabPFN_3/data/talent_large/'
# - create a volume using 'modal volume create talent-large-datasets'
# - then run 'modal volume put talent-large-datasets paper/TabPFN_3/data/talent_large/ talent_large/ --force'
#     we run evals with 'train' as training split and 'test' as test split. 
#     to get longer datasets, we could use 'train'+'val' as training split

import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

datasets = [
    "microsoft",  # not found
    "poker-hand",  # not found
    "BNG(credit-a)",  # not found
    "Higgs",  # not found
    "Smoking_and_Drinking_Dataset_with_body_signal",
    "yahoo",  # not found
    "Data_Science_for_Good_Kiva_Crowdfunding",
    "covertype",  # not found
    "CDC_Diabetes_Health_Indicators",
    "accelerometer",
    "walking-activity",
    "Rain_in_Australia",
    "customer_satisfaction_in_airline",
    "dabetes_130-us_hospitals"  # diabetes is misspelled in the benchmark suite
]

TALENT_DIR = Path("paper/TabPFN_3/data/TALENT")
OUT_DIR = Path("paper/TabPFN_3/data/talent_large")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class Dataset:
    def __init__(self, name):
        self.name = name
    
    name: str
    task_type: str
    categorical_features_indices: object
    label_map: object  # exists for classification problems, mapping encoded labels to original

    X_train: object
    y_train: object
    X_val: object
    y_val: object
    X_test: object
    y_test: object

classification_task_types = ["binclass", "multiclass"]
splits = ["train", "val", "test"]
feature_types = ["N", "C"]  # numeric and categorical

for name in datasets:
    # Checking if dataset exists
    dataset_path = TALENT_DIR/name
    if (dataset_path/"info.json").exists(): 
        print("name:", name)
        dataset = Dataset(name)
    else:
        print("name:", name, "not found")
        continue
    
    # Loading info
    with open(dataset_path/"info.json") as f: info = json.load(f)
    dataset.task_type = info["task_type"]
    print("  task_type:", dataset.task_type)

    # Loading features
    for split in splits:
        split_parts = []
        for prefix in feature_types:
            feature_path = dataset_path/f"{prefix}_{split}.npy"
            if feature_path.exists():
                X_part = np.load(feature_path, allow_pickle=True)

                # note: this block replaces categorical nans with their own category.
                # tabpfn-3 applies a missingness term to nans. we should remove this block
                # and rely on internal tabpfn-3 preprocessing, in general more robust to do that.
                if not np.issubdtype(X_part.dtype, np.number):  # not normal numeric
                    #print(X_part.dtype)  # seeing only 'object'
                    #print(X_part)  # strings specifically, with some nans
                    if prefix!="C": raise ValueError("!!!")  # should only happen for categorical ones
                    X_part = X_part.astype(str)
                    X_part = np.column_stack(  # turning into ints
                        [np.unique(X_part[:, c], return_inverse=True)[1] for c in range(X_part.shape[1])]
                    )
 
                # saving in float32, as this is what is used downstream. 
                split_parts.append(X_part.astype(np.float32))
        setattr(dataset, f"X_{split}", np.concatenate(split_parts, axis=1))
        setattr(dataset, f"y_{split}", np.load(dataset_path/f"y_{split}.npy", allow_pickle=True))
        print(f"  n_{split}:", len(getattr(dataset, f"y_{split}")))
    print("  n_features:", dataset.X_train.shape[1])

    # Label-encoding classification tasks, to not have to deal with it downstream
    if info["task_type"] in classification_task_types:
        le = LabelEncoder()
        # this assumes all classes are in the train split.
        # otherwise transform will throw an error
        le.fit(dataset.y_train)  
        for split in splits:
            y_split = getattr(dataset, f"y_{split}")
            setattr(dataset, f"y_{split}", le.transform(y_split))
        dataset.label_map = {i: orig for i, orig in enumerate(le.classes_)}

    # Noting which features were categorical
    dataset.categorical_features_indices = list(
        range(info["n_num_features"], info["n_num_features"] + info["n_cat_features"])
    ) if info["n_cat_features"] else None

    # Saving
    with open(OUT_DIR/f"{name}.pkl", "wb") as f:
        pickle.dump(dataset.__dict__, f)