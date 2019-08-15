train_config = {
    "dataset_name": "csv",
    "model_name": "resnet34",
    "device_ids": [0, 1, 2, 3],
    "seed": 7122,
    "num_workers": 8,

    "mode": "train",
    "lr": 4e-5,
    "batch_size": 1,
    "class_list":"/tmp2/patrickwu2/labeled_data/class_list.csv",
    "train_annotation":"/tmp2/patrickwu2/labeled_data/train_annotation.csv",
    "load_model_path": None,
    "param_only": False,
    "validation": True,
    "test_annotation":"/tmp2/patrickwu2/labeled_data/test_annotation.csv",
    "epoches": 100,
    "save_prefix": "img_grad_first_try",
}
