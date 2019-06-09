## Options

### Train option

#### ML\_GCN

train_dense161_gcn.json

```json
{
  "name": "densenet161_gcn_object_classification_no_balance" // name of your experiment
  , "use_tb_logger": true // whether use tensorboardX
  , "mode": "oc" // mode for solver, "os" | "wsod"
  , "is_train": true
  , "gpu_ids": [2]

  , "datasets": {
    "train": {
      "root_path": "/home/stevetod/jzy/dataset/PascalVOC/JPEGImages/" // path to your images
      , "info_path": "./data/train_anno.txt.bak" // information list of your data
      , "mode": "oc" // dataset mode
      , "MultiScaleCrop": true // use MultiScaleCrop like original ML_GCN does
      , "resize_size": 512 // resize original image to this size(square)
      , "input_size": 448 // input image size of network
      , "category_num": 20 // number of categories
      , "use_flip": true // use horizontal flip
      , "batch_size": 10
      , "num_workers": 8
    }
    , "val": {
      "root_path": "/home/stevetod/jzy/dataset/PascalVOC/JPEGImages/"
      , "info_path": "./data/test_anno.txt"
      , "mode": "oc"
      , "input_size": 448
      , "category_num": 20
      , "batch_size": 24
      , "num_workers": 8
    }
  }

  , "path": {
    "root": "./" // project root path
  }

  , "network_C": {
    "which_model_C": "densenet161_gcn" // type of model
    , "num_class": 20 // number of classes
    , "adj_file": "./data/voc_adj.pkl" // path to your adjacent matrix file
    , "inp_name": "./data/voc_glove_word2vec.pkl" // path to your word vector file
    , "in_channel": 300 // the same with the dimension of word vector
    , "t":0.4 // the threshold of matrix adj
    , "p":0.2 // the weight of matrix adj
  }

  , "solver": {
    "type": "ADAM" // optimizer type "ADAM" | "SGD"
    , "save_step": 5 // save every 5 epoch
    , "loss_type": "BCEWithLogits" // loss type, "BCEWithLogits" | "BCE"
    , "learning_rate": 1e-4
    , "weight_decay": 1e-4
    , "lr_scheme": "multisteplr" // lr scheduler, only "multisteplr"
    , "lr_steps": [10, 20, 30] // decay in theses epoches
    , "lr_gamma": 0.5 // decay rate
    , "epoch": 40 // total epoch
    , "balance_sample": false // whether balance sample(use pos_weight in "BCEWithLogits")
    , "pretrain": false // pretrain, resume or not, true | false | "resume"
  }

  , "logger": {
      "print_freq": 20 // log to tensorboardX every 20 step
  }
}
```

#### WSOD

train_dense161_HaS.json

```json
{
  "name": "densenet161_object_classification_ADAM_balance_naive_transform_HaS_valid"
  , "_comment": "ADAM optimizer, balance class weight, use resize and random crop, use HaS correctly"
  , "use_tb_logger": true
  , "mode": "oc"
  , "is_train": true
  , "gpu_ids": [4]

  , "datasets": {
    "train": {
      "root_path": "/home/stevetod/jzy/dataset/PascalVOC/JPEGImages/"
      , "info_path": "./data/train_anno.txt.bak"
      , "mode": "oc"
      , "resize_size": 256
      , "input_size": 224
      , "scale": [0.8, 1.0]
      , "category_num": 20
      , "use_flip": true
      , "batch_size": 32
      , "num_workers": 8
      , "HaS": { // Hide and seek option
        "S": 4 // number of splits for each axis, total number of patches is S*S
        , "p": 0.5 // the probability of hide
      }
    }
    , "val": {
      "root_path": "/home/stevetod/jzy/dataset/PascalVOC/JPEGImages/"
      , "info_path": "./data/test_anno.txt"
      , "mode": "oc"
      , "input_size": 224
      , "category_num": 20
      , "batch_size": 96
      , "num_workers": 8
    }
  }

  , "path": {
    "root": "./"
  }

  , "network_C": {
    "which_model_C": "densenet161"
    , "num_class": 20
  }

  , "solver": {
    "type": "ADAM"
    , "save_step": 5
    , "loss_type": "BCEWithLogits"
    , "learning_rate": 1e-4
    , "weight_decay": 1e-4
    , "lr_scheme": "multisteplr"
    , "lr_steps": [10, 15, 18]
    , "lr_gamma": 0.5
    , "epoch": 20
    , "balance_sample": true
    , "pretrain": false
  }

  , "logger": {
      "print_freq": 20
  }
}
```

### Test Option

#### WSOD

```json
{
  "name": "densenet161_object_classification_WSOD_HaS_valid"
  , "use_tb_logger": false
  , "mode": "wsod"
  , "is_train": false
  , "gpu_ids": [0]

  , "datasets": {
    "test": {
      "root_path": "/home/stevetod/jzy/dataset/PascalVOC/JPEGImages/"
      , "info_path": "./data/test_anno.txt"
      , "mode": "oc"
      , "input_size": 224
      , "category_num": 20
      , "batch_size": 100
      , "num_workers": 8
    }
  }

  , "path": {
    "root": "."
  }

  , "network_C": {
    "which_model_C": "densenet161"
    , "num_class": 20
  }

  , "solver": {
    "pretrain": true
    , "pretrained_path": "./experiments/densenet161_object_classification_ADAM_balance_naive_transform_HaS_valid/checkpoint/best_ckp.pth"
    , "bbox_th": [0.4, 0.45, 0.5, 0.55] // the threshold when transfer the CAM into binary map
    , "pred_th": 0.2 // use classes whose predicted probability is bigger than this
  }

  , "logger": {
      "print_freq": 20
  }
}
```

