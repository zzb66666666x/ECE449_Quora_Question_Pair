# Quora Question Pair

get data: https://zjuintl-my.sharepoint.com/:f:/g/personal/zhongbo_18_intl_zju_edu_cn/EqDUd6SspnhOoKTknMooYUYBCrM70iFdcwS7XNi_KB37iw?e=xSTMbY.

## Author @ ZJU-UIUC Institue

- Zhu Zhongbo
- Xiao Shuhong
- Pan Bo
- Lian Xinyu

- Lu Yichen

## How to run Bi-LSTM/CNN/MLP

github: [zzb66666666x/ECE449_Quora_Question_Pair (github.com)](https://github.com/zzb66666666x/ECE449_Quora_Question_Pair)

1. make sure you have the following pre-processed data

   ```
   Q1_TRAINING_DATA_FILE = 'q1_train.npy'
   Q2_TRAINING_DATA_FILE = 'q2_train.npy'
   Q1_TEST_DATA_FILE = 'q1_test.npy'
   Q2_TEST_DATA_FILE = 'q2_test.npy'
   TEST_ID_FILE = 'test_ids.npy'
   LABEL_TRAINING_DATA_FILE = 'label_train.npy'
   WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
   NB_WORDS_DATA_FILE = 'nb_words.json'
   TRAIN_FEAT_NPY_FILE = 'train_feat_array.npy'
   TEST_FEAT_NPY_FILE = 'test_feat_array.npy'
   ```

2. use the quick_run.ipynb to run the model, make sure you have the following file

   ```
   RNN_model.h5
   CNN_model.h5
   MLP_model.h5
   ```


## How to run xgboost

1. please put all data into a folder called data in your workspace
2. use the xgboost.ipynb to run the code
3. xgb_par.csv keeps the params needed

 
