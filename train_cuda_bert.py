from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import csv
import sys
import sklearn
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import numpy as np
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import sys
import torch
#from numba import cuda 

print( torch.version.cuda)
print (torch.backends.cudnn.enabled)
cuda_available = torch.cuda.is_available()
print ("Cuda available:")
print ( str(cuda_available))
#sys.exit(0)

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
#train_data = [['Example sentence 1 for multilabel classification.', [1, 1, 1, 1, 0, 1]], ['This is another example sentence. ', [0, 1, 1, 0, 0, 0]]]
#train_df = pd.DataFrame(train_data, columns=['text', 'labels'])

num_labels=2
train_data={}
test_data={}
train_data["text"]=[]
test_data["text"]=[]
train_data["labels"]=[]
test_data["labels"]=[]

string="/ibex/scratch/kafkass/Drug_Phenotype/sider/data4ST/sample.uniq.drug_based.train.mod.txt"
with open(string,'r') as f:
   reader=csv.reader(f,delimiter='\t')
   for text,se,ind in reader:
     labels=[]
     labels.append(int(se))
     labels.append(int(ind))
     train_data["text"].append(text)
     train_data["labels"].append(labels)

train_df = pd.DataFrame.from_dict(train_data) #, columns=['text', 'labels'])
#train_df, eval_df = train_test_split(df, test_size=0.2)

string="/ibex/scratch/kafkass/Drug_Phenotype/sider/data4ST/uniq.drug_based.test.mod.txt"
with open(string,'r') as f:
   reader=csv.reader(f,delimiter='\t')
   for text,se,ind in reader:
     labels=[]
     labels.append(int(se))
     labels.append(int(ind))
     test_data["text"].append(text)
     test_data["labels"].append(labels)

eval_df = pd.DataFrame.from_dict(test_data)

model_args={'max_seq_length': 256,
'reprocess_input_data': True,
'overwrite_output_dir': True,
'num_train_epochs': 10,
'output_dir':"/ibex/scratch/kafkass/NER/simple_transformers/cuda_bert_outputs",
'best_model_dir':"/ibex/scratch/kafkass/NER/simple_transformers/cuda_bert_outputs/best_model",
'save_eval_checkpoints': False,
'save_model_every_epoch': True,
'train_batch_size': 32,
'eval_batch_size': 32,
'fp16': True,
'save_steps': 5000,
'evaluate_during_training': True,
'evaluate_during_training_verbose': True,
'evaluate_during_training_steps': 2000,
'use_early_stopping':True,
'early_stopping_patience': 5,
'early_stopping_delta': 0,
'early_stopping_metric': "eval_loss",
'early_stopping_metric_minimize': True,
}

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel('bert', 'bert-base-cased', num_labels=2, args=model_args, use_cuda=cuda_available)
# You can set class weights by using the optional weight argument


# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)

#CALCULATE F-score

#binarization of the output
threshold, upper, lower = 0.5, 1, 0
new_out=np.where(model_outputs>threshold, upper, lower)
y_pred = np.array(new_out)

y_true=np.array([np.array(xi) for xi in eval_df["labels"]])

#recall=recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
#precision=precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
#f_score=f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

#print ("Recal=" + str (recall))
#print ("Precision=" + str (precision))
#print ("F1-score=" + str (f_score))

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

acc= accuracy_score(y_true, y_pred) 
print ("Accuracy:" + str (acc))


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_labels):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    #print ("roc-auc\t"+str(i)+"\t" +str(roc_auc[i]))
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print ("ROC-AUC:")
print (roc_auc)

##predictions, raw_outputs = model.predict(['This thing is entirely different from the other thing. '])
##print(predictions)
##print(raw_outputs)
#from scipy.special import softmax

#probabilities = softmax(raw_outputs, axis=1)
