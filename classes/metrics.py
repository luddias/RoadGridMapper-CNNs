import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import pandas as pd

class Metrics:
    def __init__(self):
        self.acc = []
        self.recall = []
        self.pre = []
        self.f1 = []
        self.iou = []
        
    def calculate_metrics(self, y_pred_flat, y_true_flat):
        y_pred_flat = y_pred_flat.flatten()
        y_true_flat = y_true_flat.flatten()
        
        # Calculate accuracy
        self.acc.append(accuracy_score(y_true_flat, y_pred_flat))

        # Calculate precision, recall, F1-score
        self.pre.append(precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0))
        self.recall.append(recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0))
        self.f1.append(f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0))

        # Calculate IoU
        self.iou.append(jaccard_score(y_true_flat, y_pred_flat, average='macro'))
        #Adicionar aqui calculo de q1, q2 e mediana da acurácia média
        
        
    def calculate_boxplot_values(self):
        # Ordenar os dados
        data = sorted(self.acc)
        # print(data)
        
        # Calcular a mediana (Q2)
        mediana = np.median(data)
        
        # Calcular o primeiro quartil (Q1)
        Q1 = np.percentile(data, 25)
        
        # Calcular o terceiro quartil (Q3)
        Q3 = np.percentile(data, 75)
        return "Mediana: "+ str(mediana), " Q1: "+ str(Q1)+ " Q3: "+ str(Q3)

    def get_results(self):
        bp_values = self.calculate_boxplot_values()
        medias_evaluation = {"Accuracy": [np.mean(self.acc)], "F1": [np.mean(self.f1)], "Precision": [np.mean(self.pre)], "Recall": [np.mean(self.recall)], "IoU": [np.mean(self.iou)]}
        return pd.DataFrame(medias_evaluation), bp_values
