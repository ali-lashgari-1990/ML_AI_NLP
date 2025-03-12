# Q1

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score , precision_score, f1_score, recall_score
from scikitplot.metrics import plot_roc, plot_ks_statistic, plot_cumulative_gain
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.stats import ks_2samp
from statsmodels.distributions.empirical_distribution import ECDF
import scikitplot as skplt
import matplotlib.pyplot as plt

class performance_measure:
    def predict_y(self, prob_of_class1, classes, thres=0.5):
        predicted_y = []
        for i in range (len(prob_of_class1)):
            predicted_y.append(np.where(prob_of_class1[i]>thres, classes[1], classes[0]).tolist())

        return predicted_y
    
    
    def confusion_based(self, target_y, predicted_y, class1_label):
        #compute tpr, tnr, fpr, fnr, precision, recall, F1measure
        confusion_matrix(target_y, predicted_y)
        tn, fp, fn, tp = confusion_matrix(target_y, predicted_y).ravel() #just for binary (0,1)
        
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        fpr = fp/(fp+tn)
        fnr = fn/(tp+fn)
        
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        F1measure = 2*((precision*recall)/(precision+recall))


        result = {"TPR":tpr, "TNR":tnr, "FPR":fpr, "FNR":fnr, "precision":precision, "recall":recall, "F1":F1measure}
        
        return result

# function for average class accuracy   
    def ave_class_accuracy(self, target_y, predicted_y):
       
        confusion_matrix(target_y, predicted_y)
        tn, fp, fn, tp = confusion_matrix(target_y, predicted_y).ravel() #just for binary (0,1)
        
        arithmatic = 0.5 * ((tp/(tp+fn))+ (tn/(tn+fp))) # for just two classes
       
        h = (1/ ((tp/(tp+fn)))+ (1/(tn/(tn+fp))))
        harmonic = 1/(0.5*h) # for just two classes
        
        return {"ave":arithmatic, "HM":harmonic}
    
    
# function for plotting ROC curve and calculating AUC, ROC index, Gini coef
    def roc(self,  target_y, predicted_prob_class1, class1_label):   
        auc = roc_auc_score(target_y, predicted_prob_class1)
        roc_index = auc
        gini = (2*roc_index)-1
#         fpr, tpr, thres=roc_curve(target_y, predicted_prob_class1, class1_label)
#         plt.plot(fpr, tpr)
#         plt.show()

#         #computing auc
#         auc = skplt.metrics.roc_auc_score(fpr, tpr)
#         #computing roc_index
#         for i in range(2, len(thres)):
#         roc_index = sum([(tpr[i]+tpr[i-1])*(fpr[i]-fpr[i-1])])/2
#         #computing gini
#         gini = (2*roc_index)-1

        return {"AUC":auc, "ROC_index":roc_index, "Gini_coef":gini}

    
    
# function for computing and plotting Kolmogorov-Smirnov Statistic
    def KS(self, target_y, predicted_prob_class1):  
        # compute KS statistics KS_stat
        # render a plot of K-S chart in the function (see Figure 12)
        
        KS_stat = skplt.metrics.plot_ks_statistic(target_y, predicted_prob_class1)
        plt.show()
        
        return KS_stat 
    
    
# function for computing and plotting
    def cum_gain(self, target_y, predicted_prob_class1, target_percent, class1_label): 
        
        df = pd.DataFrame()
        df["target_y"] = target_y
        df["prob"] = predicted_prob_class1

        
        df["C1 or not"] = np.array(np.array(target_y) == class1_label, dtype = 'int')
        k = np.sum(np.array(target_y) == class1_label)
        
        df = df.sort_values(by=["prob"], ascending=False)
        
        df["gain"] =  df["C1 or not"].div(k)
        df["cumulative_gain"] = np.cumsum(df.gain)
        print(df)
        
        for i in range(len(target_y)): 
            if df.cumulative_gain.iloc[i] == target_percent:
            
                cum_gain_at_cut_off = df.cumulative_gain.iloc[i]
                score_cutoff = df.prob.iloc[i]
                break
            elif df.cumulative_gain.iloc[i] > target_percent:
                cum_gain_at_cut_off = df.cumulative_gain.iloc[i]
                score_cutoff = df.prob.iloc[i]
                break
            else:
                cum_gain_at_cut_off = 0
                score_cutoff = 0
                
#         skplt.metrics.plot_cumulative_gain(target_y, predicted_prob_class1)
#         plt.show()
        
        return (score_cutoff, cum_gain_at_cut_off)
