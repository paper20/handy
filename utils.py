
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def cal_metrics(y_true, y_pred, task):
    result = {}

    if task == "slide-2":
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        result = {
            "accuracy": round(accuracy,4),
            "precision": round(precision,4),
            "recall": round(recall,4),
            "f1": round(f1,4)
        }
        report = classification_report(y_true, y_pred)
        result['report'] = report
        # print(f'task {task} report:\n', report)

    else:
        accuracy = accuracy_score(y_true, y_pred)
        result['accuracy'] = round(accuracy,4)

        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        result['precision_micro'] = round(precision,4)
        result['recall_micro'] = round(recall,4)
        result['f1_micro'] = round(f1,4)

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        result['precision_macro'] = round(precision,4)
        result['recall_macro'] = round(recall,4)
        result['f1_macro'] = round(f1,4)

        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        result['precision_weighted'] = round(precision,4)
        result['recall_weighted'] = round(recall,4)
        result['f1_weighted'] = round(f1,4)

        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_matrix = conf_matrix.tolist()
        result['conf_matrix'] = conf_matrix

        report = classification_report(y_true, y_pred)
        result['report'] = report
        # print(f'task {task} report:\n', report)
  
    return result

def get_average_results(final_results):
    result_avg = {}
    task = final_results['task']
    if task != "slide-2":
        accuracy_total = 0
        precision_micro_total, precision_macro_total, precision_weighted_total = 0, 0, 0
        recall_micro_total, recall_macro_total, recall_weighted_total = 0, 0, 0
        f1_micro_total, f1_macro_total, f1_weighted_total = 0, 0, 0
        for split in final_results:
            if 'split' not in split:
                continue
            accuracy_total += final_results[split]["accuracy"]
            precision_micro_total += final_results[split]["precision_micro"]
            recall_micro_total += final_results[split]["recall_micro"]
            f1_micro_total += final_results[split]["f1_micro"]
            precision_macro_total += final_results[split]["precision_macro"]
            recall_macro_total += final_results[split]["recall_macro"]
            f1_macro_total += final_results[split]["f1_macro"]
            precision_weighted_total += final_results[split]["precision_weighted"]
            recall_weighted_total += final_results[split]["recall_weighted"]
            f1_weighted_total += final_results[split]["f1_weighted"]

        accuracy_avg = round(accuracy_total/5,4)

        precision_micro_avg, precision_macro_avg, precision_weighted_avg = round(precision_micro_total/5,4), round(precision_macro_total/5,4), round(precision_weighted_total/5,4)
        recall_micro_avg, recall_macro_avg, recall_weighted_avg = round(recall_micro_total/5,4), round(recall_macro_total/5,4), round(recall_weighted_total/5,4)
        f1_micro_avg, f1_macro_avg, f1_weighted_avg = round(f1_micro_total/5,4), round(f1_macro_total/5,4), round(f1_weighted_total/5,4)

        result_avg = {
            "accuracy_avg": accuracy_avg,
            "precision_micro_vg": precision_micro_avg,
            "recall_micro_avg": recall_micro_avg,
            "f1_micro_avg": f1_micro_avg,
            "precision_macro_vg": precision_macro_avg,
            "recall_macro_avg": recall_macro_avg,
            "f1_macro_avg": f1_macro_avg,
            "precision_weighted_vg": precision_weighted_avg,
            "recall_weighted_avg": recall_weighted_avg,
            "f1_weighted_avg": f1_weighted_avg
        }
    else:
        accuracy_total = 0
        precision_total = 0
        recall_total = 0
        f1_total = 0
        for split in final_results:
            if 'split' not in split:
                continue
            accuracy_total += final_results[split]["accuracy"]
            precision_total += final_results[split]["precision"]
            recall_total += final_results[split]["recall"]
            f1_total += final_results[split]["f1"]
        accuracy_avg = round(accuracy_total/5,4)
        precision_avg = round(precision_total/5,4)
        recall_avg = round(recall_total/5,4)
        f1_avg = round(f1_total/5,4)
        result_avg = {
            "accuracy_avg": accuracy_avg,
            "precision_avg": precision_avg,
            "recall_avg": recall_avg,
            "f1_avg": f1_avg
        }
    
    return result_avg



        



    
    

