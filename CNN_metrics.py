from sklearn import metrics
from tabulate import tabulate


def get_metrics(y_true, y_pred, class_mapping):

    total_predictions = [0, 0, 0]
    total_true = [0, 0, 0]

    cm = metrics.confusion_matrix(y_true, y_pred, labels=class_mapping)

    for i in range(3):
        for j in range(3):
            total_predictions[i] += cm[j][i]
            total_true[i] += cm[i][j]

    # recall and precision for each class
    r_h = cm[0][0] / total_predictions[0]
    r_m = cm[1][1] / total_predictions[1]
    r_r = cm[2][2] / total_predictions[2]
    p_h = cm[0][0] / total_true[0]
    p_m = cm[1][1] / total_true[1]
    p_r = cm[2][2] / total_true[2]

    # data for confusion matrix
    data = [["", class_mapping[0], class_mapping[1], class_mapping[2], "Total actual"],
            [class_mapping[0], cm[0][0], cm[0][1], cm[0][2], total_true[0]],
            [class_mapping[1], cm[1][0], cm[1][1], cm[1][2], total_true[1]],
            [class_mapping[2], cm[2][0], cm[2][1], cm[2][2], total_true[2]],
            ["Total predicted:", total_predictions[0], total_predictions[1], total_predictions[2], sum(total_true)]]

    # data for f1 score for each class
    data_f1 = [["Class", "F1-score"],
               ["Healthy", round((2 * p_h * r_h) / (p_h + r_h), 2)],
               ["Misalignment", round((2 * p_m * r_m) / (p_m + r_m), 2)],
               ["Rotor damage", round((2 * p_r * r_r) / (p_r + r_r), 2)]]

    # printing confusion matrix, f1 score table, recall and precision
    print(tabulate(data, tablefmt="simple_grid"))
    print()
    print(tabulate(data_f1, tablefmt="simple_grid"))
    print(f"\nRecall for healthy class: {round(r_h, 2)}")
    print(f"Recall for misalignment class: {round(r_m, 2)}")
    print(f"Recall for rotor damage class: {round(r_r, 2)}")
    print(f"\nPrecision for healthy class: {round(p_h, 2)}")
    print(f"Precision for misalignment class: {round(p_m, 2)}")
    print(f"Precision for rotor damage class: {round(p_r, 2)}")
