"""
Author: Chi-An Chen
Date: 2025-07-10
Description: Draw Graph
"""
import ast
import matplotlib.pyplot as plt

file_path = 'saved_models_unet\output.txt'
train_losses = []
val_dice_scores = []
val_precision_scores = []
val_recall_scores = []
val_iou_scores = []

with open(file_path, 'r') as f:
    for line in f:
        try:
            start = line.find('[')
            end = line.find(']')
            values = ast.literal_eval(line[start:end+1])

            if len(values) == 5:
                train_losses.append(values[0])
                val_dice_scores.append(values[1])
                val_precision_scores.append(values[2])
                val_recall_scores.append(values[3])
                val_iou_scores.append(values[4])
        except Exception as e:
            print(f"Error parsing line: {line}\n{e}")

epochs = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(epochs, val_dice_scores, label='Val Dice', color='green')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.title('Validation Dice Score')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(epochs, val_precision_scores, label='Val Precision', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Validation Precision')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(epochs, val_recall_scores, label='Val Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Validation Recall')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(epochs, val_iou_scores, label='Val IoU', color='purple')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Validation IoU')
plt.grid(True)

plt.tight_layout()
plt.show()