import os
import torch
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from preprocessing import process_dataset
from train import PhotographClassifier


def classify_images(input_dir, model_path="best_model.pth"):
    device = torch.device('cuda')

    # Loading our model
    model = PhotographClassifier()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Resize images to 224x224 and convert to tensor (resizing to save gpu memory as a hotfix)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    # Loading in data, they need to be in folders labelling their classes
    dataset = datasets.ImageFolder(root=input_dir, transform=transform)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    y_true = []
    y_pred = []
    y_probs = []
    correct = 0
    # Classifying each image in dataset
    with torch.no_grad():
        for idx, (image, true_label) in enumerate(dataset):

            img_path = dataset.samples[idx][0]
            filename = os.path.basename(img_path)

            # Move image to GPU and get class prediction
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred = torch.argmax(output, dim=1).item()  # Largest output is the model prediction
            prob_arr = torch.softmax(output, dim=1)[0]  # softmax converts to a probability
            prob = prob_arr[pred].item()
            digital_prob = prob_arr[1].item()

            class_name = idx_to_class[pred]
            true_class = idx_to_class[true_label]

            # Count if prediction is correct
            if pred == true_label:
                correct += 1

            print(f"File: {filename}, Predicted: {class_name}, Probability: {prob:.2%}, True: {true_class}")

            # Saving variables for plotting confusion matrix & ROC curve
            y_true.append(true_label)
            y_pred.append(pred)
            y_probs.append(digital_prob)  # going to save prob for digital to construct a roc curve
            torch.cuda.empty_cache()  # Clear GPU memory each run

    accuracy = 100 * correct / len(dataset)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    return y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, classes):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, cmap='Blues')

    ax.set_title('Photograph Classifier Confusion Matrix')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')

    # Add percentages and raw counts to each cell
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            text = f'{cm_normalized[i, j]:.1%}\n({cm[i, j]})'
            ax.text(j, i, text, ha="center", va="center", color="white" if cm_normalized[i, j] > 0.5 else "black")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.tight_layout()
    plt.savefig("confusion.png", bbox_inches='tight')
    plt.show()
    return fig


def plot_roc_curve(y_true, y_probs):
    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})')

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("roc.png", bbox_inches='tight')
    plt.show()
    return fig


if __name__ == "__main__":
    """
    Note this script does not work in jupyter notebook format
    Executing this script will preprocess images and feed them into the model for classification
    """
    input_directory = r"Human Test Set - Unprocessed"
    intermediate_dir = r"unseen_classify_input"
    model_path = "best_model.pth"

    process_dataset(input_directory, intermediate_dir)

    # Classify images and save results
    true, pred, probs_dig = classify_images(
        "Unseen Test Set - Preprocessed")  # Uncomment to classify the unseen test photos
    # true, pred, probs_dig = classify_images(intermediate_dir)
    plot_confusion_matrix(true, pred, ("Film", "Digital"))
    plot_roc_curve(true, probs_dig)
