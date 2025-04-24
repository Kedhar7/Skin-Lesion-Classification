# Skin Cancer Detection and Classification using Deep Learning

## 📝 Project Overview
This project leverages **FastAI** to build a deep learning-based classifier for detecting and categorizing various skin lesions. It uses convolutional neural networks (CNNs), specifically **ResNet18**, trained on the **HAM10000** dataset. The ultimate goal is to assist in early and accurate diagnosis of skin conditions.

---

## ⚡ Recommended: Run on Google Colab

To ensure a smooth experience with GPU acceleration and minimal setup, it is **highly recommended to run this notebook on [Google Colab](https://colab.research.google.com/)**.

- Upload the notebook and dataset to Colab
- Enable **GPU** from `Runtime > Change runtime type > Hardware accelerator > GPU`
- Make sure all required libraries are installed in the first cell

---

## 📂 Dataset

We use the **HAM10000 (Human Against Machine with 10000 training images)** dataset, which contains dermatoscopic images categorized into 7 skin lesion types:

| Code  | Diagnosis                         |
|-------|----------------------------------|
| nv    | Melanocytic nevi                 |
| mel   | Melanoma                         |
| bkl   | Benign keratosis-like lesions    |
| bcc   | Basal cell carcinoma             |
| akiec | Actinic keratoses                |
| vasc  | Vascular lesions                 |
| df    | Dermatofibroma                   |

---

## ⚙️ Setup & Installation (For local use only)

To run locally (not recommended), install the following packages:

```bash
pip install fastai pandas matplotlib
```

---

## 🧪 Data Preparation

- Dataset is loaded using the **Kaggle API** via `kagglehub`
- Metadata is processed to assign readable class names
- Labels are dynamically generated using `get_label_from_dict`

```python
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=get_label_from_dict,
    item_tfms=[Resize(448), DihedralItem()],
    batch_tfms=RandomResizedCrop(size=224, min_scale=0.75, max_scale=1.0)
)
```

---

## 🧠 Model Architecture

- FastAI’s `vision_learner` is used with **ResNet18**
- Transfer learning with `fine_tune()` for 4 epochs
- Accuracy is the primary evaluation metric

```python
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(4)
```

---

## 📈 Evaluation & Results

- Predictions and metrics visualized with:
  - `dls.show_batch()`
  - Confusion matrix
  - Learning rate finder

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,6))
```

---

## 🎯 Highlights

- **DihedralItem** augmentations for better generalization
- Simplified data pipelines using **DataBlock API**
- GPU acceleration on Colab ensures quick training

---

## 🤝 Contributing
Pull requests and suggestions are welcome. Open an issue to discuss improvements.

---

## 📜 License
This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## Acknowledgements
Thanks to the Machine Learning community and the creators of the **HAM10000** dataset.
