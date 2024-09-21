# Skin Lesion Detection Using Deep Learning

## Project Overview
This project focuses on building a powerful deep learning-based application for detecting and classifying various types of skin lesions. By employing advanced convolutional neural networks (CNN), the model aims to accurately classify skin lesion images, providing a critical tool for early diagnosis and treatment of skin conditions.

## Dataset Overview
The **HAM10000** dataset forms the backbone of this project. It comprises **10,045 high-quality images** of different skin lesions, categorized into **seven distinct classes**:

- **nv**: Melanocytic nevi
- **mel**: Melanoma
- **bkl**: Benign keratosis-like lesions
- **bcc**: Basal cell carcinoma
- **akiec**: Actinic keratoses
- **vasc**: Vascular lesions
- **df**: Dermatofibroma

This diversity ensures the model can differentiate between a wide range of skin conditions with a high degree of precision.

## Installation & Setup
To get started, clone the repository and install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Required Libraries
Make sure the following libraries are installed:
- **PyTorch** and **Torchvision** for model implementation
- **TQDM** for progress tracking
- **Pandas**, **NumPy** for data manipulation
- **Scikit-learn** for evaluation metrics
- **Matplotlib** for visualization

---

## Data Preparation

1. **Image Directory**: Place all the images in your specified directory.
2. **Metadata**: Ensure the metadata file `HAM10000_metadata.csv` is located in the same directory.

Set the `base_dir` to the path where your images are stored:

```python
base_dir = '/path/to/your/image/directory'
all_image_path = glob(os.path.join(base_dir, '*.jpg'))

if not all_image_path:
    raise FileNotFoundError("No images found in the specified directory.")
else:
    print(f"Number of images found: {len(all_image_path)}")

metadata_path = os.path.join(base_dir, 'HAM10000_metadata.csv')
assert os.path.exists(metadata_path), "Metadata file not found!"
```

Next, link the image paths with the metadata and categorize the lesion types:

```python
df_original = pd.read_csv(metadata_path)
df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
```

---

## Model Training

The project utilizes the **ResNeXt** architecture, a state-of-the-art convolutional neural network model. To enhance model performance, several strategies are applied:
- **Data Augmentation**: Helps the model generalize better by artificially increasing the dataset size.
- **Class Balancing**: Ensures that the model doesn't become biased towards more frequent classes, leading to improved accuracy across all categories.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
```

---

## Model Evaluation

To assess the model’s performance, we use key evaluation metrics:
- **Confusion Matrix**: Provides insights into classification performance across classes.
- **Classification Report**: Detailed performance metrics such as precision, recall, and F1-score.

```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
```

These metrics will provide a clear picture of how well the model differentiates between various lesion types.

---

## Results
Once trained, the model demonstrates a high level of accuracy in classifying skin lesions. Comprehensive performance metrics, including precision, recall, and confusion matrices, will be shared following the model training phase.

---

## Contributing
We welcome contributions from the community! Feel free to submit pull requests or open issues to discuss new ideas and enhancements.

---

## License
This project is distributed under the **MIT License**. Please refer to the `LICENSE` file for further details.

---

## Acknowledgments
Special thanks to the creators of the **HAM10000 dataset** and the open-source community for their invaluable contributions and resources, which made this project possible.
```
