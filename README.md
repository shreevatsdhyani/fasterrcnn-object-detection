
**`faster-rcnn-object-detection-voc`**

# 📦 Object Detection with Faster R-CNN on Pascal VOC

This project implements a region-based object detection model using **Faster R-CNN** with a **ResNet-50 FPN** backbone on the **Pascal VOC 2007** dataset. It is developed as part of an internship assignment to understand end-to-end training, inference, and evaluation of a modern object detection pipeline using PyTorch and Google Colab.

---

## 📚 Model Architecture: Faster R-CNN Explained

Faster R-CNN (Region-based Convolutional Neural Network) consists of three main components:

1. **Backbone: ResNet-50 + FPN (Feature Pyramid Network)**
   - Extracts high-level feature maps at multiple scales from the input image.
   - FPN enhances the model's ability to detect objects at different sizes using a top-down pyramid structure.

2. **Region Proposal Network (RPN)**
   - Scans the feature maps to propose candidate regions (anchors) that might contain objects.
   - Each anchor is classified as object or background and its bounding box is refined.

3. **RoI Align + Detection Head**
   - Regions of interest are extracted and pooled to a fixed size using RoI Align.
   - The detection head consists of:
     - A **classifier** to assign object labels.
     - A **bounding box regressor** to fine-tune the proposed region coordinates.

This architecture enables high accuracy for both classification and localization tasks.

---

## 🧪 Training Details

| Parameter         | Value                            |
|------------------|----------------------------------|
| Dataset           | Pascal VOC 2007 (`trainval`)     |
| Model             | Faster R-CNN with ResNet-50 FPN  |
| Optimizer         | SGD with momentum                |
| Loss Functions    | RPN loss, Classification loss, BBox regression |
| Epochs (submitted)| 2                                |
| Epochs (extended) | 13 *(for better performance)*     |

> ⚠️ Due to runtime constraints on Colab, the official submission was trained for **2 epochs**. However, results significantly improved when retrained for **13 epochs**, showcasing sharper, more confident detections and reduced false positives.

---

## 📊 Results & Evaluation

- **Qualitative Evaluation** was done by plotting bounding boxes on images:
  - Green = Ground Truth
  - Red/Blue = Predicted Box with Label & Confidence

- **Visual Sample Output**:

  ![Sample Detection](./6eb473a3-9e6e-433a-b5bc-2f2954649fe4.png)

  *Detection Output: Identified persons, bottles, and a dining table using the Faster R-CNN model.*

- **Observations:**
  - Model quickly overfits small classes like “chair” with low epochs.
  - With 13 epochs, predictions became more precise and fewer false detections occurred.

---

## 🔧 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/faster-rcnn-object-detection-voc.git
   cd faster-rcnn-object-detection-voc


2. Open the Colab notebook:

   * [📔 Google Colab Notebook](#) *(Replace with actual link)*

3. Dependencies:

   * Python ≥ 3.8
   * PyTorch
   * `torchvision`
   * `matplotlib`, `numpy`

---

📊 Results & Evaluation
Qualitative Evaluation was done by plotting bounding boxes on sample test images.

Bounding boxes are:

🟥 Red or 🟦 Blue for predictions, depending on label/implementation

📌 Example Detection Output:


Figure: Output from Faster R-CNN (ResNet-50 FPN) trained on Pascal VOC — correct detections include person, bottle, and diningtable, showing decent performance even with limited training epochs.

## 🤖 AI Tools Used

| Tool           | Purpose                                                                  |
| -------------- | ------------------------------------------------------------------------ |
| ChatGPT        | Helped resolve CUDA errors, shape mismatches, and explained architecture |
| GitHub Copilot | Provided code suggestions and automated boilerplate logic                |

---

## 🧩 Key Learnings

* Region proposal-based detectors like Faster R-CNN are accurate but computationally heavy.
* Visualization aids both debugging and model interpretation.
* Pretrained models + fine-tuning greatly accelerate convergence.
* Epochs matter! More training = better predictions (if overfitting is managed).

---

## 💡 Future Improvements

* Integrate mAP calculation for quantifiable evaluation.
* Add anchor tuning or hyperparameter configuration support.
* Export the model to ONNX or TorchScript.
* Try alternative architectures: **YOLOv5, RetinaNet, or SSD**.

---

## 🙏 Final Note

This project helped me gain practical insight into object detection pipelines and training behavior. While limited by submission deadlines, my extended experiments highlighted the value of hyperparameter tuning and iterative improvement. I am enthusiastic about contributing further and building on this foundation.

Thank you for the opportunity!

— **Shreevats Dhyani**

```

Let me know if you want this zipped or linked to a GitHub repo structure too!
```
