# Driver Drowsiness Detection System

This is our graduation project at Umm Al-Qura University, Data Science Department.  
We built a real-time system that detects driver drowsiness using a webcam, with the guidance of our supervisor **Dr. Mohammad Kamal Halwani**.

---

## What does it do?

The system watches the driver through a webcam and monitors three things at the same time:

- **Eye State (PERCLOS)** — uses a deep learning model to check if the eyes are open or closed. If the eyes are closed more than 35% of the time in a 3-second window, it triggers an alert
- **Yawn Detection** — another model detects yawning. If the driver yawns 2 or more times within 2 minutes, it gives a warning
- **Head Pose** — uses MediaPipe to track head position. If the head starts nodding down for more than 20 consecutive frames, it means the driver might be falling asleep

When one indicator is active you get an **orange warning**, and when PERCLOS triggers or two indicators are active at the same time you get a **red DANGER alert**.

---

## Model & Architecture

We used **EfficientNetV2-S** as the backbone with **CBAM attention mechanism** added on top of it. The model was trained in two phases — first we froze the backbone and only trained the attention and classification head, then we fine-tuned the whole network.

For the eye detection we also trained three separate submodels based on the sensor type in the MRL dataset, then combined them using **Weighted Majority Voting**.

**Datasets used:**
- MRL Eye Dataset — 84,898 infrared eye images — http://mrl.cs.vsb.cz/eyedataset
- Yawn Eye Dataset — from Kaggle — https://www.kaggle.com/datasets/davidvazquezcic/yawn-eye-dataset-new

### Accuracy Results

- Main Eye Model: 99.11%
- Majority Voting (3 submodels): 95.69%
- Yawn Detection: 99.53%

---

## Project Structure

The main code is inside the `src/` folder:

- `model.py` — the EfficientNetV2-S + CBAM model definition
- `dataset.py` — prepares the data from the raw MRL and Yawn datasets
- `train.py` — handles the training process
- `evaluate.py` — tests the models and saves the voting weights
- `demo.py` — the live webcam demo
- `alert_system.py` — audio alert manager (warning beep + danger beep)
- `make_plots.py` — generates ROC curves, F1 bar chart, and loss/accuracy curves

Results and saved models go into `outputs/` (models in `outputs/models/`, plots in `outputs/plots/`).

---

## How to run

Install dependencies first:
```bash
pip install -r requirements.txt
```

Build the dataset (you need MRL Eye Dataset in `mrl_data/` folder):
```bash
python src/dataset.py --task eye
python src/dataset.py --task yawn
```

Train the models:
```bash
python src/train.py                   # main eye model
python src/train.py --sensor small    # small sensor submodel
python src/train.py --sensor medium
python src/train.py --sensor large
python src/train.py --task yawn       # yawn detection model
```

Evaluate:
```bash
python src/evaluate.py --task all
```

Run the live demo:
```bash
python src/demo.py --cam 0
```

Press **Q** or **ESC** to close the demo window.

---

## Requirements

- Python 3.9+
- PyTorch with CUDA (we trained on RTX 5060 Ti 16GB)
- OpenCV, MediaPipe, NumPy, Pillow, scikit-learn
- Full list in `requirements.txt`

> **Note:** The trained model files (.pth) are too large for GitHub. You can download them from the Releases section.

---

## Team Members

This project was made by:

- Salman Talaq Alsulami
- Abdulaziz Mohammed Alghriby
- Abdullah Mansour Habit
- Mujahid Naji Al-Harbi
- Nasser Jamal Banjar

**University:** Umm Al-Qura University  
**Department:** Data Science  
**Supervisor:** Dr. Mohammad Kamal Halwani  
**Year:** 2025 / 2026

---

## Repository

https://github.com/SalmanAlsulami/drowsiness-detection
