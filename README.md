# 🛡️ Real-Time Weapon & Knife Detection

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-6706CE?style=for-the-badge)

A high-performance computer vision system designed to detect **Guns**, **Knives**, and **Rifles** in real-time. Built using the latest **YOLOv11** architecture, this project is optimized for security surveillance and threat detection.

---

## 🚀 Key Features
- **Real-Time Detection:** Low-latency inference for live webcam streams.
- **High Accuracy:** Optimized weights for weapon identification.
- **Easy Integration:** Simple script structure for deployment.
- **Multi-Class Support:** Specifically trained for Guns, Knives, and Rifles.

---

## 📊 Model Performance
The model was trained for **50 epochs** using the YOLOv11n architecture. Below are the metrics achieved on the test set:

| Metric | Value |
| :--- | :--- |
| **Precision** | 🟢 82.11% |
| **Recall** | 🔵 72.65% |
| **mAP @ 0.50** | 🟠 78.95% |
| **mAP @ 0.50:0.95** | ⚪ 38.36% |

---

📂 Project Directory Structure
<img width="8192" height="1581" alt="Admin Signup Flow-2026-05-03-181859" src="https://github.com/user-attachments/assets/2f67f808-79ed-42d1-a95f-887761d5fe3a" />


---

## 🛠️ Getting Started

### 1. Clone the Repository
`ash
git clone <your-repository-link>
cd knife-weapon-detection
`

### 2. Install Requirements
`ash
pip install -r requirements.txt
`

### 3. Run Live Detection
Use your laptop camera to detect weapons in real-time:
`ash
python scripts/live_camera_detect.py --model models/best.pt
`

---

## 🗃️ Dataset Source
The training data is hosted on **Roboflow Universe**:
- **Project:** [Weapon Detection (weapon-d98sm)](https://universe.roboflow.com/new-workspace-ketde/weapon-d98sm)
- **License:** CC BY 4.0

---

## 📜 License
This project is for educational and security research purposes. Distributed under the **MIT License**.

---

<p align="center">
  Developed with ❤️ for AI Security
</p>
