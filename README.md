# Stroke-Hand-Rehabilitation-Robot
## Project Overview
Stroke patients often suffer from long-term impairments in hand function, which greatly affects their independence and quality of life. Traditional physiotherapy relies heavily on therapists and offers limited scalability and personalization. Early robotic solutions provided only fixed motion patterns and failed to adapt to each patient’s unique neuromuscular characteristics.

Our project introduces an sEMG-based hand rehabilitation robot designed to deliver personalized and adaptive rehabilitation. By analyzing surface electromyography (sEMG) signals, the system can dynamically adjust training programs according to the patient’s muscle activity, providing a more comfortable, precise, and efficient recovery process.

Under the joint guidance of Xi'an Jiaotong-Liverpool University and NUS (Suzhou) Research Institute, we designed a hand rehabilitation robot for stroke patients.

## Methods

Biomimetic cable-driven mechanism: Lightweight and flexible, enabling natural finger movements while reducing patient fatigue.

CNN-Transformer hybrid algorithm: Achieves over 90% accuracy in gesture prediction, allowing the robot to respond dynamically to patient intent.

Multi-sensor fusion: Combines sEMG, Flex sensors, and force feedback for safety and real-time adjustments.

<img width="454" height="254" alt="项目流程图" src="https://github.com/user-attachments/assets/6308d474-685b-49b8-b521-2028ebdf3c49" />

## ⚙️ Rope-Driven System

### 🪶 2.1.1 Lightweight & Natural Movement
Traditional rigid exoskeletons restrict natural hand motion and add significant inertia, forcing patients to exert more effort.  
Our **cable-driven system** relocates the power source away from the hand, significantly reducing weight. This enables **lighter, more natural, and less fatiguing movements**, which is crucial for patients with limited strength or endurance.

### 🎯 2.1.2 Precise & Adaptive Force Control
Pneumatic-driven devices struggle with precise force control due to air compression characteristics, risking overload on fingers or wrists.  
By contrast, our **motor + force feedback–based cable system** ensures **fine-grained, adaptive control** of applied force for safer, more effective rehabilitation.

### 🦴 2.1.3 Biomimetic Cable Arrangement
We employ an **underactuated differential mechanism** that uses a single motor to drive finger flexion while automatically balancing forces among fingers.

- **Palmar-side cables:** drive flexion  
- **Dorsal-side cables:** control extension  

Remote **high-torque forearm motors** manage:  
- Four-finger flexion/extension  
- Independent thumb flexion–extension and abduction–adduction  

This design mimics human tendon structure for more **biological and coordinated motion**.

### 🛡️ 2.1.4 Safety & Wearable Design
Safety is ensured through:

- **Dual control modes:** manual and automatic switching  
- **Sensor fusion:** sEMG + FLEX bend sensors  
- **Closed-loop feedback with alarms**  

The **open, adjustable design** fits various hand sizes, is easy to wear, and allows for sanitary maintenance — ideal for patients with limited mobility.

## 🧩 2.2 sEMG Signal Module

### 🔍 Overview
The system uses **surface electromyography (sEMG)** signals to capture hand muscle activity.  
A total of **222 samples** are collected, covering **five distinct hand gestures**.  
These signals are processed to train a **CNN–Transformer hybrid model** that enables accurate and adaptive gesture recognition.

### 🧠 Model Architecture
The model combines **Convolutional Neural Networks (CNNs)** for local feature extraction and **Transformers** for sequential dependency modeling.

**Architecture Summary:**
- **CNN Layers:** Capture spatial and temporal sEMG features  
- **Transformer Encoder:** Models global temporal dependencies  
- **Fully Connected Layer:** Outputs 5 gesture classes  

**Key Parameters:**
| Parameter | Value |
|------------|--------|
| Input Channels | 1 |
| Number of Classes | 5 |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Epochs | 50 |

### ⚙️ Training Process
- Data split: 80% training, 20% testing  
- Optimizer: Adam  
- Loss Function: CrossEntropyLoss  
- Gradient Clipping: Enabled (max_norm = 1.0)  
- Visualization: Loss curve & class distribution via Matplotlib  

### 📈 Results
The model achieves reliable classification performance across five gestures, demonstrating the potential of **sEMG-based control** for personalized hand rehabilitation.

## 🧾 2.3 FLEX Sensor

### 🎯 Accuracy
The **FLEX Sensor** measures finger bending angles during rehabilitation.  
Accurate calibration ensures reliable data despite differences in user skin type or hand size.

### ⚙️ Durability
Built for frequent bending and long-term use, the sensor maintains stable performance and reduces replacement costs.

# Poster
![康复机器人海报水印](https://github.com/user-attachments/assets/c17a3a4c-d83b-4c37-9b6d-256d1bcb1dc5)
