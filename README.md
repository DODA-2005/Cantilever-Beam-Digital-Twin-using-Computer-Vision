# Cantilever Beam Digital Twin using Computer Vision

<p align="center">
  <img src="working.png" width="900">
</p>

<p align="center">
  <i>Real-time AR interaction (left) synchronized with a digital twin simulation (right).</i>
</p>

---

## 🎥 Demo

[▶️ Watch Demo Video](demo.mp4)


A **real-time, computer vision–driven digital twin** of a cantilever beam that visualizes load application and bending behavior using only a webcam and hand tracking.

This project was developed as part of a **Strength of Materials (SOM)** application to bridge theoretical beam bending concepts with interactive, visual intuition—**without using physical sensors**.

---

## 🎯 Project Overview

Traditional cantilever beam analysis relies on formulas, simulations, or strain sensors to study deflection behavior.  
This project demonstrates an alternative approach:

> **Use computer vision to detect a real beam and human interaction, then simulate bending behavior through a synchronized digital twin.**

The system detects a physical cantilever beam marked with a colored strip, tracks fingertip interaction as a point load, and renders:
- an **AR bending overlay** on the real beam, and  
- a **separate digital twin simulation** that mirrors the same behavior in real time.

---

## 🧱 Physical Setup

<p align="center">
  <img src="sunboard.png" width="500">
</p>

<p align="center">
  <i>Physical cantilever beam marked with a green strip for computer vision–based detection.</i>
</p>

---

## 📐 Cantilever Beam Theory (Reference)

<p align="center">
  <img src="twin.png" width="700">
</p>

<p align="center">
  <i>Classical cantilever behavior under a point load. Deflection increases quadratically toward the free end.</i>
</p>

---

## 🧠 Core Concepts Demonstrated

- Cantilever beam behavior under a point load  
- Load position–dependent deflection  
- Fixed-end vs free-end constraints  
- Qualitative bending behavior inspired by classical SOM theory  
- Digital twin synchronization between physical and simulated systems  

> ⚠️ Note:  
> This is a **qualitative, physics-inspired visualization**, not a numerically exact EI-based solver.

---

## 🧩 System Architecture

The system integrates **computer vision**, **hand tracking**, and **custom physics logic** in a single real-time loop.

### 1. Beam Detection (Computer Vision)
- Beam is wrapped with a **green strip** for robust detection
- HSV color segmentation is used to isolate the beam
- `cv2.minAreaRect` identifies beam geometry
- The **fixed end and free end** are inferred automatically
- Pressing **`c`** locks the beam geometry as the reference (undeflected state)

### 2. Finger Tracking (MediaPipe)
- MediaPipe Hands detects 21 landmarks
- Only **Landmark 8 (index fingertip)** is used
- Fingertip is treated as a **moving point load**
- Finger position is projected:
  - along the beam axis (load location)
  - perpendicular to the beam (load direction & magnitude)

### 3. Load Classification Logic
Finger interaction is classified into three mechanical states:
1. **Near** – finger close to the beam (no contact)
2. **Touching** – contact detected, no bending yet
3. **Load Applied** – downward push detected, bending triggered

Hover filtering and directional checks prevent false load application.

### 4. Physics-Inspired Bending Model
- Deflection curve increases quadratically from fixed end to free end  
- Inspired by classical cantilever behavior:  
  δ ∝ x²
- Deflection magnitude depends on:
  - normalized load position (x/L)
  - downward finger displacement
  - tunable sensitivity parameters

### 5. Digital Twin Visualization
- A separate simulation panel renders a **digital cantilever beam**
- Displays:
  - bending shape
  - load position (x%)
  - applied force direction
- Updates **synchronously** with real-world interaction

---

## 🖥️ Controls

- **`c`** → Lock detected beam geometry  
- **`u`** → Unlock beam  
- **`q`** → Quit application  

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – image processing, geometry, rendering
- **MediaPipe** – real-time hand landmark tracking
- **NumPy** – vector math and projections

---

## 📦 Requirements

```bash
pip install opencv-python mediapipe numpy
```

---

## ▶️ How to Run

```bash
python main.py
```

Ensure:
- A webcam is connected
- The beam is clearly marked with a green strip
- Adequate lighting for color segmentation

---

## 📌 Results

- Stable beam detection using color segmentation
- Reliable finger-based load classification
- Real-time bending visualization
- Digital twin accurately mirrors qualitative cantilever behavior

---

## 🔮 Future Scope

- Integrate force sensors for quantitative analysis  
- Support multiple or distributed loads  
- Auto-generate SFD & BMD plots  
- Extend to other structural elements  
- Mobile deployment for classroom demonstrations  

---

## 📚 Academic Context

This project was developed as part of a **Strength of Materials** course to provide an interactive, visual understanding of beam bending concepts using computer vision.

---

## 👤 Authors

- **Pranav Anil**  
