# Privacy-Preserving Federated Learning Framework for Heart Disease Prediction

A privacy-preserving Federated Learning framework that enables multiple hospitals to collaboratively train a heart disease prediction model without sharing sensitive patient data. The system incorporates robust aggregation, noisy client detection, client-side dataset auditing, and a real-time monitoring dashboard to improve the reliability, security, and transparency of distributed healthcare AI.

---

## Features

- Privacy-preserving collaborative model training using Federated Learning
- Local training without sharing patient records
- Client-side dataset integrity auditing before model participation
- Server-side noisy client detection using Geometric Median and Cosine Similarity
- Robust aggregation to reduce the impact of malicious or unreliable clients
- Real-time Streamlit dashboard for monitoring training progress and client reliability
- Security alerts and node health visualization
- Supports both simulation-based and manual multi-hospital federated environments

---

## Workflow

```
Local Hospital Dataset
        ↓
Client-Side Dataset Audit
        ↓
Local Model Training
        ↓
Model Weight Sharing
        ↓
Robust Server Aggregation
        ↓
Noisy Client Detection
        ↓
Global Model Update
        ↓
Real-Time Dashboard Monitoring
```
---

## Tech Stack

- Python
- PyTorch
- Flower (Federated Learning)
- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-learn

---

## Key Components
- Heart Disease Dataset
- Local Hospital Training
- Client-Side Dataset Auditing
- Geometric Median Aggregation
- Cosine Similarity-Based Gradient Analysis
- Noisy Client Detection
- Global Model Aggregation
- Interactive Monitoring Dashboard


## Project Structure

```
Fed Health/
│── client_app.py
│── hospital_app.py
│── server_app.py
│── dashboard.py
│── run_simulation.py
│── task.py
│── generate_proof.py
│── heart.csv
│── session_stats.json
│── README.md
```
---

## Installation

```bash
git clone https://github.com/yourusername/Vertex.git
cd Vertex
pip install -r requirements.txt
```

Run the simulation:

```bash
python run_simulation.py
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

For manual hospital nodes:

```bash
python server_app.py
streamlit run hospital_app.py
```
---

## Applications

- Privacy-Preserving Healthcare AI
- Multi-Hospital Collaborative Learning
- Federated Medical Data Analysis
- Secure Clinical Decision Support
- Distributed Machine Learning Research
- Healthcare Data Privacy and Security

---

## Future Enhancements

- Differential Privacy integration
- Secure Aggregation and Homomorphic Encryption
- Blockchain-based client trust management
- Advanced AI-based anomaly detection
- Real-time hospital network deployment
- Cloud-based federated learning infrastructure

---
## Contributors

**Bavani**
- Final Year B.Tech Project – Artificial Intelligence and Data Science using Python

---

## License

This project is developed for academic and research purposes.
