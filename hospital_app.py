import streamlit as st
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =========================================================
# 1. MODEL DEFINITION
# =========================================================
class HeartModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# 2. FLOWER CLIENT
# =========================================================
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, hospital_id, local_quality_score, audit_report, blocked_reason):
        self.model = model
        self.train_loader = train_loader
        self.hospital_id = hospital_id
        self.local_quality_score = local_quality_score
        self.audit_report = audit_report
        self.blocked_reason = blocked_reason

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        # CLIENT-SIDE BLOCKING: Stops training if the audit failed
        if self.local_quality_score < 0.45:
            print(f"\n🚨 HOSPITAL {self.hospital_id} BLOCKED LOCALLY: {self.blocked_reason}")
            return parameters, 0, {
                "node_id": int(self.hospital_id),
                "local_blocked": True,
                "quality_score": float(self.local_quality_score),
                "blocked_reason": self.blocked_reason
            }

        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        self.model.train()

        # Local Training Rounds
        for _ in range(3): 
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "node_id": int(self.hospital_id),
            "local_blocked": False,
            "quality_score": float(self.local_quality_score),
            "blocked_reason": "NONE"
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.1, len(self.train_loader.dataset), {}

# =========================================================
# 3. STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Hospital Node", page_icon="🏥", layout="wide")
st.title("🏥 Vertex Secure Hospital Node")
st.caption("Client-side integrity verification before federated participation")

if 'running' not in st.session_state:
    st.session_state.running = False

st.sidebar.header("⚙️ Node Configuration")
h_id = st.sidebar.number_input("Hospital ID", 0, 10, 0)
server_addr = st.sidebar.text_input("Server Address", "localhost:8080")
st.sidebar.markdown("---")
dataset_path = st.sidebar.text_input("📁 Dataset Path", placeholder="hospital_0.csv")

if st.button("🚀 Connect to Federation") or st.session_state.running:
    if not dataset_path:
        st.error("❌ Dataset path missing")
    elif not os.path.exists(dataset_path):
        st.error("❌ Dataset file not found")
    else:
        st.session_state.running = True
        try:
            df = pd.read_csv(dataset_path)
            original_rows = len(df)

            # --- COLUMN CLEANING ---
            df.columns = [c.lower().strip() for c in df.columns]
            COLUMN_MAPPER = {'target': 'num', 'diagnosis': 'num', 'label': 'num', 'heart_disease': 'num', 'thalachh': 'thalach'}
            df.columns = [COLUMN_MAPPER.get(c, c) for c in df.columns]
            REQUIRED_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

            # --- HANDLE STRINGS ---
            le = LabelEncoder()
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = le.fit_transform(df[col].astype(str))

            target_col = 'num' if 'num' in df.columns else df.columns[-1]
            y_raw = (df[target_col].values.astype(float) > 0).astype(int)

            for col in REQUIRED_COLS:
                if col not in df.columns:
                    df[col] = 0

            X_raw = df[REQUIRED_COLS].values

            # =========================================================
            # ADVANCED CLINICAL INTEGRITY AUDIT (PRE-TRAINING)
            # =========================================================
            audit_report = []
            quality_score = 1.0

            # 1. Biological Boundary Check (Catch the 200-year-olds)
            if (df['age'] > 120).any() or (df['age'] < 0).any():
                quality_score -= 0.40
                audit_report.append(f"CRITICAL: Impossible biological ages detected (Max: {df['age'].max()})")

            # 2. Physiological Consistency (BP & Heart Rate)
            if (df['trestbps'] > 250).any() or (df['thalach'] < 30).any():
                quality_score -= 0.40
                audit_report.append("CRITICAL: Vital signs outside human physiological limits")

            # 3. Categorical Integrity (Medical Codes)
            if (df['cp'] > 3).any() or (df['slope'] > 2).any():
                quality_score -= 0.30
                audit_report.append("INVALID: Out-of-range medical category codes detected")

            # 4. Statistical Correlation Check
            correlation = df['age'].corr(df['trestbps'])
            if abs(correlation) < 0.01:
                quality_score -= 0.20
                audit_report.append("WARNING: Dataset lacks natural biological correlation (Likely Noise)")

            # 5. Label Conflict Audit
            if len(df[y_raw == 1]) > 0 and len(df[y_raw == 0]) > 0:
                if df[y_raw == 1]['chol'].mean() < df[y_raw == 0]['chol'].mean():
                    quality_score -= 0.20
                    audit_report.append("LOGIC ERROR: Disease patterns contradict clinical norms")

            quality_score = max(0.0, quality_score)

            # --- DASHBOARD DISPLAY ---
            st.subheader("📊 Local Dataset Audit")
            c1, c2, c3 = st.columns(3)
            c1.metric("Dataset Quality", f"{quality_score:.2f}")
            c2.metric("Rows", original_rows)
            c3.metric("Biological Consistency", "FAILED" if quality_score < 0.45 else "PASSED")
            
            st.divider()

            if quality_score < 0.45:
                st.error("🚨 HIGH-RISK DATASET DETECTED: Participation Blocked")
                for issue in audit_report:
                    st.warning(issue)
                blocked_reason = "Client-side clinical integrity audit failure"
            else:
                st.success("✅ Dataset passed integrity checks")
                blocked_reason = "NONE"

            # --- PREPROCESSING & TRAINING ---
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            y_t = torch.tensor(y_raw, dtype=torch.float32).view(-1, 1)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=8, shuffle=True)

            model = HeartModel(input_dim=X_t.shape[1])
            client = HospitalClient(model, loader, h_id, quality_score, audit_report, blocked_reason)
            
            st.info(f"🏥 Hospital {h_id} initialized. Starting Secure Handshake...")
            fl.client.start_client(server_address=server_addr, client=client.to_client())
            
            st.success(f"✅ Round completed for Hospital {h_id}")
            st.session_state.running = False

        except Exception as e:
            st.error(f"❌ Processing Error: {e}")
            st.session_state.running = False