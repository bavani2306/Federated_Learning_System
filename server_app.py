import flwr as fl
import numpy as np
import json
import time
import os
import torch
import torch.nn.functional as F

# ============================================
# 1. ROBUST AGGREGATOR: GEOMETRIC MEDIAN
# ============================================
def geometric_median(updates, max_iter=10, eps=1e-5):
    """Calculates the statistical center resistant to outliers."""
    median = np.mean(updates, axis=0)
    for _ in range(max_iter):
        distances = np.array([np.linalg.norm(u - median) + eps for u in updates])
        weights = 1 / distances
        new_median = np.average(updates, axis=0, weights=weights)
        if np.linalg.norm(new_median - median) < eps:
            break
        median = new_median
    return median

# ============================================
# 2. GLOBAL TRUST ENGINE
# ============================================
trust_scores = {}

class RobustStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        print(f"\n========== SECURE AUDIT ROUND {rnd} ==========")

        # --- MODULE 2: DATA EXTRACTION & SORTING [cite: 80, 82] ---
        extracted_data = []
        for client, fit_res in results:
            w = fl.common.parameters_to_ndarrays(fit_res.parameters)
            n_id = str(fit_res.metrics.get("node_id", client.cid))
            # Capture the local audit signal we added to hospital_app.py
            is_blocked_locally = fit_res.metrics.get("local_blocked", False)
            
            flat = np.concatenate([layer.flatten() for layer in w])
            extracted_data.append({
                "id": n_id,
                "weights": w,
                "flat": flat,
                "local_blocked": is_blocked_locally
            })
        
        # FIX: Explicitly sort by ID so Dashboard Index 0 always = Hospital 0
        extracted_data = sorted(extracted_data, key=lambda x: x["id"])
        
        flat_updates = [x["flat"] for x in extracted_data]
        node_ids = [x["id"] for x in extracted_data]
        weights = [x["weights"] for x in extracted_data]

        # --- CALCULATE CONSENSUS ANCHOR [cite: 57, 82] ---
        median_update = geometric_median(flat_updates)
        median_tensor = torch.tensor(median_update).unsqueeze(0)

        # --- MODULE 3: SECURITY AUDIT [cite: 56, 82] ---
        node_reports = []
        for i in range(len(flat_updates)):
            nid = node_ids[i]
            if nid not in trust_scores: trust_scores[nid] = 100

            # 1. Check Local Signal: Did the hospital block itself?
            local_fail = extracted_data[i]["local_blocked"]

            # 2. Check Mathematical Signal: Does it diverge from the median?
            u_tensor = torch.tensor(flat_updates[i]).unsqueeze(0)
            similarity = F.cosine_similarity(u_tensor, median_tensor).item()

            # --- THE MULTI-FACTOR BLOCKING GATE ---
            # If the hospital flagged its own data (biological failure)
            # OR if the math is divergent (similarity < 0.94), we drop trust.
            if local_fail or (similarity < 0.94):
                trust_scores[nid] -= 50  # Heavy penalty to force Red Bar
                status = "NOISY / BLOCKED"
                print(f"🚨 ALERT: Hospital {nid} Blocked. LocalFail={local_fail}, Sim={similarity:.4f}")
            else:
                trust_scores[nid] += 5
                status = "HEALTHY"

            trust_scores[nid] = max(0, min(100, trust_scores[nid]))
            reliability = trust_scores[nid]

            node_reports.append({
                "node_id": nid,
                "reliability": float(reliability),
                "status": status
            })

        # --- GLOBAL WEIGHT RECONSTRUCTION ---
        reference = weights[0]
        new_weights = []
        idx = 0
        for layer in reference:
            size, shape = layer.size, layer.shape
            new_weights.append(median_update[idx:idx + size].reshape(shape))
            idx += size

        # --- DASHBOARD SYNCHRONIZATION [cite: 25, 88] ---
        stats = {
            "round": int(rnd),
            "weights_sum": float(np.sum(median_update)),
            "timestamp": time.time(),
            "active_nodes": len(results),
            "node_details": node_reports,
            "status": "COMPLETED" if int(rnd) >= 5 else "TRAINING"
        }

        with open("session_stats.json", "a") as f:
            f.write(json.dumps(stats) + "\n")

        return fl.common.ndarrays_to_parameters(new_weights), {}

# ============================================
# 4. START RESILIENT SERVER
# ============================================
if __name__ == "__main__":
    if os.path.exists("session_stats.json"):
        os.remove("session_stats.json")

    print("🚀 Vertex Resilient Server Ready. Dashboard monitoring active.")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=RobustStrategy(
            min_fit_clients=3, 
            min_available_clients=3
        ),
    )