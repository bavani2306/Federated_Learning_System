import flwr as fl
import numpy as np
from task import HeartModel, load_data, train, test, get_weights, set_weights

THRESHOLD = 0.01  # Loss sharpness threshold

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id

        # Inject noise for some clients (simulate junk hospital)
        inject_noise = True if client_id == 0 else False

        self.X_train, self.y_train, self.X_test, self.y_test = load_data(
            client_id, inject_noise
        )

        self.model = HeartModel(self.X_train.shape[1])

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        losses = train(self.model, self.X_train, self.y_train, epochs=5)

        # -------------------------
        # Compute Loss Sharpness
        # -------------------------
        sharpness = np.var(losses)

        print(f"Client {self.client_id} Loss Sharpness: {sharpness}")

        if sharpness > THRESHOLD:
            print(f"Client {self.client_id} BLOCKED due to noisy data.")
            # Even if blocked, we return the node_id so the dashboard can show the status
            return parameters, 0, {"node_id": int(self.client_id)} 

        # Return weights, dataset size, and the critical node_id metric
        return get_weights(self.model), len(self.X_train), {"node_id": int(self.client_id)}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        accuracy = test(self.model, self.X_test, self.y_test)
        return float(1 - accuracy), len(self.X_test), {"accuracy": accuracy}

def client_fn(cid):
    # CID comes as a string from Flower simulation, convert to int for our logic
    return FlowerClient(int(cid))

if __name__ == "__main__":
    # This part is used for manual connection if not using run_simulation.py
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(0),
    )