import flwr as fl
from client_app import FlowerClient
from server_app import RobustStrategy


def client_fn(cid):
    return FlowerClient(int(cid))


if __name__ == "__main__":
    strategy = RobustStrategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        ray_init_args={
        "object_store_memory": 100 * 1024 * 1024, # Force 100MB (higher than the 75MB minimum)
        "num_cpus": 1,
        }
    )
