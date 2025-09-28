# server.py
import flwr as fl

def main():
    # Use default FedAvg strategy for demo
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # fraction of clients used during training
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(server_address="127.0.0.1:8080", strategy=strategy, config={"num_rounds": 3})

if _name_ == "_main_":
    main()