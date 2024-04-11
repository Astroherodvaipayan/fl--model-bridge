from flask import Flask, request, jsonify
import warnings
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils
import pickle

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    try:
        # Load data
        (X_train, y_train), (X_test, y_test) = utils.load_data(client="client2")
        
        #take only 100 rows
        X_train = X_train[:100]
        y_train = y_train[:100]
        # Partition data
        partition_id = np.random.choice(10)
        (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

        # Initialize model
        model = LogisticRegression(
            solver= 'saga',
            penalty="l2",
            max_iter=10, 
            warm_start=True
        )

        # Set initial parameters
        utils.set_initial_params(model)

        # Define FlowerClient class
        class FlowerClient(fl.client.NumPyClient):
            def get_parameters(self, config): 
                return utils.get_model_parameters(model)

            def fit(self, parameters, config): 
                utils.set_model_params(model, parameters)
                # Ignore convergence failure due to low local epochs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                    filename = f"model/client2/client_2_round_{config['server_round']}_model.sav"
                    pickle.dump(model, open(filename, 'wb'))
                return utils.get_model_parameters(model), len(X_train), {}

            def evaluate(self, parameters, config): 
                utils.set_model_params(model, parameters)
                preds = model.predict_proba(X_test)
                loss = log_loss(y_test, preds, labels=[1,0])
                accuracy = model.score(X_test, y_test)
                return loss, len(X_test), {"accuracy": accuracy}

        # Start Flower client
        fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())

        # After training is completed, return the weights as JSON
        weights = model.coef_.tolist()
        return jsonify({"weights": weights})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
