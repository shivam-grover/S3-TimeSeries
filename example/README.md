# Hyperparameter Tuning with Optuna

This provides a template for performing hyperparameter tuning for S3 using **Optuna** with your own models. You can also monitor the tuning process using the **Optuna dashboard**.

### Install dependencies
```bash
pip install optuna joblib optuna-dashbaord
```

### Modify the template for your model
To adapt this template for your own model, you'll need to replace the placeholder components with your specific model architecture, data loading, and evaluation logic. This involves adjusting the model initialization, defining how your data is loaded, and incorporating your training loop. Modify the logging and evaluation functions to capture the metrics that to be used by the objective function (usually the validation loss or accuracy).


### Viewing Results with Optuna Dashboard
Optuna provides a useful dashboard to monitor the optimization process and visualize the trials.
To launch the dashboard, run:

```bash
optuna-dashboard sqlite:///optuna_study.db
```

If you are running your experiments on a remote server and would like to visualise the optuna study on your local machine then run the command above on your server, and then on your local machine run the following command

```bash
ssh -L 8080:localhost:8080 username@remote_server_ip
```

This will forward the remote server's port 8080 to your local machine. Now, you can open a browser on your local machine and navigate to http://localhost:8080 to access the Optuna dashboard and visualize the study's progress, best trials, and hyperparameter tuning details.

For any clarification, please contact shivam.grover@queensu.ca.
