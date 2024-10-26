import torch
import optuna
import argparse
import os
import sys
import time
import logging
import pickle
import joblib

ts = time.time()

def objective(trial):
    """
    Define the objective function for Optuna. It evaluates a given model using different hyperparameter configurations.
    """
    # Suggest hyperparameters for S3 or general model configurations
    num_segments = trial.suggest_categorical("num_segments", [2, 4, 8, 16, 24])
    num_S3_layers = trial.suggest_int("num_S3_layers", 1, 3)
    segment_multiplier = trial.suggest_categorical("segment_multiplier", [0.5, 1, 2])
    shuffle_vector_dim = trial.suggest_int("shuffle_vector_dim", 1, 3)

    # Assign the suggested hyperparameters to args or a hyperparameter dictionary
    args.num_segments = num_segments
    args.num_S3_layers = num_S3_layers
    args.segment_multiplier = segment_multiplier
    args.shuffle_vector_dim = shuffle_vector_dim

    # Set the minimum sequence length required for S3 stacking
    # You would probably not need this unless your model has the capability to truncate the time-series during augmentation
    # Then you need to ensure that it has atleast args.min_seq_len number of time steps. 
    args.min_seq_len = max(args.num_segments, args.num_segments * (args.segment_multiplier ** args.num_S3_layers))

    # Log the hyperparameters used for this trial
    hyperparam_dict = {
        "num_segments": args.num_segments, 
        "num_S3_layers": args.num_S3_layers, 
        "segment_multiplier": args.segment_multiplier, 
        "shuffle_vector_dim": args.shuffle_vector_dim
    }
    print(hyperparam_dict)

    # Initialize the device. Make sure to fix the seed here to ensure that for each run the seed is the same.
    # Do not fix the seed outside the objective function.
    # device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    # Implement your own data loading logic based on the task
    print('Loading data... ', end='')
    # train_data, train_labels, val_data, val_labels, test_data, test_labels = load_your_data(args)  # Replace with your own data loading function

    # Baseline configuration
    model_config = {
        "input_dims": train_data.shape[-1], 
        "batch_size": args.batch_size, 
        "lr": args.lr, 
        "output_dims": args.repr_dims, 
        "max_train_length": args.max_train_length
    }

    # S3 configurations to the model configuration (optional)
    S3_config = {
        "enable_S3": args.enable_S3,
        "num_S3_layers": args.num_S3_layers,
        "num_segments": args.num_segments,
        "segment_multiplier": args.segment_multiplier,
        "shuffle_vector_dim": args.shuffle_vector_dim,
        "use_conv_w_avg": args.use_conv_w_avg,
        "min_seq_len": args.min_seq_len
    }

    # Update original config to have S3
    model_config.update(S3_config)

    print("Model configuration:", model_config)

    # Initialize your model and pass the config to it
    model = YourModel(  # Replace 'YourModel' with your own model class
        input_dims=train_data.shape[-1],
        device=device,
        **model_config
    )

    # Train the model (user should replace this with their training code)
    loss_log = model.fit(
        train_data, 
        val_data=val_data,
        n_epochs=args.epochs, 
        n_iters=args.iters, 
        verbose=True
    )

    # Evaluate your model. Get the validation loss for objective.
    eval_res = evaluate_model(model, train_data, val_data, train_labels, val_labels)  # Replace with your evaluation function
    print('Evaluation result:', eval_res)
    validation_loss = eval_res.get("validation_loss")

    # Log extra information in Optuna trial (e.g., test loss, segment order)
    trial.set_user_attr("val_loss", validation_loss)
    trial.set_user_attr("test_loss", eval_res.get("test_loss"))

    # Optionally, also log the segment order for the first S3 layer. This can be modified depending on what you want to visualise.
    trial.set_user_attr("S3_segments_order", model.S3.S3_layers[0].descending_indices.tolist())  # Assume get_S3_order() returns segment order (custom function)

    return validation_loss  # Return the metric to minimize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Your model's arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    

    # S3 related arguments. Arguments such as num_segments, num_S3_layers, segment_multiplier, shuffle_vector_dim will be set using optuna library
    parser.add_argument('--use_conv_w_avg', type=int, default=1, help='Whether to use convolution layer for weighted average.')
    parser.add_argument('--enable_S3', type=int, default=1, help='Whether to use S3 layers.')

    # Arguments related to Optuna
    parser.add_argument('--optuna-db', type=str, default="S3", help='SQLite DB file for Optuna storage.')
    parser.add_argument('--optuna-study-name', type=str, default="your_model+S3", help='Name of the Optuna study.')

    args = parser.parse_args()

    # Set up Optuna for hyperparameter optimization
    search_space = {
        "num_segments": [2, 4, 8, 16, 24],
        "num_S3_layers": [1, 2, 3],
        "segment_multiplier": [0.5, 1, 2],
        "shuffle_vector_dim": [1, 2, 3]
    }
    
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage='sqlite:///' + args.optuna_db,
        study_name=args.optuna_study_name,
        load_if_exists=True
    )

    # Optimize the objective function
    study.optimize(objective, n_trials=len(study.sampler._all_grids))

    # Save the Optuna study and sampler if you want to resume the study later.
    study_file_path = f"optuna_studies/{args.run_name}_study.pkl"
    joblib.dump(study, study_file_path)
    with open(study_file_path.replace("study", "sampler"), "wb") as fout:
        pickle.dump(study.sampler, fout)

    print("Study complete. Best trials:")
    print(study.best_trials)