from invoke import Context
from tasks import pipeline

def main():
    # 1. Create a Context (Invoke needs this to execute shell commands)
    c = Context()

    # 2. Define your parameters here (instead of typing them in CLI)
    params = {
        "model": "gradient_boosting",
        "test_size": 0.2,
        "cv": 5,
        "n_iter": 50,  # Increase iterations for better tuning
        "tune": True,
        "wandb": True
    }

    print(f"🚀 Running pipeline from main.py with: {params}")

    # 3. Call the pipeline function directly
    # Note: We pass 'c' as the first argument, just like Invoke does internally
    pipeline(
        c,
        model=params["model"],
        test_size=params["test_size"],
        cv=params["cv"],
        n_iter=params["n_iter"],
        tune=params["tune"],
        wandb=params["wandb"]
    )

if __name__ == "__main__":
    main()