"""
Safe WandB wrapper that handles errors gracefully without crashing training.
"""
import traceback


def init(enable=True, **kwargs):
    """Initialize WandB run with error handling."""
    if not enable:
        print("W&B disabled; proceeding without logging.")
        return None
    try:
        import wandb
    except ImportError:
        print("W&B not installed; proceeding without logging.")
        return None
    try:
        return wandb.init(**kwargs)
    except Exception as e:
        print(f"W&B init failed: {e}")
        return None


def log(run, data):
    """Log metrics to WandB with error handling."""
    if run is None:
        return
    try:
        run.log(data)
    except Exception as e:
        print(f"W&B log failed: {e}")
        # Don't finish run on log failure - just skip this log


def log_summary(run, data):
    """Log summary metrics to WandB run.summary with error handling."""
    if run is None:
        return
    try:
        for key, value in data.items():
            run.summary[key] = value
    except Exception as e:
        print(f"W&B summary log failed: {e}")


def log_confmat(run, y_true, preds, class_names, key="conf_mat_scenario"):
    """Log confusion matrix to WandB with error handling.

    Args:
        run: WandB run object
        y_true: Ground truth labels
        preds: Predicted labels
        class_names: List of class names
        key: Key name for the confusion matrix (default: "conf_mat_scenario")
    """
    if run is None:
        return
    try:
        import wandb
        cm = wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=preds,
            class_names=class_names
        )
        run.log({key: cm})
    except Exception as e:
        print(f"W&B confusion matrix log failed: {e}")
        # Don't crash training if confusion matrix fails


def safe_finish(run):
    """Safely finish WandB run with error handling."""
    if run is None:
        return
    try:
        run.finish()
    except Exception as e:
        print(f"W&B finish failed: {e}")
