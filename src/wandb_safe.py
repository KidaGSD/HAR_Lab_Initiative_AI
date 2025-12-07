"""
Safe WandB wrapper that handles errors gracefully without crashing training.
"""
import os
import traceback

# Global experiment group (set via environment or auto-generated)
_EXPERIMENT_GROUP = None


def set_group(group_name):
    """Set the experiment group for all subsequent runs."""
    global _EXPERIMENT_GROUP
    _EXPERIMENT_GROUP = group_name
    print(f"W&B group set to: {group_name}")


def get_group():
    """Get current experiment group."""
    global _EXPERIMENT_GROUP
    if _EXPERIMENT_GROUP is None:
        # Check environment variable
        _EXPERIMENT_GROUP = os.environ.get('WANDB_RUN_GROUP', None)
    return _EXPERIMENT_GROUP


def init(enable=True, **kwargs):
    """Initialize WandB run with error handling.
    
    Automatically adds group if set via set_group() or WANDB_RUN_GROUP env var.
    """
    if not enable:
        print("W&B disabled; proceeding without logging.")
        return None
    try:
        import wandb
    except ImportError:
        print("W&B not installed; proceeding without logging.")
        return None
    
    # Auto-add group if set and not already specified
    group = get_group()
    if group and 'group' not in kwargs:
        kwargs['group'] = group
        print(f"W&B run grouped under: {group}")
    
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
