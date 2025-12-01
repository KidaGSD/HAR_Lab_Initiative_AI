import os
import traceback

def init(enable=True, **kwargs):
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
        print(f"W&B init failed: {e}\n{traceback.format_exc()}")
        return None

def log(run, data):
    if run is None:
        return
    try:
        run.log(data)
    except Exception as e:
        print(f"W&B log failed: {e}\n{traceback.format_exc()}")
        safe_finish(run)


def log_confmat(run, y_true, preds, class_names):
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
        run.log({"conf_mat_scenario": cm})
    except Exception as e:
        print(f"W&B confusion matrix log failed: {e}\n{traceback.format_exc()}")
        safe_finish(run)


def safe_finish(run):
    try:
        if run is not None:
            run.finish()
    except Exception as e:
        print(f"W&B finish failed: {e}\n{traceback.format_exc()}")
