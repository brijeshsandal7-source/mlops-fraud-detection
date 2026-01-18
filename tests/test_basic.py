import os
import subprocess


def test_training_script_runs():
    """
    This test checks whether the training script
    runs successfully and creates a model file.
    """

    model_dir = "/tmp/test_model"

    # Remove old model if exists
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, f))
    else:
        os.makedirs(model_dir)

    # Run training script
    result = subprocess.run(
        [
            "python",
            "src/train.py",
            "--data-path",
            "data/sample.csv",
            "--model-dir",
            model_dir,
        ],
        capture_output=True,
        text=True,
    )

    # Assert training completed successfully
    assert result.returncode == 0

    # Assert model file exists
    assert os.path.exists(os.path.join(model_dir, "model.joblib"))

