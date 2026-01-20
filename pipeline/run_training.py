import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput


def main():
    """
    This script triggers a SageMaker training job.
    It uploads local data to S3 and tells SageMaker
    to run src/train.py on a managed instance.
    """

    # Create a SageMaker session
    sagemaker_session = sagemaker.Session()

    # Get execution role (used by SageMaker to access AWS resources)
    role = sagemaker.get_execution_role()

    # Use the default SageMaker S3 bucket
    bucket = sagemaker_session.default_bucket()

    # Upload training data to S3
    train_s3_path = sagemaker_session.upload_data(
        path="data/sample.csv",
        bucket=bucket,
        key_prefix="fraud-detection/data"
    )

    print(f"Training data uploaded to: {train_s3_path}")

    # Define the SKLearn estimator
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
    )

    # Explicitly define training input as CSV
    train_input = TrainingInput(
        s3_data=train_s3_path,
        content_type="text/csv"
    )

    # Start the training job
    estimator.fit({"train": train_input})

    print("SageMaker training job started successfully")


if __name__ == "__main__":
    main()

