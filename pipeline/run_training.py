import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput


def main():
    # SageMaker session
    sagemaker_session = sagemaker.Session()

    # Explicit IAM Role ARN
    role = "arn:aws:iam::561137843760:role/service-role/codebuild-mlops-fraud-role"

    # Default bucket
    bucket = sagemaker_session.default_bucket()

    # Upload training data
    train_s3_path = sagemaker_session.upload_data(
        path="data/sample.csv",
        bucket=bucket,
        key_prefix="fraud-detection/data"
    )

    print(f"Training data uploaded to: {train_s3_path}")

    # SKLearn Estimator (stable & simple)
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

    # CSV input
    train_input = TrainingInput(
        s3_data=train_s3_path,
        content_type="text/csv"
    )

    # Start training
    estimator.fit({"train": train_input})

    print("SageMaker training job started successfully")


if __name__ == "__main__":
    main()

