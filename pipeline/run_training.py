import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput


def main():
    # Create SageMaker session
    sagemaker_session = sagemaker.Session()

    # IMPORTANT: Use explicit IAM Role ARN (must be a STRING)
    role = "arn:aws:iam::561137843760:role/service-role/codebuild-mlops-fraud-role"

    # Get region
    region = sagemaker_session.boto_region_name

    # Get default S3 bucket
    bucket = sagemaker_session.default_bucket()

    # Upload training data to S3
    train_s3_path = sagemaker_session.upload_data(
        path="data/sample.csv",
        bucket=bucket,
        key_prefix="fraud-detection/data"
    )

    print(f"Training data uploaded to: {train_s3_path}")

    # Get SKLearn container image URI
    image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.large",
    )

    # Define the estimator
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        entry_point="train.py",
        source_dir="src",
        sagemaker_session=sagemaker_session,
    )

    # Define training input (CSV)
    train_input = TrainingInput(
        s3_data=train_s3_path,
        content_type="text/csv"
    )

    # Start training job
    estimator.fit({"train": train_input})

    print("SageMaker training job started successfully")


if __name__ == "__main__":
    main()

