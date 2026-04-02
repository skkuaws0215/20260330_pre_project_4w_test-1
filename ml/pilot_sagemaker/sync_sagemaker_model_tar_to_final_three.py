"""
Copy SageMaker Training Job model.tar.gz contents into sagemaker_final_three/artifacts/{family}/.

Usage (local tar):
  aws s3 cp s3://bucket/prefix/.../output/model.tar.gz /tmp/model.tar.gz
  python3 ml/pilot_sagemaker/sync_sagemaker_model_tar_to_final_three.py --family xgb --model-tar /tmp/model.tar.gz

Or download from S3 in one step:
  python3 ml/pilot_sagemaker/sagemaker_final_sync.py --family xgb --model-tar-s3 s3://.../model.tar.gz
"""

from sagemaker_final_sync import main

if __name__ == "__main__":
    main()
