export RAY_DEDUP_LOGS=0 # Important to disable Ray log deduplication
python ray_data_pipeline_map.py --source s3 > video_inference_s3_g5_xlarge_batch_32.out 2>&1
