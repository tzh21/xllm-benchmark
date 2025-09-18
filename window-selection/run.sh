dataset_name=jd-online

python ./window-selection/select.py /export/home/tangzihan/xllm-base/datasets/online-datasets/$dataset_name.jsonl \
    --verbose \
    2>&1 | tee local/window/$dataset_name.txt