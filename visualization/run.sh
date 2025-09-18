# python visualization/vis.py /export/home/tangzihan/xllm-base/datasets/offline-datasets/jd-offline.jsonl
# python visualization/vis.py /export/home/tangzihan/xllm-base/datasets/online-datasets/jd-online.jsonl

python visualization/vis.py /export/home/tangzihan/xllm-base/datasets/online-datasets/jd-online.jsonl --start-time 50400000 --end-time 54000000
# python visualization/vis.py /export/home/tangzihan/xllm-base/datasets/online-datasets/azure_conv.jsonl

# for ((start_hour=0; start_hour<=$((22)); start_hour+=1))
# do
#     echo $start_time $end_time
#     python visualization/vis.py /export/home/tangzihan/xllm-base/datasets/online-datasets/jd-online.jsonl \
#         --start-time $((start_hour * 3600000)) \
#         --end-time $(((start_hour + 2) * 3600000))
# done