# bash benchmark-jd/search-qps.sh v0 s code 0 1 3.5 4.0 & sleep 3
# bash benchmark-jd/search-qps.sh v0 s code 2 3 4.5 5.0 & sleep 3
# bash benchmark-jd/search-qps.sh v1 s code 4 5 3.5 4.0 & sleep 3
# bash benchmark-jd/search-qps.sh v1 s code 6 7 4.5 5.0 & sleep 3
# bash benchmark-jd/search-qps.sh v2 s code 8 9 3.5 4.0 & sleep 3
# bash benchmark-jd/search-qps.sh v2 s code 10 11 4.5 5.0 & sleep 3

# qps_range1=(0.5 1.0 1.5 2.0 2.5 3.0)
# qps_range2=(3.5 4.0 4.5 5.0 5.5 6.0)

# qps_range1=(0.0 0.5)
# qps_range2=(3.0 3.5)
qps_range1=(2.5 3.0)
qps_range2=(5.5 6.0)

bash benchmark-jd/search-qps.sh v0 s conv 0 1 "${qps_range1[@]}" & sleep 3
bash benchmark-jd/search-qps.sh v0 s conv 2 3 "${qps_range2[@]}" & sleep 3
bash benchmark-jd/search-qps.sh v1 s conv 4 5 "${qps_range1[@]}" & sleep 3
bash benchmark-jd/search-qps.sh v1 s conv 6 7 "${qps_range2[@]}" & sleep 3
bash benchmark-jd/search-qps.sh v2 s conv 8 9 "${qps_range1[@]}" & sleep 3
bash benchmark-jd/search-qps.sh v2 s conv 10 11 "${qps_range2[@]}" & sleep 3
