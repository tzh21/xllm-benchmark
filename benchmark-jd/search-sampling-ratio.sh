# 可以运行 40 次实验

for ((ratio=3; ratio<=10; ratio+=1))
do
    echo "Sampling ratio: $ratio"
    ./benchmark-jd/run.sh $ratio
done

sleep 10