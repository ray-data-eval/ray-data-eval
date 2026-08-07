export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

python producer_consumer_gpu.py --mem-limit 10 > mem-limit-10.log 2>&1
python producer_consumer_gpu.py --mem-limit 3 > mem-limit-3.log 2>&1
