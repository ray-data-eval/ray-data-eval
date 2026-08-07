export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

# python -u producer_consumer_gpu.py --mem-limit 10 > mem-limit-10.log 2>&1
python -u producer_consumer_gpu.py --mem-limit 4 > mem-limit-4.log 2>&1
