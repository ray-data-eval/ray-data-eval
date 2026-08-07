export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

# High memory
# python -u producer_consumer_gpu.py --mem-limit 10 > mem-limit-10.log 2>&1

# Low memory
python -u producer_consumer_gpu.py --mem-limit 4 > mem-limit-4.log 2>&1