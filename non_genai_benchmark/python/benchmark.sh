# This is an example bash script to run sync_benchmark.py in a loop and log results in a file.
# Not intended to run as-is, but as a starting point. This example assumes that both this bash script and sync_benchmark.py
# are in a directory with several OpenVINO models in IR format (.xml/.bin files) and benchmarks should be run on all the models,
# on CPU and GPU

# If you get an "unexpected end of file" error, the file was probably modified on Windows
# run dos2unix benchmark.sh to fix that

# Run this script with `source benchmark.sh` from a Python virtual environment with OpenVINO installed

lscpu > log.txt

# curl -O https://raw.githubusercontent.com/helena-intel/snippets/refs/heads/main/show_properties/show_compiled_model_properties.py
# python show_compiled_model_properties.py model_path >> log.txt

for model in *.xml;
do
    for i in {1..5}; do python sync_benchmark.py $model CPU --log log.csv; done
    for i in {1..5}; do python sync_benchmark.py $model GPU --log log.csv; done

    # Run once with performance counters enabled. Do not log, since performance counters itself may add some overhead
    python sync_benchmark.py $model ALL -pc
done
