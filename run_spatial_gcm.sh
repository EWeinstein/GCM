# Smoke test the spatial GCM.
#python spatial_gcm.py --n_experiments 2 --domains "(0.5,0.5);(1.0,1.0)" --output spatial_results/smoke_test_results.pkl --num_warmup 5 --num_samples 5 --svi_init True --svi_num_steps 5

# Run the full spatial GCM experiments.
timestamp=$(date +%Y%m%d%H%M%S)
output_file="spatial_results/results_${timestamp}.pkl"
python spatial_gcm.py --n_experiments 10 --domains "(0.5,0.5);(1.0,1.0)" --output ${output_file} --num_warmup 100 --num_samples 500 --svi_init True --svi_num_steps 40000 --svi_step_size 0.00005