# Smoke test the array GCM.
#python array_gcm.py --n_experiments 2 --domains "(5,10);(6,12)" --output array_results/smoke_test_results.pkl --svi_num_steps 5 --svi_step_size 0.0005

# Run the array GCM.
timestamp=$(date +%Y%m%d%H%M%S)
output_file="array_results/results_${timestamp}.pkl"
python array_gcm.py --n_experiments 10 --domains "(100,25);(200,50)" --output ${output_file} --svi_num_steps 100000 --svi_step_size 0.0005 --seed 40