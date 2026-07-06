# Smoke test the genomic CATE estimators.
# python genomic_cate.py --scenarios 0 1 --noise 0.1 0.2 --subsets 20 30 --include_no_rc --seed 3 --num_steps 100 --output results/smoke_test.npz

# Run the genomic CATE estimators for many experiments.
timestamp=$(date +%Y%m%d%H%M%S)
output_file="results/results_${timestamp}.npz"
python genomic_cate.py --scenarios 0 1 --noise 0.1 0.3 --subsets 1000 --include_no_rc --seed 45 --repeats 10 --output ${output_file}