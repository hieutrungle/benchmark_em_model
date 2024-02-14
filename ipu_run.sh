echo "poprun --vv --num-instances=2 --num-replicas=2 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 1 --num_ipus 1"
poprun --vv --num-instances=2 --num-replicas=2 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 1 --num_ipus 1

echo "poprun --vv --num-instances=4 --num-replicas=4 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 1 --num_ipus 1"
poprun --vv --num-instances=4 --num-replicas=4 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 1 --num_ipus 1

echo "poprun --vv --num-instances=8 --num-replicas=8 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 1 --num_ipus 1"
poprun --vv --num-instances=8 --num-replicas=8 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 1 --num_ipus 1

echo "poprun --vv --num-instances=8 --num-replicas=16 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 1 --num_ipus 1"
poprun --vv --num-instances=8 --num-replicas=16 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 1 --num_ipus 1

echo "poprun --vv --num-instances=1 --num-replicas=1 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2"
poprun --vv --num-instances=1 --num-replicas=1 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2

echo "poprun --vv --num-instances=2 --num-replicas=2 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2"
poprun --vv --num-instances=2 --num-replicas=2 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2

echo "poprun --vv --num-instances=4 --num-replicas=4 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2"
poprun --vv --num-instances=4 --num-replicas=4 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2

echo "poprun --vv --num-instances=8 --num-replicas=8 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2"
poprun --vv --num-instances=8 --num-replicas=8 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2

echo "poprun --vv --num-instances=8 --num-replicas=16 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2"
poprun --vv --num-instances=8 --num-replicas=16 python main.py --device ipu --batch_size 10 --device_iterations 20 --replication_factor 1 --gradient_accumulation 3 --num_ipus 2