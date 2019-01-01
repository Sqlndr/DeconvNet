for lr in 0.001 
do
	for decay in 0.0001 
	do
		for method in "model1_valid_no_fc_batch_norm_21" 
		do
			cmd="ipython backprop.py -- \
				--iter 50 \
				--epoch 1 \
				--batch-size 32 \
				--pixel 32 \
				--channels 3 \
				--use-cifar 1 \
				--model ${method} \
				--method "Adam" \
				--metric "accuracy,acc,categorical_accuracy"
				--lr=${lr} \
				--decay=${decay}"

			echo $cmd

			$cmd
		done
	done
done

