#!/bin/csh -f
# usage: aaai2025_sow.csh gpu
## dataset: { cifar100, tiny-imagenet-200}
## network: { alexnet_32sow, resnet50_32sow }

unsetenv LANG
set gpu = $argv[1]

foreach seed (0 1 2)
    foreach tasks (1 10 20)
	foreach dataset ('cifar100' 'tiny-imagenet-200')
	    foreach network ('resnet50_32sow' 'alexnet_32sow')
		./aaai2025.csh \
		    --approach sow --dataset $dataset \
		    --network $network --tasks ${tasks} \
		    --seed ${seed} --gpu ${gpu}
		exit
	    end
	end
    end
end
exit
