#!/bin/csh -f
# usage: aaai2026_fc.csh gpu
## dataset: { cifar100, tiny-imagenet-200}
## network: { resnet50_32 }

unsetenv LANG
set gpu = $argv[1]

foreach seed (0 1 2)
    foreach tasks (10 20)
	foreach dataset ('cifar100' 'tiny-imagenet-200')
	    foreach network ('resnet50_32' 'alexnet_32')
		./aaai2026.csh \
		    --approach finetuning --dataset $dataset \
		    --network $network --tasks ${tasks} \
		    --seed ${seed} --gpu ${gpu}
		exit
	    end
	end
    end
end
exit
