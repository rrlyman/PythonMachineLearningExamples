#!/bin/bash  

if [ ! -d "/tmp/plots" ]; then
	mkdir /tmp/plots
fi
echo "" > /tmp/plots/run_batch.txt        
for i in $(ls -1v ./[o-q]*.py ); do
	echo "" |& tee -a /tmp/plots/run_batch.txt
	echo "##############################################################" |& tee -a /tmp/plots/run_batch.txt
	echo "$i  ###############################" |& tee -a /tmp/plots/run_batch.txt
	echo "##############################################################" |& tee -a /tmp/plots/run_batch.txt
	echo "" |& tee -a /tmp/plots/run_batch.txt
   	python3 $i |& tee -a /tmp/plots/run_batch.txt 

done

        