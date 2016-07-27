#!/bin/bash  

if [ ! -d "./plots" ]; then
	mkdir ./plots
fi
echo "" > ./plots/run_batch.txt        
for i in $(ls -1v ./[p-q]*.py ); do
	echo "" |& tee -a ./plots/run_batch.txt
	echo "##############################################################" |& tee -a ./plots/run_batch.txt
	echo "$i  ###############################" |& tee -a ./plots/run_batch.txt
	echo "##############################################################" |& tee -a ./plots/run_batch.txt
	echo "" |& tee -a ./plots/run_batch.txt
   	python3 $i |& tee -a ./plots/run_batch.txt 

done

        