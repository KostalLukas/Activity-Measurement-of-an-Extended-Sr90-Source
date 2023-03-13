# script used to get Geant4 to run despite crashing after saving 1st dataset

n=[number of simulation]

k=`ls Macros_$n | wc -l`
k=$(($k-1))

for i in {0..$k}; do
./lab -b "Macros_$n/macro_$i.mac"
mv "data_0.txt" "Dataset_$n/data_$i.txt"
done
