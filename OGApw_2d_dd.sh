for nNeuron in 8 16 32 64 128 256 512
do
    for sigma_train in 1e-01 2e-01 5e-01
    do
        for sigma_test in 1e-01 2e-01 5e-01
        do
        python shallowoga_relu-kernelfit_pw_dd.py --task poisson2D --nNeuron $nNeuron --nTrain 1000 --nTest 500 --device 1 --res 20 --sigma_train $sigma_train --sigma_test $sigma_test
        done
    done
done
