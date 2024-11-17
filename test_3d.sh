for sigma in 1e-01 2e-01 5e-01 
do
    for nTrain in 500 1000 3000
    do
        for param in 1 2 4
        do
            for nNeuron in 8 16 32 64 128 256 512
            do
                python shallowoga_relu-kernelfit_pw.py --task cos3D --nNeuron $nNeuron --nTrain $nTrain --nTest 1000 --device 0 --res 17 --sigma $sigma --param $param --mode sub
            done
        done 
    done 
done 