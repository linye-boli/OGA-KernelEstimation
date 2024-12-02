for sigma in 1e-01 2e-01
do
    for nTrain in 1000
    do
        for param in 2
        do
            for nNeuron in 512 256 8 16 32 64 128
            do
                python shallowoga_relu-kernelfit_pw.py --task cos3D --nNeuron $nNeuron --nTrain $nTrain --nTest 1000 --device 1 --res 17 --sigma $sigma --param $param
                python shallowoga_relu-kernelfit_pw.py --task logcos3D --nNeuron $nNeuron --nTrain $nTrain --nTest 1000 --device 1 --res 17 --sigma $sigma --param $param
            done 
        done 
    done 
done 



# python shallowoga_relu-kernelfit_pw.py --task poisson3D --nNeuron $nNeuron --nTrain $nTrain --nTest 1000 --device 0 --res 17 --sigma 1e-01