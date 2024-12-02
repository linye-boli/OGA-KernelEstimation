for sigma in 5e-01 2e-01
do
    python shallowoga_relu-kernelfit.py --task poisson2D --nNeuron 1024 --nTrain 1000 --nTest 500 --device 1 --res 20 --sigma $sigma
    python shallowoga_relu-kernelfit.py --task helmholtz2D --nNeuron 1024 --nTrain 1000 --nTest 500 --device 1 --res 20 --sigma $sigma
    python shallowoga_relu-kernelfit.py --task poisson2Dhdomain --nNeuron 1024 --nTrain 1000 --nTest 500 --device 1 --res 30 --sigma $sigma
    python shallowoga_relu-kernelfit.py --task helmholtz2Dhdomain --nNeuron 1024 --nTrain 1000 --nTest 500 --device 1 --res 30 --sigma $sigma
done

for param in 1 2 4
do
    for sigma in 5e-01 2e-01
    do
    python shallowoga_relu-kernelfit.py --task cos2D --nNeuron 1024 --nTrain 1000 --nTest 500 --device 1 --res 20 --sigma $sigma --param $param
    done
done