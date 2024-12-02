# for res in 20 #30 #50
# do 
#     for nNeuron in 8 16 32 64 128 256 512
#     do
#         python shallowoga_relu-kernelfit_pw.py --task poisson2D --nNeuron $nNeuron --nTrain 1000 --nTest 500 --device 0 --res $res --sigma 5e-01
#         python shallowoga_relu-kernelfit_pw.py --task helmholtz2D --nNeuron $nNeuron --nTrain 1000 --nTest 500 --device 0 --res $res --sigma 5e-01
#         python shallowoga_relu-kernelfit_pw.py --task cos2D --nNeuron $nNeuron --nTrain 1000 --nTest 500 --device 0 --res $res --sigma 2e-01 --param 2
#     done
# done

# for res in 20 #30 #50
# do 
#     for nNeuron in 8 16 32 64 128 256 512
#     do
#         python shallowoga_relu-kernelfit_pw.py --task poisson2D --nNeuron $nNeuron --nTrain 1000 --nTest 500 --device 0 --res $res --sigma 5e-01
#         python shallowoga_relu-kernelfit_pw.py --task helmholtz2D --nNeuron $nNeuron --nTrain 1000 --nTest 500 --device 0 --res $res --sigma 5e-01
#         python shallowoga_relu-kernelfit_pw.py --task cos2D --nNeuron $nNeuron --nTrain 1000 --nTest 500 --device 0 --res $res --sigma 2e-01 --param 2
#     done
# done

for sigma in 2e-01 5e-01 1e-01
do
    for nTrain in 1000 100 500 
    do
        for nNeuron in 512 8 16 32 64 128 256 
        do
            python shallowoga_relu-kernelfit_pw.py --task poisson2D --nNeuron $nNeuron --nTrain $nTrain --nTest 500 --device 0 --res 20 --sigma $sigma
            python shallowoga_relu-kernelfit_pw.py --task poisson2Dhdomain --nNeuron $nNeuron --nTrain $nTrain --nTest 500 --device 0 --res 30 --sigma $sigma 
            python shallowoga_relu-kernelfit_pw.py --task helmholtz2D --nNeuron $nNeuron --nTrain $nTrain --nTest 500 --device 0 --res 20 --sigma $sigma
            python shallowoga_relu-kernelfit_pw.py --task helmholtz2Dhdomain --nNeuron $nNeuron --nTrain $nTrain --nTest 500 --device 0 --res 30 --sigma $sigma
        done
    done
done

for sigma in 1e-01 2e-01 5e-01 
do
    for nTrain in 100 500 1000
    do
        for param in 1 2 4
        do
            for nNeuron in 8 16 32 64 128 256 512
            do
                python shallowoga_relu-kernelfit_pw.py --task cos2D --nNeuron $nNeuron --nTrain $nTrain --nTest 500 --device 0 --res 20 --sigma $sigma --param $param
                python shallowoga_relu-kernelfit_pw.py --task cos2Dhdomain --nNeuron $nNeuron --nTrain $nTrain --nTest 500 --device 0 --res 30 --sigma $sigma --param $param
            done
        done 
    done 
done 