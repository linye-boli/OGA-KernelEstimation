for sigma in 2e-01 5e-01
do 
    python don.py --task poisson2D --nIter 50000 --nTrain 1000 --nTest 500 --res 20 --sigma $sigma
    python don.py --task poisson2Dhdomain --nIter 50000 --nTrain 1000 --nTest 500 --res 30 --sigma $sigma
    python don.py --task helmholtz2D --nIter 50000 --nTrain 1000 --nTest 500 --res 20 --sigma $sigma
    python don.py --task helmholtz2Dhdomain --nIter 50000 --nTrain 1000 --nTest 500 --res 30 --sigma $sigma

    python greenlearning.py --task poisson2D --nIter 10000 --nTrain 1000 --nTest 500 --device 0 --res 20 --sigma $sigma
    python greenlearning.py --task helmholtz2D --nIter 10000 --nTrain 1000 --nTest 500 --device 0 --res 20 --sigma $sigma
    python greenlearning.py --task poisson2Dhdomain --nIter 10000 --nTrain 1000 --nTest 500 --device 0 --res 30 --sigma $sigma
    python greenlearning.py --task helmholtz2Dhdomain --nIter 10000 --nTrain 1000 --nTest 500 --device 0 --res 30 --sigma $sigma
done

for sigma in 2e-01 1e-01
do 
    python don.py --task cos3D --nIter 50000 --nTrain 1000 --nTest 1000 --res 17 --sigma $sigma --param 2
    python don.py --task logcos3D --nIter 50000 --nTrain 1000 --nTest 1000 --res 17 --sigma $sigma --param 2

    python fno.py --task cos3D --nIter 500 --nTrain 1000 --nTest 1000 --res 17 --sigma $sigma --param 2
    python fno.py --task logcos3D --nIter 500 --nTrain 1000 --nTest 1000 --res 17 --sigma $sigma --param 2
done

