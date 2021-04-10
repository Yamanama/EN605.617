OS=`uname`
MAC="Darwin"
if [ $OS == $MAC ]; then
    c++ Convolution.cpp -framework OpenCL -o Convolution.exe
else
    g++ -I /usr/include/nvidia-396/ Convolution.cpp -lOpenCL -o Convolution.exe                                 
fi

./Convolution.exe