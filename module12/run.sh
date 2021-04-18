OS=`uname`
MAC="Darwin"
if [ $OS == $MAC ]; then
    c++ simple.cpp -framework OpenCL -o simple.exe
else
    g++ -I /usr/include/nvidia-396/ simple.cpp -lOpenCL -o simple.exe                                 
fi

./simple.exe