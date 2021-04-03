OS=`uname`
MAC="Darwin"
if [ $OS == $MAC ]; then
    c++ HelloWorld.cpp -framework OpenCL -o HelloWorld.exe
else
    g++ -I /usr/include/nvidia-396/ HelloWorld.cpp -lOpenCL -o HelloWorld.exe                                 
fi

./HelloWorld.exe