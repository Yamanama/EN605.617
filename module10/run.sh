OS=`uname`
MAC="Darwin"
if [ $OS == $MAC ]; then
    c++ HelloWorld.cpp -framework OpenCL -o HelloWorld.exe
else
    c++ -I /usr/include/nvidia-396/ HelloWorld.cpp -lOpenCL                                  
fi

./HelloWorld.exe