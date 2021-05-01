OS=`uname`
MAC="Darwin"
# compile
if [ $OS == $MAC ]; then
    g++ ImageFilter.cpp common/FreeImage/lib/darwin/libfreeimage.a -framework OpenCL -I common/FreeImage/include -o ImageFilter.exe
else
    g++ ImageFilter.cpp common/FreeImage/lib/linux/x86_64/libfreeimage.a -lOpenCL -I common/FreeImage/include -o ImageFilter.exe                                
fi
# get the number of runs from the cli. defaults to 5
# ex ./run.sh 10
if [[ $1 > 0 ]]; then
    end=$1
else
    end=5
fi
# get the delete flag. if 1 will remove intermediate runs
# useful for large runs
# ex ./run.sh 100 1
if [[ $2 ]]; then
    delFlag=1
fi
echo Running $end executions. Picture will get progressively blurry
./ImageFilter.exe lena.bmp lena-filtered-1.bmp
# run test harness
for i in `seq 1 $end`; do
    name=lena-filtered-${i}.bmp
    next=lena-filtered-$((i+1)).bmp
    ./ImageFilter.exe $name $next
    if [[ $delFlag == 1 ]]; then
        rm $name
    fi
done