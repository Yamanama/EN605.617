//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void average(__global * buffer, int width)
{
	const int x = get_global_id(0);
    const int y = get_global_id(1);
	uint sum = 0;
	uint count = 0;
	for (int r = 0; r < width; r++)
    {
        const int idxIntmp = (y + r) * width + x;

        for (int c = 0; c < width; c++)
        {
			sum += buffer[(r * width)  + c] * buffer[idxIntmp + c];
			count++;
        }
    } 

	buffer[y * get_global_size(0) + x] = sum/count;
}