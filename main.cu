#include <cuda.h>
#include <stdio.h>
#include <time.h>
#define MAX_VALUE 255

__global__ void blackwhite(unsigned char *buffer, unsigned char *outBuf)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int redIdx = i * 3;
    int greenIdx = i * 3 + 1;
    int blueIdx = i * 3 + 2;

    // conversion formula of rgb to gray
    int y = (buffer[redIdx] * 0.3) + (buffer[greenIdx] * 0.59) + (buffer[blueIdx] * 0.11);

    outBuf[redIdx] = y;
    outBuf[blueIdx] = y;
    outBuf[greenIdx] = y;
}

__global__ void sepia(unsigned char *buffer, unsigned char *outBuf)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int redIdx = i * 3;
    int greenIdx = i * 3 + 1;
    int blueIdx = i * 3 + 2;

    // conversion formula of rgb to sepia
    float r = buffer[redIdx];
    float g = buffer[greenIdx];
    float b = buffer[blueIdx];
    outBuf[redIdx] = min(255.0f, r * 0.393f + g * 0.769f + b * 0.189f);
    outBuf[greenIdx] = min(255.0f, r * 0.349f + g * 0.686f + b * 0.168f);
    outBuf[blueIdx] = min(255.0f, r * 0.272f + g * 0.534f + b * 0.131f);
}

__global__ void gaussianBlur(unsigned char *buffer, unsigned char *outBuf, int width, int height, int kernelSize)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int redIdx = i * 3;
    int greenIdx = i * 3 + 1;
    int blueIdx = i * 3 + 2;

    int halfKernelSize = kernelSize / 2;

    float sumRed = 0.0f;
    float sumGreen = 0.0f;
    float sumBlue = 0.0f;

    int count = 0;

    for (int y = -halfKernelSize; y <= halfKernelSize; y++)
    {
        for (int x = -halfKernelSize; x <= halfKernelSize; x++)
        {
            int idx = (i + y * width + x) * 3;

            if (idx >= 0 && idx < width * height * 3)
            {
                sumRed += buffer[idx];
                sumGreen += buffer[idx + 1];
                sumBlue += buffer[idx + 2];
                count++;
            }
        }
    }

    outBuf[redIdx] = sumRed / count;
    outBuf[greenIdx] = sumGreen / count;
    outBuf[blueIdx] = sumBlue / count;
}

__global__ void colourInversion(unsigned char *buffer, unsigned char *outBuf, int width, int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int redIdx = i * 3;
    int greenIdx = i * 3 + 1;
    int blueIdx = i * 3 + 2;

    outBuf[redIdx] = 255 - buffer[redIdx];
    outBuf[greenIdx] = 255 - buffer[greenIdx];
    outBuf[blueIdx] = 255 - buffer[blueIdx];
}

int main()
{
    FILE *fIn = fopen("input.bmp", "rb");    // Input File name
    FILE *fOut = fopen("output.bmp", "wb+"); // Output File name

    unsigned char byte[54];

    if (fIn == NULL) // check if the input file has not been opened succesfully.
    {
        printf("File does not exist.\n");
    }

    for (int i = 0; i < 54; i++) // read the 54 byte header from fIn
    {
        byte[i] = getc(fIn);
    }

    fwrite(byte, sizeof(unsigned char), 54, fOut); // write the header back

    // extract image height, width and bitDepth from imageHeader
    int height = *(int *)&byte[18];
    int width = *(int *)&byte[22];

    int size = height * width; // calculate image size

    unsigned char buffer[size][3]; // to store the image data
    unsigned char outBuf[size][3];

    for (int i = 0; i < size; i++)
    {
        buffer[i][2] = getc(fIn); // blue
        buffer[i][1] = getc(fIn); // green
        buffer[i][0] = getc(fIn); // red
    }

    unsigned char *d_buffer;
    unsigned char *d_outBuf;

    cudaMalloc((void **)&d_buffer, size * 3);
    cudaMalloc((void **)&d_outBuf, size * 3);

    cudaMemcpy(d_buffer, buffer, size * 3, cudaMemcpyHostToDevice);

    // Make this kernel Call Menu Driven with 4 Filters: Black and White, Sepia, Blur, Sharpen

    int choice;
    printf("Enter your choice:\n1. Black and White\n2. Sepia\n3. Blur\n4. Colour Inversion\n");
    scanf("%d", &choice);

    switch (choice)
    {
    case 1:
        blackwhite<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf);
        break;
    case 2:
        sepia<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf);
        break;
    case 3:
        gaussianBlur<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf, width, height, 5);
        break;
    case 4:
        colourInversion<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf, width, height);
        break;
    default:
        printf("Invalid Choice\n");
        return 0;
    }

    // blackwhite<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf);

    cudaMemcpy(outBuf, d_outBuf, size * 3, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        putc(outBuf[i][2], fOut);
        putc(outBuf[i][1], fOut);
        putc(outBuf[i][0], fOut);
    }

    fclose(fOut);
    fclose(fIn);

    return 0;
}