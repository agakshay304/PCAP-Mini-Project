#include <cuda.h>
#include <stdio.h>
#include <time.h>
#define MAX_VALUE 255

__global__ void brightness(unsigned char *buffer, unsigned char *outBuf, int brightness)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int redIdx = i * 3;
    int greenIdx = i * 3 + 1;
    int blueIdx = i * 3 + 2;

    int r = buffer[redIdx] + brightness;
    int g = buffer[greenIdx] + brightness;
    int b = buffer[blueIdx] + brightness;

    // clamp values to the valid range of [0, 255]
    r = min(max(r, 0), 255);
    g = min(max(g, 0), 255);
    b = min(max(b, 0), 255);

    outBuf[redIdx] = r;
    outBuf[greenIdx] = g;
    outBuf[blueIdx] = b;
}

__global__ void contrast(unsigned char *buffer, unsigned char *outBuf, float contrast)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int redIdx = i * 3;
    int greenIdx = i * 3 + 1;
    int blueIdx = i * 3 + 2;

    float r = ((float)buffer[redIdx] / 255.0 - 0.5) * contrast + 0.5;
    float g = ((float)buffer[greenIdx] / 255.0 - 0.5) * contrast + 0.5;
    float b = ((float)buffer[blueIdx] / 255.0 - 0.5) * contrast + 0.5;

    // clamp values to the valid range of [0, 255]
    r = min(max(r, 0.0f), 1.0f) * 255.0f;
    g = min(max(g, 0.0f), 1.0f) * 255.0f;
    b = min(max(b, 0.0f), 1.0f) * 255.0f;

    outBuf[redIdx] = (unsigned char)r;
    outBuf[greenIdx] = (unsigned char)g;
    outBuf[blueIdx] = (unsigned char)b;
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
    FILE *fIn = fopen("snail.bmp", "rb");    // Input File name
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

    // Make this kernel Call Menu Driven with 4 Filters: 1. Colour Inversion 2. Blur 3. Adjust Contrast 4.Adjust Brightness

    int choice;
    printf("Enter your choice:\n 1. Colour Inversion\n 2. Blur\n 3. Adjust Contrast\n 4.Adjust Brightness\n");
    scanf("%d", &choice);

    switch (choice)
    {
    case 1:
        colourInversion<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf, width, height);
        break;
    case 2:
        gaussianBlur<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf, width, height, 5);
        break;
    case 3:
        int cont;
        printf("Enter contrast value: (Negative for decreasing Contrast, Positive for increasing Contrast\n)");
        scanf("%d", &cont);
        contrast<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf, cont);
        break;
    case 4:
        int bright;
        printf("Enter brightness value: (Negative for darker, Positive for brighter\n)");
        scanf("%d", &bright);
        brightness<<<ceil(size + 1024 - 1) / 1024, 1024>>>(d_buffer, d_outBuf, bright);
        break;
    default:
        printf("Invalid Choice\n");
        return 0;
    }

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