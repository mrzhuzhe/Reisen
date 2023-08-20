#ifndef __PBA_CUDA_H__
#define __PBA_CUDA_H__

// Initialize CUDA and allocate memory
// textureSize is 2^k with k >= 6
extern "C" void pba2DInitialization(int textureSize, int phase1Band); 

// Deallocate memory in GPU
extern "C" void pba2DDeinitialization(); 

// Compute 2D Voronoi diagram
// Input: a 2D texture. Each pixel is represented as two "short" integer. 
//    For each site at (x, y), the pixel at coordinate (x, y) should contain 
//    the pair (x, y). Pixels that are not sites should contain the pair (MARKER, MARKER)
// Output: 2 2D texture. Each pixel is represented as two "short" integer 
//    refering to its nearest site. 
// See original paper for the effect of the three parameters: 
//     phase1Band, phase2Band, phase3Band
// Parameters must divide textureSize
extern "C" void pba2DVoronoiDiagram(short *input, short *output, int phase1Band,
                                    int phase2Band, int phase3Band);
                                    
#define MARKER      -32768

#endif