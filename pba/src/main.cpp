#include <iostream>
#include "pba/pba2D.h"
#include <math.h>
#include <climits>
//#include <ctype.h>

//int fboSize = 2048;
//int nVertices = 100;

int fboSize = 1024;
int nVertices = 100;

// int phase1Band = 32;
// int phase2Band = 32;
int phase1Band = 16;
int phase2Band = 16;
int phase3Band = 2;

typedef struct {
    double totalDistError, maxDistError;
    int errorCount;
} ErrorStatistics;

short *inputPoints, *inputVoronoi, *outputVoronoi;

ErrorStatistics pba;

// Random Point Generator
// Random number generator, obtained from http://oldmill.uchicago.edu/~wilder/Code/random/
unsigned long z, w, jsr, jcong;
void randinit(unsigned long x_)
{ z=x_; w=x_; jsr=x_; jcong = x_;}
unsigned long znew()
{ return (z = 36969 * (z & 0xfffful) + (z >> 16)); }
unsigned long wnew()
{ return (w=18000*(w & 0xfffful) + (w>>16));}
unsigned long MWC()
{ return ((znew() << 16) + wnew()); }
unsigned long SHR3()
{ jsr ^=(jsr << 17); jsr ^= (jsr>>13); return (jsr ^= (jsr << 5)); }
unsigned long CONG()
{ return (jcong = 69069 * jcong + 1234567); }
unsigned long rand_init()
{ return ((MWC()^CONG()) + SHR3()); }
double my_random()
{ return ((double) rand_init() / (double(ULONG_MAX) + 1)); }

// Generate input points
void generateRandomPoints(int width, int height, int nPoints)
{
    int tx, ty;
    randinit(0);
    for (int i = 0; i < width * height* 2ULL; i++){
        inputVoronoi[i] = MARKER;
    }

    for (int i = 0; i < nPoints; i++){
        do {
            tx = int(my_random() * width);
            ty = int(my_random() * height);
        } while (inputVoronoi[(ty * width + tx)*2] != MARKER);
        inputVoronoi[(ty*width+tx)*2] = tx;
        inputVoronoi[(ty*width+tx)*2 + 1] = ty;

        inputPoints[i*2] = tx;
        inputPoints[i*2+1] = ty;
    }
}


// Deinitialization
void deinitialization()
{
    pba2DDeinitialization(); 

    free(inputPoints); 
    free(inputVoronoi); 
    free(outputVoronoi); 
}

// Initialization                                                                           
#define ULL unsigned long long
void initialization()
{
    pba2DInitialization(fboSize, phase1Band); 

    inputPoints     = (short *) malloc((ULL)nVertices * 2ULL * (ULL)sizeof(short)); 
    inputVoronoi    = (short *) malloc((ULL)fboSize * (ULL)fboSize * 2ULL * (ULL)sizeof(short)); 
    outputVoronoi   = (short *) malloc((ULL)fboSize * (ULL)fboSize * 2ULL * (ULL)sizeof(short)); 
}
#undef ULL

void print_mat(short *mat){
    for (int i = 0; i < fboSize; i++) {
        for (int j = 0; j < fboSize; j++) {
            int id = j * fboSize + i; 
            std::cout << " (" <<  mat[id * 2] << "," 
            << mat[id * 2 + 1] << ") "; 
        }
        std::cout << "\n";
    }
}


// Verify the output Voronoi Diagram
void verifyResult(ErrorStatistics *e) 
{
    e->totalDistError = 0.0; 
    e->maxDistError = 0.0; 
    e->errorCount = 0; 

    int tx, ty; 
    double dist, myDist, correctDist, error;

    for (int i = 0; i < fboSize; i++) {
        for (int j = 0; j < fboSize; j++) {
            int id = j * fboSize + i; 

            tx = outputVoronoi[id * 2] - i; 
            ty = outputVoronoi[id * 2 + 1] - j; 
            correctDist = myDist = tx * tx + ty * ty; 

            for (int t = 0; t < nVertices; t++) {
                tx = inputPoints[t * 2] - i; 
                ty = inputPoints[t * 2 + 1] - j; 
                dist = tx * tx + ty * ty; 

                if (dist < correctDist)
                    correctDist = dist; 
            }

            if (correctDist != myDist) {
                error = fabs(sqrt(myDist) - sqrt(correctDist)); 

                e->errorCount++; 
                e->totalDistError += error; 

                if (error > e->maxDistError)
                    e->maxDistError = error; 
            }
        }
    }
}

void printStatistics(ErrorStatistics *e)
{
    double avgDistError = e->totalDistError / e->errorCount; 

    if (e->errorCount == 0)
        avgDistError = 0.0; 

    printf("* Error count           : %i -> %.3f\n", e->errorCount, 
        (double(e->errorCount) / nVertices) * 100.0);
    printf("* Max distance error    : %.5f\n", e->maxDistError);
    printf("* Average distance error: %.5f\n", avgDistError);
}

// Run the tests
void runTests()
{
    generateRandomPoints(fboSize, fboSize, nVertices); 

    //print_mat(inputVoronoi);

    pba2DVoronoiDiagram(inputVoronoi, outputVoronoi, phase1Band, phase2Band, phase3Band); 

    printf("Verifying the result...\n"); 
    verifyResult(&pba);

    printf("-----------------\n");
    printf("Texture: %dx%d\n", fboSize, fboSize);
    printf("Points: %d\n", nVertices);
    printf("-----------------\n");

    printStatistics(&pba); 
}

int main(){    

    initialization();

    runTests();

    deinitialization();

    return 0;
}