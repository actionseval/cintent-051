#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h> // For malloc, free, rand, RAND_MAX

// Assuming walker.h would contain these or similar definitions.
// If walker.h is more comprehensive, these might be redundant or need adjustment.

#ifndef WALKER_H_INTERNAL_DEFS
#define WALKER_H_INTERNAL_DEFS

#define RAND_MID (RAND_MAX / 2.0)

// Generic min/max macros if not provided by a common header or walker.h
#ifndef max
    #define max(a,b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
    #define min(a,b) ((a) < (b) ? (a) : (b))
#endif

enum BorderStatus {
    INSIDE,
    TOP,
    BOTTOM,
    LEFT,
    RIGHT,
    FRONT, // Added from 3D
    BACK   // Added from 3D
};

#endif // WALKER_H_INTERNAL_DEFS


#ifdef _WIN32
    #define WALKER_EXPORT __declspec(dllexport)
#else
    #define WALKER_EXPORT __attribute__((visibility ("default")))
#endif

// Typedefs from walker2d.c
typedef struct {
    int* voidMap;
    int* solidMap;
    int height;
    int width;
    double minLen;
    double maxLen;
    double minAngle;
    double maxAngle;
} HeatMap2D;

// Typedefs from walker3d.c
typedef struct {
    int* voidMap;
    int* solidMap;
    int layers;
    int height;
    int width;
    double minLen;
    double maxLen;
    double minTheta;
    double maxTheta;
    double minPhi;
    double maxPhi;
} HeatMap3D;

// Functions from walker2d.c
void printBoolImage2D(bool* image, int height, int width) {
    for (int i = 0; i < width * height; ++i) {
        printf("%d ", image[i]);
        if (i > 0 && (i + 1) % width == 0) {
            printf("\n");
        }
    }
}

void printDoubleImage2D(double* image, int height, int width) {
    for (int i = 0; i < width * height; ++i) {
        printf("%lf ", image[i]);
        if (i > 0 && (i + 1) % width == 0) {
            printf("\n");
        }
    }
}

enum BorderStatus getBorderStatus2D(double posRow, double posCol, int height, int width) {
    if (posRow < 0.0) {
        return TOP;
    } else if (posRow >= (double)height) {
        return BOTTOM;
    } else if (posCol < 0.0) {
        return LEFT;
    } else if (posCol >= (double)width) {
        return RIGHT;
    } else {
        return INSIDE;
    }
}

void getStartPos2D(double* distances, int height, int width, double* startRow, double* startCol) {
    for (int i = height * width / 2; i < height * width; ++i) {
        if (distances[i] < 2.0 && distances[i] != 0.0 && distances[i] > 0.0) { // Corrected: distances[i] (was distances[i] && distances[i])
            *startRow = (double)(i / width); // Corrected: original was i / height, assuming row-major
            *startCol = (double)(i % width);
            return;
        }
    }
    // Fallback if no suitable start found (e.g., start at center or corner)
    *startRow = height / 2.0;
    *startCol = width / 2.0;
     printf("Warning: No suitable starting wall found in getStartPos2D. Defaulting to center.\n");
}

void getStepSizes2D(double* stepRow, double* stepCol) {
    *stepRow = (double)rand() - RAND_MID;
    *stepCol = (double)rand() - RAND_MID;

    while (*stepCol == 0.0 || *stepRow == 0.0) {
        if (*stepCol == 0.0) {
            *stepCol = (double)rand() - RAND_MID;
        }
        if (*stepRow == 0.0) { // Changed else to if, to ensure both are checked again if one was zero
            *stepRow = (double)rand() - RAND_MID;
        }
    }

    if (fabs(*stepCol) > fabs(*stepRow)) {
        *stepRow = (*stepRow / fabs(*stepCol));
        *stepCol = *stepCol > 0.0 ? 1.0 : -1.0;
    } else {
        *stepCol = (*stepCol / fabs(*stepRow));
        *stepRow = *stepRow > 0.0 ? 1.0 : -1.0;
    }
}

void updateMap2D(HeatMap2D* heatMap, double voidLen, double solidLen, double angle)
{
    double minLen = heatMap->minLen;
    double maxLen = heatMap->maxLen;
    double minAngle = heatMap->minAngle;
    double maxAngle = heatMap->maxAngle;
    int mapHeight = heatMap->height;
    int mapWidth = heatMap->width;

    // Ensure maxLen - minLen is not zero to avoid division by zero
    double lenRange = maxLen - minLen;
    if (lenRange == 0) lenRange = 1.0; // Avoid division by zero, effectively scaling to mapHeight-1 or 0

    double angleRange = maxAngle - minAngle;
    if (angleRange == 0) angleRange = 1.0; // Avoid division by zero

    int posRowVoid = (int)((min(max(voidLen, minLen), maxLen) - minLen) / lenRange * (mapHeight - 1));
    int posRowSolid = (int)((min(max(solidLen, minLen), maxLen) - minLen) / lenRange * (mapHeight - 1));
    int posCol = (int)((min(max(angle, minAngle), maxAngle) - minAngle) / angleRange * (mapWidth - 1));
    
    // Clamp values to be safe
    posRowVoid = max(0, min(posRowVoid, mapHeight - 1));
    posRowSolid = max(0, min(posRowSolid, mapHeight - 1));
    posCol = max(0, min(posCol, mapWidth - 1));


#ifdef DEBUG
    printf("angle: %lf\n", angle);
    printf("posCol: %d\n", posCol);
    printf("voidLen: %lf\n", voidLen);
    printf("posRowVoid %d\n", posRowVoid);
    printf("solidLen: %lf\n", solidLen);
    printf("posRowSolid %d\n", posRowSolid);
#endif

    if(posRowVoid * mapWidth + posCol < mapHeight * mapWidth && posRowVoid * mapWidth + posCol >=0)
        ++(heatMap->voidMap[posRowVoid * mapWidth + posCol]);
    if (solidLen > 0.0) {
         if(posRowSolid * mapWidth + posCol < mapHeight * mapWidth && posRowSolid * mapWidth + posCol >=0)
            ++(heatMap->solidMap[posRowSolid * mapWidth + posCol]);
    }
}

int getContPos2D(double posRow, double posCol, int height, int width) {
    // Clamping to ensure valid array access before conversion to int
    int r = (int)floor(posRow);
    int c = (int)floor(posCol);
    r = max(0, min(r, height - 1));
    c = max(0, min(c, width - 1));
    return r * width + c;
}

WALKER_EXPORT
HeatMap2D* createHeatMap2D(int* voidMap, int* solidMap, int height, int width,
                           double minLen, double maxLen, double minAngle, double maxAngle)
{
    HeatMap2D* heatMap = (HeatMap2D*)malloc(sizeof(HeatMap2D));
    if (!heatMap) return NULL; // Allocation check
    heatMap->voidMap = voidMap;
    heatMap->solidMap = solidMap;
    heatMap->height = height;
    heatMap->width = width;
    heatMap->minLen = minLen;
    heatMap->maxLen = maxLen;
    heatMap->minAngle = minAngle;
    heatMap->maxAngle = maxAngle;

    return heatMap;
}

WALKER_EXPORT
void destroyHeatMap2D(HeatMap2D* heatMap) {
    free(heatMap);
}

WALKER_EXPORT
void walk2D(bool* image, int height, int width, double* distances, int iterations, HeatMap2D* heatMap, int* path)
{
    srand(time(NULL));

    double posRow = 0.0;
    double posCol = 0.0;

    getStartPos2D(distances, height, width, &posRow, &posCol);

    printf("Start row: %lf\n", posRow);
    printf("Start col: %lf\n", posCol);

    time_t startTime = clock();
    for (int i = 0; i < iterations; ++i) {
        double stepRow = 0.0;
        double stepCol = 0.0;

        getStepSizes2D(&stepRow, &stepCol);

        double solidSteps = 0.0;
        double voidSteps = 0.0;
        double lastStepSize = 0.0;
        bool lastPhase = false; // True when walker is in void and moving towards a wall
        enum BorderStatus borderStatus; 
        
        // Ensure initial pos is valid
        posRow = max(0.0, min(posRow, (double)height - 1.0));
        posCol = max(0.0, min(posCol, (double)width - 1.0));
        int contPos = getContPos2D(posRow, posCol, height, width);

        // The loop condition implies we start INSIDE an object (image[contPos] is false)
        // or we start in void (image[contPos] is true) and look for a wall.
        // The original getStartPos2D tries to find a wall (distance < 2.0), suggesting starting near or on a wall.
        // If starting on a wall pixel (image[contPos] is false), lastPhase should be false.
        // If starting in void (image[contPos] is true), lastPhase should be true.
        lastPhase = image[contPos];


        int iter_count_inner = 0; // Safety break for inner loop
        const int MAX_INNER_ITER = height * width * 2; // Heuristic limit

        while (iter_count_inner < MAX_INNER_ITER) {
            iter_count_inner++;
            contPos = getContPos2D(posRow, posCol, height, width);

            if (lastPhase && !image[contPos]) { // Transitioned from void to solid, meaning we hit the target wall
                break; 
            }
            if (!lastPhase && image[contPos]) { // Transitioned from solid to void, ready for lastPhase
                lastPhase = true;
            }

            lastStepSize = max(distances[contPos], 1.0);
            
            double nextPosRow = posRow + lastStepSize * stepRow;
            double nextPosCol = posCol + lastStepSize * stepCol;

            borderStatus = getBorderStatus2D(nextPosRow, nextPosCol, height, width);
            
            if (borderStatus != INSIDE) {
                // Hit border, reflect
                // Simplified: just reverse component, don't try to place exactly on border
                if (borderStatus == LEFT || borderStatus == RIGHT) {
                    stepCol *= -1.0;
                } else { // TOP or BOTTOM
                    stepRow *= -1.0;
                }
                // Do not update posRow, posCol, try new direction from current spot
                // voidSteps and solidSteps are not incremented for border hits if we don't move.
                // Or, alternatively, treat border as a wall and break or count step.
                // For now, just reflects and continues from same spot.
                // If stuck reflecting, outer loop (iterations) will eventually finish.
                // Or add a small random perturbation to stepRow/stepCol here.
                lastPhase = false; // Reset phase as we are changing direction significantly
                solidSteps += 1.0; // Count border hit as a "solid" interaction for length calculation
                continue; // Try new direction
            }

            // Move to next position
            posRow = nextPosRow;
            posCol = nextPosCol;
            contPos = getContPos2D(posRow, posCol, height, width); // Update contPos after move

            if (image[contPos]) { // In void
                if (path) path[contPos] += 1;
                voidSteps += lastStepSize;
                // lastPhase is true (or becomes true if just entered void)
            } else { // In solid
                if (path) path[contPos] -= 1; // Or some other marker for solid path
                solidSteps += lastStepSize; // Accumulate length travelled in solid
                // lastPhase remains false if we were already in solid, or becomes false.
                // If we are looking for the void-to-solid transition, this means we are still in solid.
            }
        }
        if(iter_count_inner >= MAX_INNER_ITER){
            // Inner loop timed out, could mean stuck. For now, just proceed to next iteration.
            // printf("Warning: Inner loop max iterations reached for iteration %d.\n", i);
        }


        // Position is now on the wall (image[contPos] is false) after being in void (lastPhase was true)
        // Or, if starting in solid, position is on the first void pixel encountered.
        // The logic implies we want the length of the void ray *before* hitting the wall.
        // And solid length is travel *through* solids.

        #ifdef DEBUG
        printf("---Iteration %d---\n", i);
        printf("Void Steps (distance units): %lf\n", voidSteps);
        printf("Solid Steps (distance units): %lf\n", solidSteps);
        printf("Final pos: (Row: %lf), (Col: %lf)\n", posRow, posCol); // Using %lf for double
        printf("stepRow: %lf\n", stepRow);
        printf("stepCol: %lf\n", stepCol);
        #endif

        // Hypotenuse here is based on normalized step vectors, representing direction.
        // The voidSteps and solidSteps already incorporate distance from `distances` array.
        // So, actual length is already voidSteps / solidSteps if distances are actual lengths.
        // If distances are multipliers, then this calculation is more complex.
        // Assuming stepRow/Col define a unit direction vector for the segment for angle calculation,
        // and voidSteps/solidSteps are the lengths.
        
        // The original code calculates voidLen/solidLen by multiplying steps by hypotenuse.
        // This seems to imply that voidSteps/solidSteps are number of "unit" steps, not total distance.
        // However, lastStepSize = max(distances[contPos], 1.0) suggests distances are used.
        // Let's assume voidSteps and solidSteps are accumulated actual path lengths.
        double voidLen = voidSteps;
        double solidLen = solidSteps;
        double angle = atan2(stepRow, stepCol); // atan2 is generally better for angles from components.
        updateMap2D(heatMap, voidLen, solidLen, angle);
    }
    time_t endTime = clock();

    printf("End row: %lf\n", posRow);
    printf("End col: %lf\n", posCol);
    printf("Time elapsed for 2D walk %d iterations: %lfms\n", iterations, (double)(endTime - startTime) * 1000.0 / CLOCKS_PER_SEC);
}


// Functions from walker3d.c

void printBoolImage3D(bool* image, int layers, int height, int width) { // Order changed to L, H, W for consistency
    for (int l = 0; l < layers; ++l) {
        printf("Layer %d:\n", l);
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                printf("%d ", image[l * height * width + r * width + c]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// Note: getBorderStatus3D uses (double height, double width, double layers) parameters,
// but typically these are int. Changed to int for consistency.
enum BorderStatus getBorderStatus3D(double posRow, double posCol, double posLayer, int height, int width, int layers) {
    if (posLayer < 0.0) {
        return FRONT; // Assuming negative layer is "in front"
    } else if (posLayer >= (double)layers) {
        return BACK;
    } else if (posRow < 0.0) {
        return TOP;
    } else if (posRow >= (double)height) {
        return BOTTOM;
    } else if (posCol < 0.0) {
        return LEFT;
    } else if (posCol >= (double)width) {
        return RIGHT;
    } else {
        return INSIDE;
    }
}

void getStartPos3D(double* distances, int layers, int height, int width, double* startLayer, double* startRow, double* startCol) {
    // Search from the middle outwards
    for (int i = layers * height * width / 2; i < layers * height * width; ++i) {
        if (distances[i] < 2.0 && distances[i] != 0.0 && distances[i] > 0.0) { // Corrected condition
            *startLayer = (double)(i / (height * width));
            int remainder = i % (height * width);
            *startRow = (double)(remainder / width);
            *startCol = (double)(remainder % width);
            return;
        }
    }
    // Fallback if no suitable start found
    *startLayer = layers / 2.0;
    *startRow = height / 2.0;
    *startCol = width / 2.0;
    printf("Warning: No suitable starting wall found in getStartPos3D. Defaulting to center.\n");
}

void getStepSizes3D(double* stepRow, double* stepCol, double* stepLayer) {
    *stepRow = (double)rand() - RAND_MID;
    *stepCol = (double)rand() - RAND_MID;
    *stepLayer = (double)rand() - RAND_MID;

    while (*stepCol == 0.0 || *stepRow == 0.0 || *stepLayer == 0.0) {
        if (*stepCol == 0.0) *stepCol = (double)rand() - RAND_MID;
        if (*stepRow == 0.0) *stepRow = (double)rand() - RAND_MID;
        if (*stepLayer == 0.0) *stepLayer = (double)rand() - RAND_MID;
    }

    double maxVal = fabs(*stepRow);
    if (fabs(*stepCol) > maxVal) maxVal = fabs(*stepCol);
    if (fabs(*stepLayer) > maxVal) maxVal = fabs(*stepLayer);

    // Normalize so the largest component is 1 or -1
    if (maxVal != 0) { // Avoid division by zero if all somehow still zero (though loop prevents this)
        *stepRow /= maxVal;
        *stepCol /= maxVal;
        *stepLayer /= maxVal;
    }
}

void updateMap3D(HeatMap3D* heatMap, double voidLen, double solidLen, double theta, double phi) {
    int mapLayers = heatMap->layers;
    int mapHeight = heatMap->height;
    int mapWidth = heatMap->width;
    double minLen = heatMap->minLen;
    double maxLen = heatMap->maxLen;
    double minTheta = heatMap->minTheta;
    double maxTheta = heatMap->maxTheta;
    double minPhi = heatMap->minPhi;
    double maxPhi = heatMap->maxPhi;

    double lenRange = maxLen - minLen;
    if (lenRange == 0) lenRange = 1.0;
    double thetaRange = maxTheta - minTheta;
    if (thetaRange == 0) thetaRange = 1.0;
    double phiRange = maxPhi - minPhi;
    if (phiRange == 0) phiRange = 1.0;


    int posLayerVoid = (int)((min(max(voidLen, minLen), maxLen) - minLen) / lenRange * (mapLayers - 1));
    int posLayerSolid = (int)((min(max(solidLen, minLen), maxLen) - minLen) / lenRange * (mapLayers - 1));
    int posRowTheta = (int)((min(max(theta, minTheta), maxTheta) - minTheta) / thetaRange * (mapHeight - 1)); // Assuming theta maps to mapHeight
    int posColPhi = (int)((min(max(phi, minPhi), maxPhi) - minPhi) / phiRange * (mapWidth - 1));      // Assuming phi maps to mapWidth

    // Clamp values
    posLayerVoid = max(0, min(posLayerVoid, mapLayers - 1));
    posLayerSolid = max(0, min(posLayerSolid, mapLayers - 1));
    posRowTheta = max(0, min(posRowTheta, mapHeight - 1));
    posColPhi = max(0, min(posColPhi, mapWidth - 1));
    
    long long voidIdx = (long long)posLayerVoid * mapWidth * mapHeight + (long long)posRowTheta * mapWidth + posColPhi;
    long long solidIdx = (long long)posLayerSolid * mapWidth * mapHeight + (long long)posRowTheta * mapWidth + posColPhi;
    long long mapTotalSize = (long long)mapLayers * mapHeight * mapWidth;


    if(voidIdx >= 0 && voidIdx < mapTotalSize)
        ++(heatMap->voidMap[voidIdx]);

    if (solidLen > 0.0) {
        if(solidIdx >=0 && solidIdx < mapTotalSize)
            ++(heatMap->solidMap[solidIdx]);
    }
}

int getContPos3D(double posLayer, double posRow, double posCol, int layers, int height, int width) {
    // Clamping to ensure valid array access before conversion to int
    int l = (int)floor(posLayer);
    int r = (int)floor(posRow);
    int c = (int)floor(posCol);

    l = max(0, min(l, layers - 1));
    r = max(0, min(r, height - 1));
    c = max(0, min(c, width - 1));
    
    return l * height * width + r * width + c;
}


WALKER_EXPORT
HeatMap3D* createHeatMap3D(int* voidMap, int* solidMap, int layers, int height, int width, double minLen,
                           double maxLen, double minTheta, double maxTheta, double minPhi, double maxPhi)
{
    HeatMap3D* heatMap = (HeatMap3D*)malloc(sizeof(HeatMap3D));
    if (!heatMap) return NULL;
    heatMap->voidMap = voidMap;
    heatMap->solidMap = solidMap;
    heatMap->layers = layers;
    heatMap->height = height;
    heatMap->width = width;
    heatMap->minLen = minLen;
    heatMap->maxLen = maxLen;
    heatMap->minTheta = minTheta;
    heatMap->maxTheta = maxTheta;
    heatMap->minPhi = minPhi;
    heatMap->maxPhi = maxPhi;

    return heatMap;
}

WALKER_EXPORT
void destroyHeatMap3D(HeatMap3D* heatMap) {
    free(heatMap);
}

WALKER_EXPORT
void walk3D(bool* image, int layers, int height, int width, double* distances, int iterations, HeatMap3D* heatMap, int* path)
{
    srand(time(NULL)); // Consider seeding once globally if called frequently

    double posLayer = 0.0;
    double posRow = 0.0;
    double posCol = 0.0;

    getStartPos3D(distances, layers, height, width, &posLayer, &posRow, &posCol);

    printf("Start layer: %lf\n", posLayer);
    printf("Start row: %lf\n", posRow);
    printf("Start col: %lf\n", posCol);

    time_t startTime = clock();
    for (int i = 0; i < iterations; ++i) {
        double stepLayer = 0.0;
        double stepRow = 0.0;
        double stepCol = 0.0;

        getStepSizes3D(&stepRow, &stepCol, &stepLayer);

        double solidSteps = 0.0; // Accumulated length in solid
        double voidSteps = 0.0;  // Accumulated length in void
        double lastStepSize = 0.0;
        bool lastPhase = false; // True if current phase is moving through void towards a solid wall
        enum BorderStatus borderStatus;

        // Ensure initial pos is valid
        posLayer = max(0.0, min(posLayer, (double)layers - 1.0));
        posRow = max(0.0, min(posRow, (double)height - 1.0));
        posCol = max(0.0, min(posCol, (double)width - 1.0));
        int contPos = getContPos3D(posLayer, posRow, posCol, layers, height, width);
        
        lastPhase = image[contPos]; // if start in void, lastPhase=true. If in solid, lastPhase=false.

        int iter_count_inner = 0;
        const int MAX_INNER_ITER = layers * height * width * 2; // Heuristic limit

        while (iter_count_inner < MAX_INNER_ITER) {
            iter_count_inner++;
            contPos = getContPos3D(posLayer, posRow, posCol, layers, height, width);

            if (lastPhase && !image[contPos]) { // Was in void, hit solid: this is the wall we were looking for.
                break;
            }
            if (!lastPhase && image[contPos]) { // Was in solid, entered void: now start "lastPhase".
                lastPhase = true;
            }
            
            lastStepSize = max(distances[contPos], 1.0);

            double nextPosLayer = posLayer + lastStepSize * stepLayer;
            double nextPosRow = posRow + lastStepSize * stepRow;
            double nextPosCol = posCol + lastStepSize * stepCol;

            borderStatus = getBorderStatus3D(nextPosRow, nextPosCol, nextPosLayer, height, width, layers);

            if (borderStatus != INSIDE) {
                // Hit border, reflect component perpendicular to border
                if (borderStatus == LEFT || borderStatus == RIGHT) {
                    stepCol *= -1.0;
                } else if (borderStatus == TOP || borderStatus == BOTTOM) {
                    stepRow *= -1.0;
                } else { // FRONT or BACK
                    stepLayer *= -1.0;
                }
                lastPhase = false; // Direction changed significantly
                solidSteps += 1.0; // Count border hit as a "solid" interaction for length calculation
                continue; // Try new direction from current spot
            }
            
            // Move to next position
            posLayer = nextPosLayer;
            posRow = nextPosRow;
            posCol = nextPosCol;
            contPos = getContPos3D(posLayer, posRow, posCol, layers, height, width); // Update contPos

            if (image[contPos]) { // In void
                if (path) path[contPos] += 1;
                voidSteps += lastStepSize;
            } else { // In solid
                if (path) path[contPos] -= 1; 
                solidSteps += lastStepSize;
            }
        }
         if(iter_count_inner >= MAX_INNER_ITER){
            // printf("Warning: Inner loop max iterations reached for 3D iteration %d.\n", i);
        }

        // Position is now on the wall (image[contPos] is false)
        // voidSteps is the length of the ray in void until hitting this wall.
        // solidSteps is the length of travel through solid parts (if any before reaching void).

        #ifdef DEBUG
        printf("---Iteration %d---\n", i);
        printf("Void Steps (distance units): %lf\n", voidSteps);
        printf("Solid Steps (distance units): %lf\n", solidSteps);
        printf("Final pos: (Lay: %lf), (Row: %lf), (Col: %lf)\n", posLayer, posRow, posCol);
        printf("stepLayer: %lf, stepRow: %lf, stepCol: %lf\n", stepLayer, stepRow, stepCol);
        #endif

        double voidLen = voidSteps;
        double solidLen = solidSteps;
        
        // Spherical coordinates:
        // theta (polar angle, from Z-axis): atan2(sqrt(x^2+y^2), z)
        // phi (azimuthal angle, from X-axis in XY plane): atan2(y, x)
        // Mapping steps to (x,y,z): stepCol -> x, stepRow -> y, stepLayer -> z (arbitrary but consistent)
        double r_xy = sqrt(stepCol * stepCol + stepRow * stepRow);
        double theta = atan2(r_xy, stepLayer); // Angle from Z-axis (stepLayer)
        double phi = atan2(stepRow, stepCol);   // Angle in XY plane (stepRow vs stepCol)

        updateMap3D(heatMap, voidLen, solidLen, theta, phi);
    }
    time_t endTime = clock();

    printf("End layer: %lf\n", posLayer);
    printf("End row: %lf\n", posRow);
    printf("End col: %lf\n", posCol);
    printf("Time elapsed for 3D walk %d iterations: %lfms\n", iterations, (double)(endTime - startTime) * 1000.0 / CLOCKS_PER_SEC);
}
