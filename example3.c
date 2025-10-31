/**********************************************************************
Tracks features across *all* sequentially numbered images (img0.pgm,
img1.pgm, img2.pgm, ... until no more files are found). Resizes images 
to a larger size to make computation heavier for profiling.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "pnmio.h"
#include "klt.h"

/* Resize image to larger resolution (simple nearest-neighbor scaling) */
unsigned char* resizeImage(unsigned char* src, int srcCols, int srcRows, 
                           int scale, int* dstCols, int* dstRows) {
    *dstCols = srcCols * scale;
    *dstRows = srcRows * scale;
    unsigned char* dst = (unsigned char*) malloc((*dstCols) * (*dstRows) * sizeof(unsigned char));

    for (int r = 0; r < *dstRows; r++) {
        for (int c = 0; c < *dstCols; c++) {
            int srcR = r / scale;
            int srcC = c / scale;
            dst[r * (*dstCols) + c] = src[srcR * srcCols + srcC];
        }
    }
    return dst;
}

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
    unsigned char *img1, *img2, *resized1, *resized2;
    char fnamein[100], fnameout[100];
    KLT_TrackingContext tc;
    KLT_FeatureList fl;
    KLT_FeatureTable ft;
    int nFeatures = 100;   /* more features since images are larger */
    int nFrames = 0;
    int ncols, nrows, rcols, rrows;
    int scale = 2;         /* resize factor */
    int i;

    tc = KLTCreateTrackingContext();
    fl = KLTCreateFeatureList(nFeatures);

    tc->sequentialMode = TRUE;
    tc->writeInternalImages = FALSE;
    tc->affineConsistencyCheck = -1;

    /* Load first image */
    sprintf(fnamein, "img0.pgm");
    img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
    resized1 = resizeImage(img1, ncols, nrows, scale, &rcols, &rrows);

    /* Prepare feature table, we don’t know nFrames yet so overallocate */
    ft = KLTCreateFeatureTable(1000, nFeatures);  /* large enough upper bound */

    /* Select good features on first image */
    KLTSelectGoodFeatures(tc, resized1, rcols, rrows, fl);
    KLTStoreFeatureList(fl, ft, 0);
    //KLTWriteFeatureListToPPM(fl, resized1, rcols, rrows, "feat0.ppm");
    nFrames = 1;

    /* Process subsequent images until file not found */
    for (i = 1; ; i++) {
        sprintf(fnamein, "img%d.pgm", i);
        FILE* f = fopen(fnamein, "rb");
        if (!f) break;  /* stop when no more files */
        fclose(f);

        img2 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
        resized2 = resizeImage(img2, ncols, nrows, scale, &rcols, &rrows);

        KLTTrackFeatures(tc, resized1, resized2, rcols, rrows, fl);
#ifdef REPLACE
        KLTReplaceLostFeatures(tc, resized2, rcols, rrows, fl);
#endif
        KLTStoreFeatureList(fl, ft, i);

        sprintf(fnameout, "feat%d.ppm", i);
        //KLTWriteFeatureListToPPM(fl, resized2, rcols, rrows, fnameout);

        free(resized1);
        free(img1);
        resized1 = resized2;
        img1 = img2;
        nFrames++;
    }

    /* Write out feature table */
    //KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
    //KLTWriteFeatureTable(ft, "features.ft", NULL);

    KLTFreeFeatureTable(ft);
    KLTFreeFeatureList(fl);
    KLTFreeTrackingContext(tc);
    free(resized1);
    free(img1);

    return 0;
}

