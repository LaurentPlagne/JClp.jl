#ifndef JCLP_CF_POD_H
#define JCLP_CF_POD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* POD view of CoinIndexedVector buffers for zero-copy C++ <-> Julia calls. */
typedef struct CoinIndexedVecPod {
    double  *elements;
    int32_t *indices;
    int32_t  n_elements;
    int32_t  capacity;
    int32_t  packed_mode;
    int32_t  _pad;
} CoinIndexedVecPod;

/* POD view of CoinFactorization arrays & scalars for B5a (BTRAN + FTRAN) & B5b.
   All 8-byte fields come first, followed by 4-byte int32_t fields, guaranteeing
   identical struct layout in C/C++ and Julia on 64-bit platforms. */
typedef struct CoinFactorizationPodView {
    /* --- 8-byte pointers --- */
    double  *elementU;
    int32_t *indexRowU;
    int32_t *startColumnU;
    int32_t *numberInColumn;
    int32_t *numberInColumnPlus;
    int32_t *startRowU;
    int32_t *numberInRow;
    int32_t *indexColumnU;
    int32_t *convertRowToColumnU;

    double  *elementL;
    int32_t *indexRowL;
    int32_t *startColumnL;
    double  *elementByRowL;
    int32_t *indexColumnL;
    int32_t *startRowL;

    double  *elementR;
    int32_t *indexRowR;
    int32_t *startColumnR;       /* raw startColumnRArray_ (NOT shifted by -numberRows_) */

    double  *pivotRegion;
    int32_t *pivotColumn;        /* pivotColumnArray_ */
    int32_t *pivotColumnBack;    /* pivotColumnBack() */
    int32_t *permute;            /* permuteArray_ */
    int32_t *permuteBack;        /* permuteBackArray_ */
    int32_t *nextColumn;         /* nextColumnArray_ */
    int32_t *lastColumn;         /* lastColumnArray_ */

    int32_t *sparseArray;        /* sparseArray_ (work area for DFS/bitmask) */
    double  *denseAreaAddress;   /* denseAreaAddress_ */
    int32_t *densePermute;       /* densePermute_ */

    /* Mutable statistics pointers */
    int32_t *p_numberBtranCounts;
    double  *p_btranCountInput;
    double  *p_btranCountAfterU;
    double  *p_btranCountAfterR;
    double  *p_btranCountAfterL;

    int32_t *p_numberFtranCounts;
    double  *p_ftranCountInput;
    double  *p_ftranCountAfterL;
    double  *p_ftranCountAfterR;
    double  *p_ftranCountAfterU;

    /* Mutable R/U write-back fields */
    int32_t *p_lengthR;
    int32_t *p_numberR;

    /* --- 8-byte doubles --- */
    double   zeroTolerance;
    double   slackValue;
    double   btranAverageAfterU;
    double   btranAverageAfterR;
    double   btranAverageAfterL;
    double   ftranAverageAfterL;
    double   ftranAverageAfterR;
    double   ftranAverageAfterU;

    /* --- 4-byte int32_t scalars (even count for 8-byte alignment) --- */
    int32_t  numberRows;
    int32_t  numberRowsExtra;
    int32_t  maximumRowsExtra;
    int32_t  numberColumnsExtra;
    int32_t  maximumColumnsExtra;
    int32_t  numberGoodU;
    int32_t  numberU;
    int32_t  numberL;
    int32_t  baseL;
    int32_t  numberR;
    int32_t  lengthR;
    int32_t  lengthAreaR;
    int32_t  lengthAreaU;
    int32_t  numberDense;
    int32_t  numberSlacks;
    int32_t  numberPivots;
    int32_t  maximumPivots;
    int32_t  sparseThreshold;
    int32_t  sparseThreshold2;
    int32_t  doForrestTomlin;
    int32_t  collectStatistics;
    int32_t  _pad;
} CoinFactorizationPodView;

/* Mutable state passed alongside CoinFactorizationPodView for B5b (replaceColumn). */
typedef struct CoinFactorizationReplaceState {
    /* 8-byte pointers */
    int32_t *nextRow;
    int32_t *lastRow;
    /* 8-byte double */
    double   relaxCheck;
    /* 4-byte in/out scalars (14 fields -> 56 bytes, multiple of 8) */
    int32_t  numberRowsExtra;
    int32_t  numberColumnsExtra;
    int32_t  numberU;
    int32_t  numberGoodU;
    int32_t  numberPivots;
    int32_t  numberR;
    int32_t  lengthR;
    int32_t  lengthU;
    int32_t  maximumU;
    int32_t  totalElements;
    int32_t  factorElements;
    int32_t  lengthAreaL;
    int32_t  numberCompressions;
    int32_t  status;
    int32_t  dropNumberInColumnPlus;
    int32_t  _pad;
} CoinFactorizationReplaceState;

#ifdef __cplusplus
}
#endif

#endif /* JCLP_CF_POD_H */
