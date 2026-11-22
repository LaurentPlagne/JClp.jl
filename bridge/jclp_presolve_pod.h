#ifndef JCLP_PRESOLVE_POD_H
#define JCLP_PRESOLVE_POD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PresolveHLinkPod {
    int32_t pre;
    int32_t suc;
} PresolveHLinkPod;

/* POD mirror of CoinPresolveMatrix for B7 (B7a/B7b/B7c).
   39 8-byte pointers (312B) + 5 doubles (40B) + 16 int32_t (64B) = 416 bytes. */
typedef struct CoinPresolvePodView {
    /* Column-major */
    int32_t          *mcstrt;
    int32_t          *hincol;
    int32_t          *hrow;
    double           *colels;
    PresolveHLinkPod *clink;

    /* Row-major */
    int32_t          *mrstrt;
    int32_t          *hinrow;
    int32_t          *hcol;
    double           *rowels;
    PresolveHLinkPod *rlink;

    /* Rim & solution */
    double           *clo;
    double           *cup;
    double           *rlo;
    double           *rup;
    double           *cost;
    double           *sol;
    double           *acts;
    double           *rowduals;
    double           *rcosts;
    uint8_t          *colstat;
    uint8_t          *rowstat;
    uint8_t          *integerType;

    /* Original index maps */
    int32_t          *originalColumn;
    int32_t          *originalRow;

    /* Change & ToDo tracking */
    uint8_t          *colChanged;
    uint8_t          *rowChanged;
    int32_t          *colsToDo;
    int32_t          *nextColsToDo;
    int32_t          *rowsToDo;
    int32_t          *nextRowsToDo;

    /* Scratch work arrays */
    int32_t          *usefulRowInt;
    int32_t          *usefulColumnInt;
    int32_t          *infiniteUp;
    int32_t          *infiniteDown;
    double           *usefulRowDouble;
    double           *usefulColumnDouble;
    double           *randomNumber;
    double           *sumUp;
    double           *sumDown;

    /* 5 doubles */
    double            dobias;
    double            maxmin;
    double            ztolzb;
    double            ztoldj;
    double            feasibilityTolerance;

    /* 16 int32_t scalars */
    int32_t           nrows;
    int32_t           ncols;
    int32_t           nrows0;
    int32_t           ncols0;
    int32_t           nelems;
    int32_t           bulk0;
    int32_t           status;
    int32_t           presolveOptions;
    int32_t           anyProhibited;
    int32_t           anyInteger;
    int32_t           numberColsToDo;
    int32_t           numberNextColsToDo;
    int32_t           numberRowsToDo;
    int32_t           numberNextRowsToDo;
    int32_t           pass;
    int32_t           _pad;
} CoinPresolvePodView;

/* POD mirror of CoinPostsolveMatrix for B7.
   16 8-byte pointers (128B) + 3 doubles (24B) + 8 int32_t (32B) = 184 bytes. */
typedef struct CoinPostsolvePodView {
    int32_t          *mcstrt;
    int32_t          *hincol;
    int32_t          *hrow;
    int32_t          *link;
    double           *colels;

    double           *clo;
    double           *cup;
    double           *rlo;
    double           *rup;
    double           *cost;
    double           *sol;
    double           *acts;
    double           *rowduals;
    double           *rcosts;
    uint8_t          *colstat;
    uint8_t          *rowstat;

    double            maxmin;
    double            ztolzb;
    double            ztoldj;

    int32_t           nrows;
    int32_t           ncols;
    int32_t           nrows0;
    int32_t           ncols0;
    int32_t           nelems;
    int32_t           bulk0;
    int32_t           free_list;
    int32_t           _pad;
} CoinPostsolvePodView;

/* Exact mirrors of B7a C++ action structs */
typedef struct DroppedZeroPod {
    int32_t row;
    int32_t col;
} DroppedZeroPod;

typedef struct DropEmptyColActionPod {
    double  clo;
    double  cup;
    double  cost;
    double  sol;
    int32_t jcol;
    int32_t _pad;
} DropEmptyColActionPod;

typedef struct DropEmptyRowActionPod {
    double  rlo;
    double  rup;
    int32_t row;
    int32_t fill_row;
} DropEmptyRowActionPod;

typedef struct UselessActionPod {
    double        rlo;
    double        rup;
    const int32_t *rowcols;
    const double  *rowels;
    int32_t       row;
    int32_t       ninrow;
} UselessActionPod;

typedef struct RemoveFixedActionPod {
    int32_t col;
    int32_t start;
    double  sol;
} RemoveFixedActionPod;

typedef struct MakeFixedActionPod {
    double  bound;
    int32_t col;
    int32_t _pad;
} MakeFixedActionPod;

#ifdef __cplusplus
}
#endif

#endif /* JCLP_PRESOLVE_POD_H */
