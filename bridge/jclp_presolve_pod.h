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
    int32_t           maxSubstLevel;
} CoinPresolvePodView;

/* POD mirror of CoinPostsolveMatrix for B7.
   18 8-byte pointers (144B) + 3 doubles (24B) + 8 int32_t (32B) = 200 bytes. */
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
    const double     *originalRowLower;
    const double     *originalRowUpper;

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

/* Exact mirrors of B7b C++ action structs */
typedef struct TightenActionPod {
    int32_t *rows;
    double  *lbound;
    double  *ubound;
    int32_t  col;
    int32_t  nrows;
    int32_t  direction;
    int32_t  _pad;
} TightenActionPod;

typedef struct ForcingActionPod {
    const int32_t *rowcols;
    const double  *bounds;
    int32_t        row;
    int32_t        nlo;
    int32_t        nup;
    int32_t        _pad;
} ForcingActionPod;

typedef struct SlackActionPod {
    double  clo;
    double  cup;
    double  rlo;
    double  rup;
    double  coeff;
    int32_t col;
    int32_t row;
} SlackActionPod;

typedef struct DualActionPod {
    double  rlo;
    double  rup;
    int32_t ndx;
    int32_t _pad;
} DualActionPod;

/* Exact mirrors of B7c1 C++ action structs */
typedef struct DoubletonActionPod {
    double  clox;
    double  cupx;
    double  costx;
    double  costy;
    double  rlo;
    double  coeffx;
    double  coeffy;
    double *colel;
    int32_t icolx;
    int32_t icoly;
    int32_t row;
    int32_t ncolx;
    int32_t ncoly;
    int32_t _pad;
} DoubletonActionPod;

typedef struct TripletonActionPod {
    int32_t icolx;
    int32_t icolz;
    int32_t row;
    int32_t icoly;
    double  cloy;
    double  cupy;
    double  costy;
    double  clox;
    double  cupx;
    double  costx;
    double  rlo;
    double  rup;
    double  coeffx;
    double  coeffy;
    double  coeffz;
    double *colel;
    int32_t ncolx;
    int32_t ncoly;
} TripletonActionPod;

typedef struct ImpliedFreeActionPod {
    int32_t       row;
    int32_t       col;
    double        clo;
    double        cup;
    double        rlo;
    double        rup;
    const double *rowels;
    const double *costs;
    int32_t       ninrow;
    int32_t       _pad;
} ImpliedFreeActionPod;

/* Exact mirrors of B7c2 C++ action structs */
typedef struct SubstActionPod {
    double       *rlos;
    double       *rups;
    double       *coeffxs;
    int32_t      *rows;
    int32_t      *ninrowxs;
    int32_t      *rowcolsxs;
    double       *rowelsxs;
    const double *costsx;
    int32_t       col;
    int32_t       rowy;
    int32_t       nincol;
    int32_t       _pad;
} SubstActionPod;

typedef struct DupcolActionPod {
    double  thislo;
    double  thisup;
    double  lastlo;
    double  lastup;
    int32_t ithis;
    int32_t ilast;
    double *colels;
    int32_t nincol;
    int32_t _pad;
} DupcolActionPod;

typedef struct GubrowActionPod {
    double   rhs;
    int32_t *deletedRow;
    double  *rowels;
    int32_t *indices;
    int32_t  nDrop;
    int32_t  ninrow;
} GubrowActionPod;

typedef struct TwoxtwoActionPod {
    double  lbound_row;
    double  ubound_row;
    double  lbound_col;
    double  ubound_col;
    double  cost_col;
    double  cost_othercol;
    int32_t row;
    int32_t col;
    int32_t othercol;
    int32_t _pad;
} TwoxtwoActionPod;

#ifdef __cplusplus
}
#endif

#endif /* JCLP_PRESOLVE_POD_H */
