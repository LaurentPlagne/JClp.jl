#ifndef JCLP_MPS_POD_H
#define JCLP_MPS_POD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CoinHashLinkPod {
    int32_t index;
    int32_t next;
} CoinHashLinkPod;

typedef char *(*jclp_mps_gets_fn)(void *ctx, char *buf, int32_t max_len);
typedef void (*jclp_mps_log_line_fn)(void *ctx, int32_t card_number, const char *card);

/* POD mirror of CoinMpsCardReader state for B6 (`JCLP_BRICK_B6=1`).
   10 8-byte pointers + 1 double + 8 int32_t = 120 bytes, 8-byte aligned. */
typedef struct CoinMpsCardReaderPod {
    char   *card;
    char   *position;
    char   *eol;
    char   *rowName;
    char   *columnName;
    char   *valueString;
    jclp_mps_gets_fn     gets_cb;
    jclp_mps_log_line_fn log_line_cb;
    void   *cb_ctx;
    void   *_pad_ptr;

    double  value;

    int32_t mpsType;
    int32_t section;
    int32_t cardNumber;
    int32_t freeFormat;
    int32_t ieeeFormat;
    int32_t eightChar;
    int32_t stringsAllowed;
    int32_t maxCardLength;
} CoinMpsCardReaderPod;

#ifdef __cplusplus
}
#endif

#endif /* JCLP_MPS_POD_H */
