#ifndef JCLP_PRESOLVE_HELPERS_HPP
#define JCLP_PRESOLVE_HELPERS_HPP

#include <cstdio>
#include <cstdlib>
#include "CoinPresolveMatrix.hpp"
#include "jclp_bridge.h"
#include "jclp_presolve_pod.h"

static inline void jclp_check_presolve_abi()
{
  static bool checked = false;
  if (__builtin_expect(!checked, 0)) {
    using sz_fn_t = int32_t (*)();
    int32_t jl_ps = JCLP_RESOLVE_FN(sz_fn_t, "jclp_ps_pod_sizeof")();
    int32_t jl_post = JCLP_RESOLVE_FN(sz_fn_t, "jclp_post_pod_sizeof")();
    if (jl_ps != static_cast<int32_t>(sizeof(CoinPresolvePodView)) ||
        jl_post != static_cast<int32_t>(sizeof(CoinPostsolvePodView))) {
      fprintf(stderr, "[JClp] FATAL Presolve ABI mismatch: ps C=%zu/Jl=%d post C=%zu/Jl=%d\n",
              sizeof(CoinPresolvePodView), jl_ps, sizeof(CoinPostsolvePodView), jl_post);
      abort();
    }
    checked = true;
  }
}

static inline CoinPresolvePodView jclp_pack_presolve(CoinPresolveMatrix *prob)
{
  CoinPresolvePodView v{};
  v.mcstrt = prob->mcstrt_;
  v.hincol = prob->hincol_;
  v.hrow = prob->hrow_;
  v.colels = prob->colels_;
  v.clink = reinterpret_cast<PresolveHLinkPod *>(prob->clink_);

  v.mrstrt = prob->mrstrt_;
  v.hinrow = prob->hinrow_;
  v.hcol = prob->hcol_;
  v.rowels = prob->rowels_;
  v.rlink = reinterpret_cast<PresolveHLinkPod *>(prob->rlink_);

  v.clo = prob->clo_;
  v.cup = prob->cup_;
  v.rlo = prob->rlo_;
  v.rup = prob->rup_;
  v.cost = prob->cost_;
  v.sol = prob->sol_;
  v.acts = prob->acts_;
  v.rowduals = prob->rowduals_;
  v.rcosts = prob->rcosts_;
  v.colstat = prob->colstat_;
  v.rowstat = prob->rowstat_;
  v.integerType = prob->integerType_;

  v.originalColumn = prob->originalColumn_;
  v.originalRow = prob->originalRow_;

  v.colChanged = prob->colChanged_;
  v.rowChanged = prob->rowChanged_;
  v.colsToDo = prob->colsToDo_;
  v.nextColsToDo = prob->nextColsToDo_;
  v.rowsToDo = prob->rowsToDo_;
  v.nextRowsToDo = prob->nextRowsToDo_;

  v.usefulRowInt = prob->usefulRowInt_;
  v.usefulColumnInt = prob->usefulColumnInt_;
  v.infiniteUp = prob->infiniteUp_;
  v.infiniteDown = prob->infiniteDown_;
  v.usefulRowDouble = prob->usefulRowDouble_;
  v.usefulColumnDouble = prob->usefulColumnDouble_;
  v.randomNumber = prob->randomNumber_;
  v.sumUp = prob->sumUp_;
  v.sumDown = prob->sumDown_;

  v.dobias = prob->dobias_;
  v.maxmin = prob->maxmin_;
  v.ztolzb = prob->ztolzb_;
  v.ztoldj = prob->ztoldj_;
  v.feasibilityTolerance = prob->feasibilityTolerance_;

  v.nrows = prob->nrows_;
  v.ncols = prob->ncols_;
  v.nrows0 = prob->nrows0_;
  v.ncols0 = prob->ncols0_;
  v.nelems = prob->nelems_;
  v.bulk0 = prob->bulk0_;
  v.status = prob->status_;
  v.presolveOptions = prob->presolveOptions_;
  v.anyProhibited = prob->anyProhibited_ ? 1 : 0;
  v.anyInteger = prob->anyInteger_ ? 1 : 0;
  v.numberColsToDo = prob->numberColsToDo_;
  v.numberNextColsToDo = prob->numberNextColsToDo_;
  v.numberRowsToDo = prob->numberRowsToDo_;
  v.numberNextRowsToDo = prob->numberNextRowsToDo_;
  v.pass = prob->pass_;
  v.maxSubstLevel = prob->maxSubstLevel_;
  return v;
}

static inline void jclp_unpack_presolve(CoinPresolveMatrix *prob, const CoinPresolvePodView &v)
{
  prob->dobias_ = v.dobias;
  prob->nrows_ = v.nrows;
  prob->ncols_ = v.ncols;
  prob->nelems_ = v.nelems;
  prob->status_ = v.status;
  prob->numberColsToDo_ = v.numberColsToDo;
  prob->numberNextColsToDo_ = v.numberNextColsToDo;
  prob->numberRowsToDo_ = v.numberRowsToDo;
  prob->numberNextRowsToDo_ = v.numberNextRowsToDo;
}

static inline CoinPostsolvePodView jclp_pack_postsolve(CoinPostsolveMatrix *prob)
{
  CoinPostsolvePodView v{};
  v.mcstrt = prob->mcstrt_;
  v.hincol = prob->hincol_;
  v.hrow = prob->hrow_;
  v.link = prob->link_;
  v.colels = prob->colels_;

  v.clo = prob->clo_;
  v.cup = prob->cup_;
  v.rlo = prob->rlo_;
  v.rup = prob->rup_;
  v.cost = prob->cost_;
  v.sol = prob->sol_;
  v.acts = prob->acts_;
  v.rowduals = prob->rowduals_;
  v.rcosts = prob->rcosts_;
  v.colstat = prob->colstat_;
  v.rowstat = prob->rowstat_;
  v.originalRowLower = prob->originalRowLower_;
  v.originalRowUpper = prob->originalRowUpper_;

  v.maxmin = prob->maxmin_;
  v.ztolzb = prob->ztolzb_;
  v.ztoldj = prob->ztoldj_;

  v.nrows = prob->nrows_;
  v.ncols = prob->ncols_;
  v.nrows0 = prob->nrows0_;
  v.ncols0 = prob->ncols0_;
  v.nelems = prob->nelems_;
  v.bulk0 = prob->bulk0_;
  v.free_list = prob->free_list_;
  return v;
}

static inline void jclp_unpack_postsolve(CoinPostsolveMatrix *prob, const CoinPostsolvePodView &v)
{
  prob->nrows_ = v.nrows;
  prob->ncols_ = v.ncols;
  prob->free_list_ = v.free_list;
}

#endif /* JCLP_PRESOLVE_HELPERS_HPP */
