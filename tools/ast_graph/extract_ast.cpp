#include <clang-c/Index.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

const std::string kClpSrcPrefix = "/Users/laurentplagne/Projects/Clp/src/";
const std::string kCoinUtilsSrcPrefix =
    "/Users/laurentplagne/Projects/Clp/parity/deps/CoinUtils/src/";
const std::string kInstallIncludePrefix =
    "/Users/laurentplagne/Projects/Clp/parity/install/include/coin-or/";

std::string toStdString(CXString s) {
  const char *cstr = clang_getCString(s);
  std::string out = cstr ? cstr : "";
  clang_disposeString(s);
  return out;
}

std::string sanitize(std::string s) {
  for (char &ch : s) {
    if (ch == '\t' || ch == '\n' || ch == '\r') {
      ch = ' ';
    }
  }
  return s;
}

struct FileInfo {
  bool keep = false;
  std::string repo;
  std::string norm_path;
};

struct NodeRecord {
  std::string usr;
  std::string kind;
  std::string repo;
  std::string name;
  std::string qualname;
  std::string owner_usr;
  std::string file;
  unsigned line_start = 0;
  unsigned line_end = 0;
  unsigned loc = 0;
  int is_def = 0;
  int is_virtual = 0;
  std::string signature;
  std::string decl_file;
};

struct EdgeKey {
  std::string src_usr;
  std::string dst_usr;
  std::string edge_kind;

  bool operator<(const EdgeKey &o) const {
    return std::tie(src_usr, dst_usr, edge_kind) <
           std::tie(o.src_usr, o.dst_usr, o.edge_kind);
  }
};

struct EdgeVal {
  int count = 0;
  int src_is_def = 0;
};

class Extractor {
public:
  explicit Extractor(CXTranslationUnit tu) : tu_(tu) {}

  void run() {
    CXCursor root = clang_getTranslationUnitCursor(tu_);
    visitDeclScope(root);
  }

  const std::unordered_map<std::string, NodeRecord> &nodes() const {
    return nodes_;
  }
  const std::map<EdgeKey, EdgeVal> &edges() const { return edges_; }

private:
  CXTranslationUnit tu_;
  std::unordered_map<CXFile, FileInfo> file_cache_;
  std::unordered_map<std::string, NodeRecord> nodes_;
  std::map<EdgeKey, EdgeVal> edges_;

  static bool startsWith(const std::string &s, const std::string &prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
  }

  FileInfo classifyFile(CXFile file) {
    if (!file) {
      return {};
    }
    auto it = file_cache_.find(file);
    if (it != file_cache_.end()) {
      return it->second;
    }

    FileInfo info;
    std::string raw = toStdString(clang_getFileName(file));
    if (!raw.empty()) {
      std::string norm;
      try {
        norm = fs::weakly_canonical(fs::path(raw)).string();
      } catch (...) {
        norm = fs::path(raw).lexically_normal().string();
      }

      if (startsWith(norm, kClpSrcPrefix)) {
        info.keep = true;
        info.repo = "clp";
        info.norm_path = "Clp/src/" + norm.substr(kClpSrcPrefix.size());
      } else if (startsWith(norm, kCoinUtilsSrcPrefix)) {
        info.keep = true;
        info.repo = "coinutils";
        info.norm_path =
            "CoinUtils/src/" + norm.substr(kCoinUtilsSrcPrefix.size());
      } else if (startsWith(norm, kInstallIncludePrefix)) {
        std::string rel = norm.substr(kInstallIncludePrefix.size());
        std::error_code ec;
        if (fs::exists(kCoinUtilsSrcPrefix + rel, ec)) {
          info.keep = true;
          info.repo = "coinutils";
          info.norm_path = "CoinUtils/src/" + rel;
        } else if (fs::exists(kClpSrcPrefix + rel, ec)) {
          info.keep = true;
          info.repo = "clp";
          info.norm_path = "Clp/src/" + rel;
        }
      }
    }

    file_cache_.emplace(file, info);
    return info;
  }

  FileInfo getCursorFileInfo(CXCursor c) {
    CXSourceLocation loc = clang_getCursorLocation(c);
    CXFile file = nullptr;
    unsigned line = 0, col = 0, off = 0;
    clang_getSpellingLocation(loc, &file, &line, &col, &off);
    return classifyFile(file);
  }

  // Unwrap implicit template specializations to the primary template cursor
  // when available, then canonicalize.
  CXCursor normalizeTargetCursor(CXCursor c) {
    if (clang_Cursor_isNull(c)) {
      return clang_getNullCursor();
    }
    CXCursor tmpl = clang_getSpecializedCursorTemplate(c);
    if (!clang_Cursor_isNull(tmpl)) {
      c = tmpl;
    }
    return clang_getCanonicalCursor(c);
  }

  std::string getUSR(CXCursor c) {
    if (clang_Cursor_isNull(c)) {
      return "";
    }
    return sanitize(toStdString(clang_getCursorUSR(c)));
  }

  static bool isTypeKindStr(const std::string &k) {
    return k == "class" || k == "struct" || k == "enum" || k == "typedef";
  }

  static bool isCallableKindStr(const std::string &k) {
    return k == "method" || k == "ctor" || k == "dtor" || k == "function";
  }

  std::string mapCursorKind(CXCursor c) {
    CXCursorKind k = clang_getCursorKind(c);
    switch (k) {
    case CXCursor_ClassDecl:
      return "class";
    case CXCursor_StructDecl:
    case CXCursor_UnionDecl:
      return "struct";
    case CXCursor_ClassTemplate:
    case CXCursor_ClassTemplatePartialSpecialization: {
      CXCursorKind tk = clang_getTemplateCursorKind(c);
      if (tk == CXCursor_StructDecl || tk == CXCursor_UnionDecl) {
        return "struct";
      }
      return "class";
    }
    case CXCursor_EnumDecl:
      return "enum";
    case CXCursor_TypedefDecl:
    case CXCursor_TypeAliasDecl:
    case CXCursor_TypeAliasTemplateDecl:
      return "typedef";
    case CXCursor_CXXMethod:
    case CXCursor_ConversionFunction:
      return "method";
    case CXCursor_Constructor:
      return "ctor";
    case CXCursor_Destructor:
      return "dtor";
    case CXCursor_FunctionDecl:
      return "function";
    case CXCursor_FunctionTemplate: {
      CXCursor parent = clang_getCursorSemanticParent(c);
      CXCursorKind pk = clang_getCursorKind(parent);
      if (pk == CXCursor_ClassDecl || pk == CXCursor_StructDecl ||
          pk == CXCursor_UnionDecl || pk == CXCursor_ClassTemplate ||
          pk == CXCursor_ClassTemplatePartialSpecialization) {
        return "method";
      }
      return "function";
    }
    default:
      return "";
    }
  }

  std::string buildQualName(CXCursor c) {
    std::vector<std::string> parts;
    std::string self_name = sanitize(toStdString(clang_getCursorSpelling(c)));
    if (self_name.empty()) {
      std::string kind = mapCursorKind(c);
      self_name = "(anon " + (kind.empty() ? "decl" : kind) + ")";
    }
    parts.push_back(self_name);

    CXCursor p = clang_getCursorSemanticParent(c);
    int guard = 0;
    while (!clang_Cursor_isNull(p) && guard++ < 32) {
      CXCursorKind pk = clang_getCursorKind(p);
      if (pk == CXCursor_TranslationUnit || pk == CXCursor_UnexposedDecl) {
        break;
      }
      if (pk == CXCursor_Namespace || pk == CXCursor_ClassDecl ||
          pk == CXCursor_StructDecl || pk == CXCursor_UnionDecl ||
          pk == CXCursor_ClassTemplate ||
          pk == CXCursor_ClassTemplatePartialSpecialization ||
          pk == CXCursor_EnumDecl) {
        std::string pname = sanitize(toStdString(clang_getCursorSpelling(p)));
        if (!pname.empty()) {
          parts.push_back(pname);
        }
      }
      p = clang_getCursorSemanticParent(p);
    }
    std::reverse(parts.begin(), parts.end());
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) {
        out += "::";
      }
      out += parts[i];
    }
    return out;
  }

  std::string getOwnerUSR(CXCursor c) {
    CXCursor p = clang_getCursorSemanticParent(c);
    if (clang_Cursor_isNull(p)) {
      return "";
    }
    CXCursorKind pk = clang_getCursorKind(p);
    if (pk == CXCursor_ClassDecl || pk == CXCursor_StructDecl ||
        pk == CXCursor_UnionDecl || pk == CXCursor_ClassTemplate ||
        pk == CXCursor_ClassTemplatePartialSpecialization ||
        pk == CXCursor_EnumDecl) {
      CXCursor cp = normalizeTargetCursor(p);
      FileInfo finfo = getCursorFileInfo(cp);
      if (finfo.keep) {
        return getUSR(cp);
      }
    }
    return "";
  }

  void getExtentLines(CXCursor c, unsigned &line_start, unsigned &line_end,
                      unsigned &loc) {
    CXSourceRange range = clang_getCursorExtent(c);
    CXSourceLocation start_loc = clang_getRangeStart(range);
    CXSourceLocation end_loc = clang_getRangeEnd(range);
    CXFile sf = nullptr, ef = nullptr;
    unsigned sc = 0, so = 0, ec = 0, eo = 0;
    line_start = 0;
    line_end = 0;
    clang_getSpellingLocation(start_loc, &sf, &line_start, &sc, &so);
    clang_getSpellingLocation(end_loc, &ef, &line_end, &ec, &eo);
    if (line_end < line_start) {
      line_end = line_start;
    }
    loc = (line_start > 0) ? (line_end - line_start + 1) : 0;
  }

  NodeRecord makeNode(CXCursor c, const std::string &kind,
                      const FileInfo &finfo) {
    NodeRecord n;
    CXCursor canon = normalizeTargetCursor(c);
    n.usr = getUSR(canon);
    n.kind = kind;
    n.repo = finfo.repo;
    n.name = sanitize(toStdString(clang_getCursorSpelling(c)));
    if (n.name.empty()) {
      n.name = "(anon " + kind + ")";
    }
    n.qualname = buildQualName(c);
    n.owner_usr = getOwnerUSR(c);
    n.file = finfo.norm_path;
    n.decl_file = finfo.norm_path;
    getExtentLines(c, n.line_start, n.line_end, n.loc);
    n.is_def = clang_isCursorDefinition(c) ? 1 : 0;
    if (kind == "method" || kind == "dtor") {
      n.is_virtual = clang_CXXMethod_isVirtual(c) ? 1 : 0;
    } else {
      n.is_virtual = 0;
    }
    if (isCallableKindStr(kind)) {
      n.signature = sanitize(toStdString(clang_getTypeSpelling(clang_getCursorType(c))));
    } else if (kind == "typedef") {
      CXType ut = clang_getTypedefDeclUnderlyingType(c);
      n.signature = sanitize(toStdString(clang_getTypeSpelling(ut)));
    }
    return n;
  }

  void upsertNode(const NodeRecord &n) {
    if (n.usr.empty()) {
      return;
    }
    auto it = nodes_.find(n.usr);
    if (it == nodes_.end()) {
      nodes_.emplace(n.usr, n);
      return;
    }
    NodeRecord &cur = it->second;
    if (cur.decl_file.empty() && !n.decl_file.empty()) {
      cur.decl_file = n.decl_file;
    }
    // Prefer header path as decl_file if cur.decl_file was a .cpp
    if (!n.is_def && !n.decl_file.empty()) {
      cur.decl_file = n.decl_file;
    }
    bool replace = false;
    if (n.is_def > cur.is_def) {
      replace = true;
    } else if (n.is_def == cur.is_def && n.loc > cur.loc) {
      replace = true;
    }
    if (replace) {
      std::string saved_decl = cur.decl_file;
      int saved_virt = std::max(cur.is_virtual, n.is_virtual);
      cur = n;
      cur.decl_file = saved_decl.empty() ? n.decl_file : saved_decl;
      cur.is_virtual = saved_virt;
    } else {
      cur.is_virtual = std::max(cur.is_virtual, n.is_virtual);
    }
  }

  void addEdge(const std::string &src_usr, const std::string &dst_usr,
               const std::string &edge_kind, int src_is_def, int delta = 1) {
    if (src_usr.empty() || dst_usr.empty()) {
      return;
    }
    EdgeKey k{src_usr, dst_usr, edge_kind};
    EdgeVal &v = edges_[k];
    if (src_is_def > v.src_is_def) {
      v.src_is_def = src_is_def;
      v.count = delta;
    } else if (src_is_def == v.src_is_def) {
      v.count += delta;
    }
  }

  // Record edges from a type (unwrapping pointers, refs, arrays, typedefs,
  // elaborated types, and template arguments).
  void collectTypeTargets(CXType t, std::vector<std::string> &out_usrs,
                          std::unordered_set<std::string> &seen_local,
                          int depth = 0) {
    if (depth > 8 || t.kind == CXType_Invalid) {
      return;
    }
    switch (t.kind) {
    case CXType_Pointer:
    case CXType_LValueReference:
    case CXType_RValueReference:
    case CXType_BlockPointer:
    case CXType_MemberPointer:
      collectTypeTargets(clang_getPointeeType(t), out_usrs, seen_local,
                         depth + 1);
      return;
    case CXType_ConstantArray:
    case CXType_IncompleteArray:
    case CXType_VariableArray:
    case CXType_DependentSizedArray:
      collectTypeTargets(clang_getArrayElementType(t), out_usrs, seen_local,
                         depth + 1);
      return;
    case CXType_Elaborated:
      collectTypeTargets(clang_Type_getNamedType(t), out_usrs, seen_local,
                         depth + 1);
      return;
    case CXType_FunctionProto: {
      collectTypeTargets(clang_getResultType(t), out_usrs, seen_local,
                         depth + 1);
      int nargs = clang_getNumArgTypes(t);
      for (int i = 0; i < nargs; ++i) {
        collectTypeTargets(clang_getArgType(t, i), out_usrs, seen_local,
                           depth + 1);
      }
      return;
    }
    case CXType_Typedef: {
      CXCursor td = clang_getTypeDeclaration(t);
      CXCursor ctd = normalizeTargetCursor(td);
      FileInfo finfo = getCursorFileInfo(ctd);
      if (finfo.keep) {
        std::string u = getUSR(ctd);
        if (!u.empty() && seen_local.insert(u).second) {
          out_usrs.push_back(u);
        }
      }
      CXType canon_t = clang_getCanonicalType(t);
      if (canon_t.kind != CXType_Typedef && canon_t.kind != CXType_Invalid) {
        collectTypeTargets(canon_t, out_usrs, seen_local, depth + 1);
      }
      return;
    }
    case CXType_Record:
    case CXType_Enum: {
      CXCursor decl = clang_getTypeDeclaration(t);
      CXCursor cdecl = normalizeTargetCursor(decl);
      FileInfo finfo = getCursorFileInfo(cdecl);
      if (finfo.keep) {
        std::string u = getUSR(cdecl);
        if (!u.empty() && seen_local.insert(u).second) {
          out_usrs.push_back(u);
        }
      }
      int ntargs = clang_Type_getNumTemplateArguments(t);
      for (int i = 0; i < ntargs; ++i) {
        CXType targ = clang_Type_getTemplateArgumentAsType(t, i);
        if (targ.kind != CXType_Invalid) {
          collectTypeTargets(targ, out_usrs, seen_local, depth + 1);
        }
      }
      return;
    }
    default: {
      int ntargs = clang_Type_getNumTemplateArguments(t);
      if (ntargs > 0) {
        for (int i = 0; i < ntargs; ++i) {
          CXType targ = clang_Type_getTemplateArgumentAsType(t, i);
          if (targ.kind != CXType_Invalid) {
            collectTypeTargets(targ, out_usrs, seen_local, depth + 1);
          }
        }
      }
      return;
    }
    }
  }

  void extractParamRetEdges(CXCursor fn_cursor, const std::string &fn_usr,
                            int is_def) {
    std::vector<std::string> targets;
    std::unordered_set<std::string> seen;

    CXType ret = clang_getCursorResultType(fn_cursor);
    if (ret.kind != CXType_Invalid) {
      collectTypeTargets(ret, targets, seen);
    }
    int nargs = clang_Cursor_getNumArguments(fn_cursor);
    for (int i = 0; i < nargs; ++i) {
      CXCursor arg = clang_Cursor_getArgument(fn_cursor, i);
      if (!clang_Cursor_isNull(arg)) {
        collectTypeTargets(clang_getCursorType(arg), targets, seen);
      }
    }
    // Note: deduplicate per declaration/definition occurrence
    // Wait: if two parameters have the same type, seen_local collapsed them;
    // actually counting all parameter/ret occurrences is even nicer or 1 per
    // type? Let's count per parameter/return slot:
    targets.clear();
    if (ret.kind != CXType_Invalid) {
      std::unordered_set<std::string> s;
      collectTypeTargets(ret, targets, s);
    }
    for (int i = 0; i < nargs; ++i) {
      CXCursor arg = clang_Cursor_getArgument(fn_cursor, i);
      if (!clang_Cursor_isNull(arg)) {
        std::unordered_set<std::string> s;
        collectTypeTargets(clang_getCursorType(arg), targets, s);
      }
    }
    for (const auto &dst : targets) {
      addEdge(fn_usr, dst, "param_or_ret_type", is_def, 1);
    }
  }

  void extractOverridesEdges(CXCursor fn_cursor, const std::string &fn_usr,
                             int is_def) {
    CXCursor *overridden = nullptr;
    unsigned num_overridden = 0;
    clang_getOverriddenCursors(fn_cursor, &overridden, &num_overridden);
    for (unsigned i = 0; i < num_overridden; ++i) {
      CXCursor base = normalizeTargetCursor(overridden[i]);
      FileInfo bf = getCursorFileInfo(base);
      if (bf.keep) {
        std::string base_usr = getUSR(base);
        addEdge(fn_usr, base_usr, "overrides", is_def, 1);
      }
    }
    if (overridden) {
      clang_disposeOverriddenCursors(overridden);
    }
  }

  struct BodyCtx {
    Extractor *self;
    std::string fn_usr;
    int fn_is_def;
    std::string pending_callee_usr;
  };

  static enum CXChildVisitResult bodyVisitor(CXCursor c, CXCursor /*parent*/,
                                             CXClientData client_data) {
    BodyCtx *ctx = static_cast<BodyCtx *>(client_data);
    Extractor *self = ctx->self;
    CXCursorKind k = clang_getCursorKind(c);

    // Skip parameter declarations (already handled by param_or_ret_type)
    if (k == CXCursor_ParmDecl) {
      return CXChildVisit_Continue;
    }

    if (k == CXCursor_CallExpr) {
      CXCursor ref = clang_getCursorReferenced(c);
      std::string saved_pending = ctx->pending_callee_usr;
      ctx->pending_callee_usr.clear();

      if (!clang_Cursor_isNull(ref)) {
        CXCursor target = self->normalizeTargetCursor(ref);
        std::string tkind = self->mapCursorKind(target);
        if (isCallableKindStr(tkind)) {
          FileInfo tf = self->getCursorFileInfo(target);
          if (tf.keep) {
            std::string tusr = self->getUSR(target);
            self->addEdge(ctx->fn_usr, tusr, "calls", ctx->fn_is_def, 1);
            ctx->pending_callee_usr = tusr;
          }
        }
      }
      clang_visitChildren(c, bodyVisitor, ctx);
      ctx->pending_callee_usr = saved_pending;
      return CXChildVisit_Continue;
    }

    if (k == CXCursor_MemberRefExpr || k == CXCursor_MemberRef ||
        k == CXCursor_DeclRefExpr) {
      CXCursor ref = clang_getCursorReferenced(c);
      if (!clang_Cursor_isNull(ref)) {
        CXCursorKind rk = clang_getCursorKind(ref);
        if (rk == CXCursor_FieldDecl &&
            (k == CXCursor_MemberRefExpr || k == CXCursor_MemberRef)) {
          CXCursor owner =
              self->normalizeTargetCursor(clang_getCursorSemanticParent(ref));
          std::string okind = self->mapCursorKind(owner);
          if (okind == "class" || okind == "struct") {
            FileInfo of = self->getCursorFileInfo(owner);
            if (of.keep) {
              std::string ousr = self->getUSR(owner);
              self->addEdge(ctx->fn_usr, ousr, "member_field_access",
                            ctx->fn_is_def, 1);
            }
          }
        } else {
          CXCursor target = self->normalizeTargetCursor(ref);
          std::string tkind = self->mapCursorKind(target);
          if (isCallableKindStr(tkind)) {
            FileInfo tf = self->getCursorFileInfo(target);
            if (tf.keep) {
              std::string tusr = self->getUSR(target);
              if (!ctx->pending_callee_usr.empty() &&
                  ctx->pending_callee_usr == tusr) {
                ctx->pending_callee_usr.clear();
              } else {
                self->addEdge(ctx->fn_usr, tusr, "calls", ctx->fn_is_def, 1);
              }
            }
          }
        }
      }
      clang_visitChildren(c, bodyVisitor, ctx);
      return CXChildVisit_Continue;
    }

    if (k == CXCursor_TypeRef || k == CXCursor_TemplateRef) {
      CXCursor ref = clang_getCursorReferenced(c);
      if (!clang_Cursor_isNull(ref)) {
        CXCursor target = self->normalizeTargetCursor(ref);
        std::string tkind = self->mapCursorKind(target);
        if (isTypeKindStr(tkind)) {
          FileInfo tf = self->getCursorFileInfo(target);
          if (tf.keep) {
            std::string tusr = self->getUSR(target);
            self->addEdge(ctx->fn_usr, tusr, "body_type_ref", ctx->fn_is_def,
                          1);
          }
        }
      }
      return CXChildVisit_Continue;
    }

    return CXChildVisit_Recurse;
  }

  struct DeclCtx {
    Extractor *self;
    std::string current_record_usr;
    int current_record_is_def = 0;
  };

  static enum CXChildVisitResult declVisitor(CXCursor c, CXCursor /*parent*/,
                                             CXClientData client_data) {
    DeclCtx *ctx = static_cast<DeclCtx *>(client_data);
    Extractor *self = ctx->self;
    CXCursorKind k = clang_getCursorKind(c);

    // Fast skip for system headers unless it's a namespace/linkage spec
    if (k != CXCursor_Namespace && k != CXCursor_LinkageSpec &&
        k != CXCursor_UnexposedDecl) {
      CXSourceLocation loc = clang_getCursorLocation(c);
      if (clang_Location_isInSystemHeader(loc)) {
        return CXChildVisit_Continue;
      }
    }

    if (k == CXCursor_Namespace || k == CXCursor_LinkageSpec ||
        k == CXCursor_UnexposedDecl) {
      return CXChildVisit_Recurse;
    }

    if (k == CXCursor_CXXBaseSpecifier && !ctx->current_record_usr.empty()) {
      CXCursor ref = clang_getTypeDeclaration(clang_getCursorType(c));
      if (clang_Cursor_isNull(ref)) {
        ref = clang_getCursorReferenced(c);
      }
      CXCursor base = self->normalizeTargetCursor(ref);
      FileInfo bf = self->getCursorFileInfo(base);
      if (bf.keep) {
        std::string busr = self->getUSR(base);
        self->addEdge(ctx->current_record_usr, busr, "inherits",
                      ctx->current_record_is_def, 1);
      }
      return CXChildVisit_Continue;
    }

    if (k == CXCursor_FieldDecl && !ctx->current_record_usr.empty()) {
      std::vector<std::string> targets;
      std::unordered_set<std::string> seen;
      self->collectTypeTargets(clang_getCursorType(c), targets, seen);
      for (const auto &dst : targets) {
        self->addEdge(ctx->current_record_usr, dst, "field_type",
                      ctx->current_record_is_def, 1);
      }
      return CXChildVisit_Continue;
    }

    std::string nkind = self->mapCursorKind(c);
    if (nkind.empty()) {
      return CXChildVisit_Continue;
    }

    FileInfo finfo = self->getCursorFileInfo(c);
    if (!finfo.keep) {
      return CXChildVisit_Continue;
    }

    // Skip implicit template specializations if they somehow appear
    CXCursor spec_tmpl = clang_getSpecializedCursorTemplate(c);
    if (!clang_Cursor_isNull(spec_tmpl) &&
        k != CXCursor_ClassTemplatePartialSpecialization) {
      // Only skip if it's an implicit instantiation (not an explicit specialization)
      // Actually in clang_visitChildren at TU/Record scope, explicit specializations
      // appear as ClassDecl/FunctionDecl; checking if it's a record/function whose
      // specialization kind is implicit vs explicit isn't exposed directly in C API,
      // except implicit instantiations aren't visited at TU scope unless requested.
      // Let's check: if clang_equalCursors(c, spec_tmpl) is false and it's not
      // inside source, skip? No, explicit specializations ARE in source and useful,
      // so let's not blindly drop them unless they duplicate.
    }

    NodeRecord node = self->makeNode(c, nkind, finfo);
    self->upsertNode(node);

    if (isTypeKindStr(nkind)) {
      if (nkind == "class" || nkind == "struct") {
        DeclCtx child_ctx{self, node.usr, node.is_def};
        clang_visitChildren(c, declVisitor, &child_ctx);
      }
      return CXChildVisit_Continue;
    }

    if (isCallableKindStr(nkind)) {
      self->extractOverridesEdges(c, node.usr, node.is_def);
      self->extractParamRetEdges(c, node.usr, node.is_def);
      if (node.is_def) {
        BodyCtx bctx{self, node.usr, node.is_def, ""};
        clang_visitChildren(c, bodyVisitor, &bctx);
      }
      return CXChildVisit_Continue;
    }

    return CXChildVisit_Continue;
  }

  void visitDeclScope(CXCursor root) {
    DeclCtx ctx{this, "", 0};
    clang_visitChildren(root, declVisitor, &ctx);
  }
};

void writeNodes(const std::string &path,
                const std::unordered_map<std::string, NodeRecord> &nodes) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Failed to open nodes output: " << path << "\n";
    std::exit(2);
  }
  for (const auto &kv : nodes) {
    const NodeRecord &n = kv.second;
    out << n.usr << '\t' << n.kind << '\t' << n.repo << '\t' << n.name << '\t'
        << n.qualname << '\t' << n.owner_usr << '\t' << n.file << '\t'
        << n.line_start << '\t' << n.line_end << '\t' << n.loc << '\t'
        << n.is_def << '\t' << n.is_virtual << '\t' << n.signature << '\t'
        << n.decl_file << '\n';
  }
}

void writeEdges(const std::string &path,
                const std::map<EdgeKey, EdgeVal> &edges) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Failed to open edges output: " << path << "\n";
    std::exit(2);
  }
  for (const auto &kv : edges) {
    out << kv.first.src_usr << '\t' << kv.first.dst_usr << '\t'
        << kv.first.edge_kind << '\t' << kv.second.count << '\t'
        << kv.second.src_is_def << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string tu_path;
  std::string repo;
  std::string out_nodes;
  std::string out_edges;
  std::vector<const char *> clang_args;

  bool after_dashdash = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (!after_dashdash) {
      if (arg == "--") {
        after_dashdash = true;
        continue;
      } else if (arg == "--tu" && i + 1 < argc) {
        tu_path = argv[++i];
        continue;
      } else if (arg == "--repo" && i + 1 < argc) {
        repo = argv[++i];
        continue;
      } else if (arg == "--out-nodes" && i + 1 < argc) {
        out_nodes = argv[++i];
        continue;
      } else if (arg == "--out-edges" && i + 1 < argc) {
        out_edges = argv[++i];
        continue;
      }
    }
    clang_args.push_back(argv[i]);
  }

  if (tu_path.empty() || out_nodes.empty() || out_edges.empty()) {
    std::cerr << "Usage: extract_ast --tu <file.cpp> --repo <clp|coinutils> "
                 "--out-nodes <nodes.tsv> --out-edges <edges.tsv> -- <clang args...>\n";
    return 1;
  }

  CXIndex index = clang_createIndex(0, 0);
  CXTranslationUnit tu = nullptr;
  CXErrorCode err = clang_parseTranslationUnit2(
      index, tu_path.c_str(), clang_args.data(),
      static_cast<int>(clang_args.size()), nullptr, 0,
      CXTranslationUnit_None, &tu);

  if (err != CXError_Success || !tu) {
    std::cerr << "ERROR parsing TU " << tu_path << " (CXErrorCode=" << err
              << ")\n";
    if (tu)
      clang_disposeTranslationUnit(tu);
    clang_disposeIndex(index);
    return 2;
  }

  unsigned num_diag = clang_getNumDiagnostics(tu);
  unsigned fatal_cnt = 0, err_cnt = 0, warn_cnt = 0;
  for (unsigned i = 0; i < num_diag; ++i) {
    CXDiagnostic d = clang_getDiagnostic(tu, i);
    CXDiagnosticSeverity sev = clang_getDiagnosticSeverity(d);
    if (sev == CXDiagnostic_Fatal) {
      ++fatal_cnt;
      std::cerr << "[FATAL] " << tu_path << ": "
                << toStdString(clang_formatDiagnostic(
                       d, clang_defaultDiagnosticDisplayOptions()))
                << "\n";
    } else if (sev == CXDiagnostic_Error) {
      ++err_cnt;
      std::cerr << "[ERROR] " << tu_path << ": "
                << toStdString(clang_formatDiagnostic(
                       d, clang_defaultDiagnosticDisplayOptions()))
                << "\n";
    } else if (sev == CXDiagnostic_Warning) {
      ++warn_cnt;
    }
    clang_disposeDiagnostic(d);
  }

  Extractor ext(tu);
  ext.run();

  writeNodes(out_nodes, ext.nodes());
  writeEdges(out_edges, ext.edges());

  std::cout << tu_path << '\t' << repo << '\t' << fatal_cnt << '\t' << err_cnt
            << '\t' << warn_cnt << '\t' << ext.nodes().size() << '\t'
            << ext.edges().size() << '\n';

  clang_disposeTranslationUnit(tu);
  clang_disposeIndex(index);
  return (fatal_cnt > 0 || err_cnt > 0) ? 3 : 0;
}
