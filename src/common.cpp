#include "common.h"

namespace pps {

std::vector<std::string> splitString(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(sep, start);
    if (end == std::string::npos) {
      if (start < s.size()) out.push_back(s.substr(start));
      break;
    }
    if (end > start) out.push_back(s.substr(start, end - start));
    start = end + 1;
  }
  return out;
}

}  // namespace pps
