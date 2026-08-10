// Swift sees the engine through the C ABI alone. The header ships inside bmoe.xcframework
// (scripts/build-ios.sh); Xcode's xcframework handling puts it on the search path.
#include "bmoe_c.h"
