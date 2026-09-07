#pragma once
// gRPC / protobuf headers under UE compile flags. Include only from .cpp files.
#include "HAL/Platform.h"
THIRD_PARTY_INCLUDES_START
#pragma push_macro("check")
#pragma push_macro("verify")
#pragma push_macro("TEXT")
#undef check
#undef verify
#undef TEXT
#include <grpcpp/grpcpp.h>
#include "grpc_service.grpc.pb.h"
#pragma pop_macro("TEXT")
#pragma pop_macro("verify")
#pragma pop_macro("check")
THIRD_PARTY_INCLUDES_END
