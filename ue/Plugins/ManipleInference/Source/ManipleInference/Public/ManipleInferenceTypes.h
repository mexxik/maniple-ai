#pragma once

#include "CoreMinimal.h"

/** A dense tensor in Triton terms: name, shape, datatype string ("FP32", "INT64", ...), raw little-endian bytes. */
struct MANIPLEINFERENCE_API FManipleTensor
{
	FString Name;
	TArray<int64> Shape;
	FString Datatype = TEXT("FP32");
	TArray<uint8> Data;

	static FManipleTensor MakeFloat(const FString& InName, TConstArrayView<int64> InShape, TConstArrayView<float> Values);

	int64 NumElements() const;
	/** Bytes per element for the datatype, 0 if unknown/variable (BYTES). */
	static int32 ElementSize(const FString& Datatype);
	/** Typed views (no copy). Caller must check the datatype. */
	TConstArrayView<float> AsFloats() const
	{
		return TConstArrayView<float>(reinterpret_cast<const float*>(Data.GetData()), Data.Num() / sizeof(float));
	}
};

struct MANIPLEINFERENCE_API FManipleInferResult
{
	bool bSuccess = false;
	int32 StatusCode = 0; // grpc::StatusCode (0 = OK)
	FString Error;
	FString ModelName;
	FString ModelVersion;
	TArray<FManipleTensor> Outputs;
	/** Wall-clock round trip measured by the client, in milliseconds. */
	double LatencyMs = 0.0;

	const FManipleTensor* FindOutput(const FString& Name) const;
};

DECLARE_DELEGATE_OneParam(FManipleInferComplete, const FManipleInferResult&);
DECLARE_DELEGATE_OneParam(FManipleReadyComplete, bool /*bReady*/);
