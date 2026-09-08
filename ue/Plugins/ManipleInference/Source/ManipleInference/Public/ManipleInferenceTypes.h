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
	static FManipleTensor MakeInt64(const FString& InName, TConstArrayView<int64> InShape, TConstArrayView<int64> Values);
	static FManipleTensor MakeBool(const FString& InName, TConstArrayView<int64> InShape, TConstArrayView<bool> Values);
	/** BYTES tensor of shape [Strings.Num()] (Triton STRING: 4-byte little-endian length + utf-8 per element). */
	static FManipleTensor MakeStrings(const FString& InName, TConstArrayView<FString> Strings);
	static FManipleTensor MakeString(const FString& InName, const FString& Value) { return MakeStrings(InName, {Value}); }

	int64 NumElements() const;
	/** Bytes per element for the datatype, 0 if unknown/variable (BYTES). */
	static int32 ElementSize(const FString& Datatype);
	/** Typed views (no copy). Caller must check the datatype. */
	TConstArrayView<float> AsFloats() const
	{
		return TConstArrayView<float>(reinterpret_cast<const float*>(Data.GetData()), Data.Num() / sizeof(float));
	}
	TConstArrayView<int64> AsInt64s() const
	{
		return TConstArrayView<int64>(reinterpret_cast<const int64*>(Data.GetData()), Data.Num() / sizeof(int64));
	}
	/** Decodes a BYTES tensor into strings (empty if the datatype is not BYTES). */
	TArray<FString> AsStrings() const;
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
