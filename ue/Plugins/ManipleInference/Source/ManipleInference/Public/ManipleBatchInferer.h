#pragma once

#include "CoreMinimal.h"
#include "ManipleInferenceTypes.h"

class FManipleTritonClient;

/**
 * Client-side batching: many agents submit one observation row each during a tick,
 * Flush() sends a single [N, InDim] request and scatters the [N, OutDim] result back
 * to the per-row callbacks. Game thread only.
 */
class MANIPLEINFERENCE_API FManipleBatchInferer
{
public:
	/** Called per row: bOk=false with an empty view if the request failed. */
	using FRowCallback = TFunction<void(bool bOk, TConstArrayView<float> Output)>;
	using FBatchCallback = TFunction<void(const FManipleInferResult&, int32 Rows)>;

	FManipleBatchInferer(TSharedPtr<FManipleTritonClient> InClient, const FString& InModel,
		const FString& InInputName, int32 InInDim, const FString& InOutputName, int32 InOutDim);

	void Submit(TConstArrayView<float> Row, FRowCallback Callback);
	int32 NumPending() const { return Callbacks.Num(); }

	/** Sends the pending rows as one request. Returns false if nothing was pending. */
	bool Flush(FBatchCallback OnBatchDone = nullptr);

private:
	TSharedPtr<FManipleTritonClient> Client;
	FString Model, InputName, OutputName;
	int32 InDim, OutDim;
	TArray<float> Rows;
	TArray<FRowCallback> Callbacks;
};
