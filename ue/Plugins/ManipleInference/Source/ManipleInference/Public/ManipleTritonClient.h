#pragma once

#include "CoreMinimal.h"
#include "ManipleInferenceTypes.h"

/**
 * Triton client over gRPC (GRPCInferenceService). One channel per client; requests use
 * raw_input_contents / raw_output_contents (no per-element encoding).
 *
 * Async methods must be called from the game thread; completions are delivered on the game thread
 * during the engine tick (or by the *Sync variants, which pump until done). Thread-safe internally.
 */
class MANIPLEINFERENCE_API FManipleTritonClient
{
public:
	explicit FManipleTritonClient(const FString& InTarget = TEXT("localhost:8001"), float InTimeoutSec = 5.f);
	~FManipleTritonClient();

	const FString& GetTarget() const { return Target; }

	/** ServerReady RPC. */
	void IsServerReady(FManipleReadyComplete OnComplete);
	bool IsServerReadySync(float WaitSec = 5.f);

	/**
	 * ModelInfer RPC. @param OutputNames optional subset; empty = all outputs of the model.
	 */
	void Infer(const FString& Model, TArray<FManipleTensor> Inputs, FManipleInferComplete OnComplete,
		const TArray<FString>& OutputNames = {}, const FString& Version = FString());
	FManipleInferResult InferSync(const FString& Model, TArray<FManipleTensor> Inputs,
		const TArray<FString>& OutputNames = {}, const FString& Version = FString());

	/** Delivers finished completions to their callbacks. Called automatically each frame; also usable manually. */
	void PumpCompletions();

	/** Number of requests in flight. */
	int32 NumPending() const;

private:
	struct FImpl;
	TUniquePtr<FImpl> Impl;
	FString Target;
	float TimeoutSec;
};
