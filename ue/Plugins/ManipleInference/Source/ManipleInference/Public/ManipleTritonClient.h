#pragma once

#include "CoreMinimal.h"
#include "ManipleInferenceTypes.h"

/**
 * Minimal Triton client speaking the KServe v2 inference protocol over HTTP,
 * with the binary tensor extension (raw bytes, no JSON number arrays).
 *
 * Async methods must be called from the game thread; callbacks arrive on the game thread.
 * The *Sync variants pump the HTTP manager until completion and exist for tests and tools only.
 */
class MANIPLEINFERENCE_API FManipleTritonClient
{
public:
	explicit FManipleTritonClient(const FString& InBaseUrl = TEXT("http://localhost:8000"), float InTimeoutSec = 5.f);

	const FString& GetBaseUrl() const { return BaseUrl; }

	/** GET /v2/health/ready */
	void IsServerReady(FManipleReadyComplete OnComplete) const;
	bool IsServerReadySync(float WaitSec = 5.f) const;

	/**
	 * POST /v2/models/{Model}[/versions/{Version}]/infer with binary inputs; all outputs requested as binary.
	 * @param OutputNames  optional subset of outputs; empty = all outputs of the model.
	 */
	void Infer(const FString& Model, TArray<FManipleTensor> Inputs, FManipleInferComplete OnComplete,
		const TArray<FString>& OutputNames = {}, const FString& Version = FString()) const;
	FManipleInferResult InferSync(const FString& Model, TArray<FManipleTensor> Inputs,
		const TArray<FString>& OutputNames = {}, const FString& Version = FString()) const;

	/** Build the KServe v2 request body (JSON header + raw tensor bytes). Exposed for tests. */
	static TArray<uint8> BuildInferBody(const TArray<FManipleTensor>& Inputs, const TArray<FString>& OutputNames, int32& OutJsonLength);
	/** Parse a KServe v2 response body. JsonLength < 0 means no binary header: the body is plain JSON. */
	static bool ParseInferResponse(const TArray<uint8>& Body, int32 JsonLength, FManipleInferResult& Out);

private:
	FString BaseUrl;
	float TimeoutSec;
};
