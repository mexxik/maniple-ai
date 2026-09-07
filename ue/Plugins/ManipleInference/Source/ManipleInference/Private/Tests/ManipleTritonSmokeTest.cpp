// Smoke test against a running Triton: readiness, correctness vs the exported reference, round-trip latency.
// Server URL: -ManipleTritonUrl=host:8001 (default localhost:8001).
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Interfaces/IPluginManager.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "ManipleTritonClient.h"
#include "ManipleInference.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FManipleTritonSmokeTest, "Maniple.Triton.Smoke",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FManipleTritonSmokeTest::RunTest(const FString& Parameters)
{
	FString Url = TEXT("localhost:8001");
	FParse::Value(FCommandLine::Get(), TEXT("ManipleTritonUrl="), Url);

	// reference produced by python/export_policy.py
	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("ManipleInference"));
	if (!TestTrue(TEXT("plugin found"), Plugin.IsValid())) return false;
	FString Json;
	if (!TestTrue(TEXT("reference json loads"), FFileHelper::LoadFileToString(Json, *FPaths::Combine(Plugin->GetBaseDir(), TEXT("Resources/policy_ref.json"))))) return false;
	TSharedPtr<FJsonObject> Ref;
	if (!TestTrue(TEXT("reference json parses"), FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Ref) && Ref.IsValid())) return false;
	const FString Model = Ref->GetStringField(TEXT("model"));
	const FString InName = Ref->GetStringField(TEXT("input"));
	const FString OutName = Ref->GetStringField(TEXT("output"));
	const int32 ObsDim = Ref->GetIntegerField(TEXT("obs_dim"));
	const int32 ActDim = Ref->GetIntegerField(TEXT("act_dim"));
	const int32 RefBatch = Ref->GetIntegerField(TEXT("batch"));
	TArray<float> RefObs, RefAct;
	for (const auto& V : Ref->GetArrayField(TEXT("obs"))) RefObs.Add((float)V->AsNumber());
	for (const auto& V : Ref->GetArrayField(TEXT("action"))) RefAct.Add((float)V->AsNumber());

	FManipleTritonClient Client(Url, 5.f);
	if (!TestTrue(FString::Printf(TEXT("triton ready at %s"), *Url), Client.IsServerReadySync(5.f))) return false;

	// 1) correctness
	{
		const int64 Shape[2] = { RefBatch, ObsDim };
		FManipleInferResult R = Client.InferSync(Model, { FManipleTensor::MakeFloat(InName, Shape, RefObs) });
		if (!TestTrue(FString::Printf(TEXT("infer ok: %s"), *R.Error), R.bSuccess)) return false;
		const FManipleTensor* Out = R.FindOutput(OutName);
		if (!TestNotNull(TEXT("output present"), Out)) return false;
		TestEqual(TEXT("output datatype"), Out->Datatype, FString(TEXT("FP32")));
		TestEqual(TEXT("output rank"), Out->Shape.Num(), 2);
		TestEqual(TEXT("output batch"), (int32)Out->Shape[0], RefBatch);
		TestEqual(TEXT("output dim"), (int32)Out->Shape[1], ActDim);
		TConstArrayView<float> Act = Out->AsFloats();
		if (!TestEqual(TEXT("output element count"), Act.Num(), RefAct.Num())) return false;
		float MaxErr = 0.f;
		for (int32 i = 0; i < Act.Num(); ++i) MaxErr = FMath::Max(MaxErr, FMath::Abs(Act[i] - RefAct[i]));
		UE_LOG(LogManipleInference, Display, TEXT("[smoke] %s v%s batch %d: max abs err %.3g, round-trip %.2f ms"), *R.ModelName, *R.ModelVersion, RefBatch, MaxErr, R.LatencyMs);
		TestTrue(TEXT("matches reference (1e-4)"), MaxErr < 1e-4f);
	}

	// 2) latency by batch size (sequential, warm)
	for (int32 B : { 1, 64, 512 })
	{
		TArray<float> Obs; Obs.SetNum(B * ObsDim);
		for (int32 i = 0; i < Obs.Num(); ++i) Obs[i] = FMath::Sin(0.01f * i);
		const int64 Shape[2] = { B, ObsDim };
		const int32 Iters = 10;
		double Sum = 0, Best = 1e9;
		for (int32 k = 0; k < Iters; ++k)
		{
			FManipleInferResult R = Client.InferSync(Model, { FManipleTensor::MakeFloat(InName, Shape, Obs) });
			if (!TestTrue(FString::Printf(TEXT("batch %d iter %d ok: %s"), B, k, *R.Error), R.bSuccess)) return false;
			Sum += R.LatencyMs; Best = FMath::Min(Best, R.LatencyMs);
		}
		UE_LOG(LogManipleInference, Display, TEXT("[smoke] batch %4d: round-trip best %.2f ms avg %.2f ms (%.1f us/agent)"), B, Best, Sum / Iters, Sum / Iters * 1000.0 / B);
	}
	return true;
}

#endif // WITH_AUTOMATION_TESTS
