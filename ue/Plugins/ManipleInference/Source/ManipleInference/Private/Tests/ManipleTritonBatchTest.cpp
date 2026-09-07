// Client-side batching: N rows in one request must equal N individual requests.
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Interfaces/IPluginManager.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "HttpModule.h"
#include "HttpManager.h"
#include "HAL/PlatformTime.h"
#include "HAL/PlatformProcess.h"
#include "ManipleTritonClient.h"
#include "ManipleBatchInferer.h"
#include "ManipleInference.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FManipleTritonBatchTest, "Maniple.Triton.Batch",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FManipleTritonBatchTest::RunTest(const FString& Parameters)
{
	FString Url = TEXT("http://localhost:8000");
	FParse::Value(FCommandLine::Get(), TEXT("ManipleTritonUrl="), Url);

	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("ManipleInference"));
	FString Json;
	if (!TestTrue(TEXT("reference json loads"), Plugin.IsValid() && FFileHelper::LoadFileToString(Json, *FPaths::Combine(Plugin->GetBaseDir(), TEXT("Resources/policy_ref.json"))))) return false;
	TSharedPtr<FJsonObject> Ref;
	if (!TestTrue(TEXT("reference json parses"), FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Ref) && Ref.IsValid())) return false;
	const FString Model = Ref->GetStringField(TEXT("model")), InName = Ref->GetStringField(TEXT("input")), OutName = Ref->GetStringField(TEXT("output"));
	const int32 ObsDim = Ref->GetIntegerField(TEXT("obs_dim")), ActDim = Ref->GetIntegerField(TEXT("act_dim")), RefBatch = Ref->GetIntegerField(TEXT("batch"));
	TArray<float> RefObs, RefAct;
	for (const auto& V : Ref->GetArrayField(TEXT("obs"))) RefObs.Add((float)V->AsNumber());
	for (const auto& V : Ref->GetArrayField(TEXT("action"))) RefAct.Add((float)V->AsNumber());

	TSharedPtr<FManipleTritonClient> Client = MakeShared<FManipleTritonClient>(Url, 5.f);
	if (!TestTrue(TEXT("triton ready"), Client->IsServerReadySync(5.f))) return false;

	FManipleBatchInferer Batch(Client, Model, InName, ObsDim, OutName, ActDim);
	TArray<TArray<float>> RowOut; RowOut.SetNum(RefBatch);
	int32 Done = 0, Failed = 0;
	for (int32 i = 0; i < RefBatch; ++i)
	{
		Batch.Submit(TConstArrayView<float>(RefObs).Slice(i * ObsDim, ObsDim), [&, i](bool bOk, TConstArrayView<float> Out)
		{
			++Done; if (!bOk) ++Failed; else RowOut[i] = TArray<float>(Out.GetData(), Out.Num());
		});
	}
	TestEqual(TEXT("pending rows"), Batch.NumPending(), RefBatch);
	bool bBatchDone = false; double BatchMs = 0;
	if (!TestTrue(TEXT("flush sends"), Batch.Flush([&](const FManipleInferResult& R, int32 Rows) { bBatchDone = true; BatchMs = R.LatencyMs; }))) return false;
	TestEqual(TEXT("pending cleared"), Batch.NumPending(), 0);
	const double Deadline = FPlatformTime::Seconds() + 6.0;
	while (!bBatchDone && FPlatformTime::Seconds() < Deadline) { FHttpModule::Get().GetHttpManager().Tick(0.01f); FPlatformProcess::Sleep(0.001f); }
	if (!TestTrue(TEXT("batch completed"), bBatchDone)) return false;
	TestEqual(TEXT("all rows called back"), Done, RefBatch);
	TestEqual(TEXT("no row failed"), Failed, 0);

	float MaxErrRef = 0.f, MaxErrIndividual = 0.f;
	for (int32 i = 0; i < RefBatch; ++i)
	{
		if (!TestEqual(FString::Printf(TEXT("row %d size"), i), RowOut[i].Num(), ActDim)) return false;
		const int64 Shape[2] = { 1, ObsDim };
		FManipleInferResult Single = Client->InferSync(Model, { FManipleTensor::MakeFloat(InName, Shape, TConstArrayView<float>(RefObs).Slice(i * ObsDim, ObsDim)) });
		if (!TestTrue(FString::Printf(TEXT("single row %d ok"), i), Single.bSuccess)) return false;
		TConstArrayView<float> S = Single.FindOutput(OutName)->AsFloats();
		for (int32 k = 0; k < ActDim; ++k)
		{
			MaxErrRef = FMath::Max(MaxErrRef, FMath::Abs(RowOut[i][k] - RefAct[i * ActDim + k]));
			MaxErrIndividual = FMath::Max(MaxErrIndividual, FMath::Abs(RowOut[i][k] - S[k]));
		}
	}
	UE_LOG(LogManipleInference, Display, TEXT("[batch] %d rows in one request: %.2f ms, max err vs reference %.3g, vs individual requests %.3g"), RefBatch, BatchMs, MaxErrRef, MaxErrIndividual);
	TestTrue(TEXT("batch matches reference"), MaxErrRef < 1e-4f);
	TestTrue(TEXT("batch matches individual"), MaxErrIndividual < 1e-5f);
	return true;
}

#endif
