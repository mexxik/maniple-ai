// Protocol round trip against a running Triton: register a throwaway policy on ppo_train, act on the latest
// weights, send one batch of transitions back, read status. Server URL: -ManipleTritonUrl=host:8001.
#include "Misc/AutomationTest.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "ManipleTritonClient.h"
#include "ManipleAgentClient.h"
#include "ManipleInference.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FManipleTritonSmokeTest, "Maniple.Triton.Smoke", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FManipleTritonSmokeTest::RunTest(const FString& Parameters)
{
	FString Url = TEXT("localhost:8001");
	FParse::Value(FCommandLine::Get(), TEXT("ManipleTritonUrl="), Url);

	constexpr int32 ObsDim = 4, ActDim = 2, Rows = 3;

	TSharedPtr<FManipleTritonClient> Client = MakeShared<FManipleTritonClient>(Url, 5.f);
	if (!TestTrue(FString::Printf(TEXT("triton ready at %s"), *Url), Client->IsServerReadySync(5.f)))
		return false;

	FManipleAgentClient Agent(Client, TEXT("uesmoke"));

	// 1) register (idempotent)
	FManipleAgentSpec Spec;
	Spec.ObsDim = ObsDim;
	Spec.ActDim = ActDim;
	Spec.Preset = TEXT("small");
	FManipleAgentStatus Reg = Agent.RegisterSync(Spec);
	if (!TestTrue(FString::Printf(TEXT("register ok: %s"), *Reg.Error), Reg.bOk))
		return false;
	UE_LOG(LogManipleInference, Display, TEXT("[smoke] registered v%lld in %.2f ms"), Reg.GetInt(TEXT("version")), Reg.LatencyMs);

	// 2) act on latest, greedy: deterministic
	TArray<float> Obs;
	for (int32 i = 0; i < Rows * ObsDim; ++i)
		Obs.Add(0.1f * i);
	FManipleActResult A = Agent.ActSync(Obs, Rows, ObsDim, false, TEXT("latest"));
	if (!TestTrue(FString::Printf(TEXT("act ok: %s"), *A.Error), A.bOk))
		return false;
	TestEqual(TEXT("action count"), A.Action.Num(), Rows * ActDim);
	TestEqual(TEXT("logp count"), A.LogP.Num(), Rows);
	FManipleActResult B = Agent.ActSync(Obs, Rows, ObsDim, false, TEXT("latest"));
	float MaxDiff = 0.f;
	for (int32 i = 0; i < A.Action.Num() && i < B.Action.Num(); ++i)
		MaxDiff = FMath::Max(MaxDiff, FMath::Abs(A.Action[i] - B.Action[i]));
	TestTrue(TEXT("greedy act is deterministic"), MaxDiff < 1e-6f);
	UE_LOG(LogManipleInference, Display, TEXT("[smoke] act v%lld rows %d: %.2f ms, first action (%.3f, %.3f)"), A.PolicyVersion, Rows,
		A.LatencyMs, A.Action[0], A.Action[1]);

	// 3) act with exploration, then observe the transitions
	FManipleActResult E = Agent.ActSync(Obs, Rows, ObsDim, true, TEXT("latest"));
	if (!TestTrue(FString::Printf(TEXT("explore act ok: %s"), *E.Error), E.bOk))
		return false;
	FManipleTransitionBatch Batch(ObsDim, ActDim);
	for (int32 i = 0; i < Rows; ++i)
		Batch.Add(
			TConstArrayView<float>(Obs).Slice(i * ObsDim, ObsDim), E.Row(i, ActDim), 0.5f, i == Rows - 1, i, 0, E.PolicyVersion, E.LogP[i]);
	bool bObserved = false;
	FManipleAgentStatus Obs1;
	Agent.Observe(Batch,
		[&](const FManipleAgentStatus& S)
		{
			bObserved = true;
			Obs1 = S;
		});
	const double Deadline = FPlatformTime::Seconds() + 5.0;
	while (!bObserved && FPlatformTime::Seconds() < Deadline)
	{
		Client->PumpCompletions();
		FPlatformProcess::Sleep(0.0005f);
	}
	if (!TestTrue(FString::Printf(TEXT("observe ok: %s"), *Obs1.Error), bObserved && Obs1.bOk))
		return false;
	TestEqual(TEXT("observe accepted rows"), (int32)Obs1.GetInt(TEXT("accepted")), Rows);

	// 4) status
	FManipleAgentStatus St = Agent.StatusSync();
	TestTrue(TEXT("status ok"), St.bOk);
	UE_LOG(LogManipleInference, Display, TEXT("[smoke] status: %s"), *St.ToString());
	return true;
}

#endif // WITH_AUTOMATION_TESTS
