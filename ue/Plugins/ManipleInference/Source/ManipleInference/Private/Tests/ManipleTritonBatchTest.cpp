// Latency by batch size through the agent protocol: act on the trainer's latest weights (ppo_train) and,
// when an exported version exists, on the best channel (ppo_infer). Uses the policy registered by the smoke test.
#include "Misc/AutomationTest.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "ManipleTritonClient.h"
#include "ManipleAgentClient.h"
#include "ManipleInference.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FManipleTritonBatchTest, "Maniple.Triton.Batch", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FManipleTritonBatchTest::RunTest(const FString& Parameters)
{
	FString Url = TEXT("localhost:8001");
	FParse::Value(FCommandLine::Get(), TEXT("ManipleTritonUrl="), Url);

	constexpr int32 ObsDim = 4, ActDim = 2;

	TSharedPtr<FManipleTritonClient> Client = MakeShared<FManipleTritonClient>(Url, 5.f);
	if (!TestTrue(TEXT("triton ready"), Client->IsServerReadySync(5.f)))
		return false;

	FManipleAgentClient Agent(Client, TEXT("uesmoke"));
	FManipleAgentSpec Spec;
	Spec.ObsDim = ObsDim;
	Spec.ActDim = ActDim;
	Spec.Preset = TEXT("small");
	if (!TestTrue(TEXT("register ok"), Agent.RegisterSync(Spec).bOk))
		return false;

	for (const TCHAR* Channel : {TEXT("latest"), TEXT("best")})
	{
		for (int32 Rows : {1, 64, 512})
		{
			TArray<float> Obs;
			Obs.SetNum(Rows * ObsDim);
			for (int32 i = 0; i < Obs.Num(); ++i)
				Obs[i] = FMath::Sin(0.01f * i);

			const int32 Iters = 10;
			double Sum = 0, Best = 1e9;
			bool bSkip = false;
			for (int32 k = 0; k < Iters; ++k)
			{
				FManipleActResult R = Agent.ActSync(Obs, Rows, ObsDim, true, Channel);
				if (!R.bOk)
				{
					// best needs an exported version; a fresh policy has none yet
					UE_LOG(LogManipleInference, Display, TEXT("[batch] channel %s unavailable: %s"), Channel, *R.Error);
					bSkip = true;
					break;
				}
				TestEqual(TEXT("action count"), R.Action.Num(), Rows * ActDim);
				Sum += R.LatencyMs;
				Best = FMath::Min(Best, R.LatencyMs);
			}
			if (bSkip)
				break;
			UE_LOG(LogManipleInference, Display, TEXT("[batch] %-6s rows %4d: round-trip best %.2f ms avg %.2f ms (%.1f us/agent)"),
				Channel, Rows, Best, Sum / Iters, Sum / Iters * 1000.0 / Rows);
		}
	}
	return true;
}

#endif
