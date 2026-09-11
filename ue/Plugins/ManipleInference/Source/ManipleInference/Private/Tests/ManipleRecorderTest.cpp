// Recorder round trip on disk: write two batches with extra columns, patch the header, read everything back.
// No Triton needed.
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "Dom/JsonObject.h"
#include "ManipleAgentClient.h"
#include "ManipleRecorder.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FManipleRecorderTest, "Maniple.Recorder.RoundTrip", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FManipleRecorderTest::RunTest(const FString& Parameters)
{
	const FString Dir = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Maniple"), TEXT("test-recording"));
	IFileManager::Get().DeleteDirectory(*Dir, false, true);

	// ---- header helpers
	{
		const TArray<uint8> Head = ManipleNpy::Header(TEXT("<f4"), 12345, {84, 84, 4});
		TestEqual(TEXT("header size"), Head.Num(), ManipleNpy::HeaderSize);
		FString Descr;
		TArray<int64> Shape;
		TestEqual(TEXT("header parses"), ManipleNpy::ParseHeader(Head, Descr, Shape), ManipleNpy::HeaderSize);
		TestEqual(TEXT("descr"), Descr, FString(TEXT("<f4")));
		TestEqual(TEXT("shape"), Shape, TArray<int64>({12345, 84, 84, 4}));
		const TArray<uint8> Flat = ManipleNpy::Header(TEXT("|b1"), 7, {});
		TestEqual(TEXT("flat parses"), ManipleNpy::ParseHeader(Flat, Descr, Shape), ManipleNpy::HeaderSize);
		TestEqual(TEXT("flat shape"), Shape, TArray<int64>({7}));
	}

	// ---- write
	constexpr int32 ObsDim = 3, ActDim = 2;
	{
		TSharedPtr<FJsonObject> Meta = MakeShared<FJsonObject>();
		Meta->SetStringField(TEXT("game"), TEXT("test"));
		FManipleRecorder Rec;
		Rec.FlushRows = 3;
		if (!TestTrue(TEXT("open"), Rec.Open(Dir, Meta)))
			return false;

		FManipleTransitionBatch B(ObsDim, ActDim);
		for (int32 Tick = 0; Tick < 2; ++Tick)
		{
			B.Reset();
			for (int32 Agent = 0; Agent < 2; ++Agent)
			{
				const float Row = (float)(Tick * 2 + Agent);
				const float Obs[ObsDim] = {Row, Row + 0.5f, -Row};
				const float Act[ActDim] = {1.f, -1.f};
				B.Add(Obs, Act, Row * 0.1f, Tick == 1 && Agent == 0, Agent, 0, 42, -0.3f);
			}
			const int64 Flat[1] = {B.Num()};
			const int64 Pose[2] = {B.Num(), 2};
			const float Time[2] = {(float)Tick, (float)Tick};
			const uint8 Kind[2] = {1, 1};
			const float PoseData[4] = {10.f, 20.f, 30.f, 40.f};
			TArray<FManipleTensor> Extra;
			Extra.Add(FManipleTensor::MakeFloat(TEXT("time"), Flat, Time));
			Extra.Add(FManipleTensor::MakeUInt8(TEXT("kind"), Flat, Kind));
			Extra.Add(FManipleTensor::MakeFloat(TEXT("pose"), Pose, PoseData));
			Rec.Write(B, Extra);
		}
		TestEqual(TEXT("rows counted"), Rec.NumRows(), (int64)4);
		Rec.Close();
	}

	// ---- read
	{
		FManipleRecording R;
		FString Error;
		if (!TestTrue(TEXT("load: ") + Error, R.Load(Dir, Error)))
			return false;
		TestEqual(TEXT("rows"), R.Rows, (int64)4);
		TestEqual(TEXT("columns"), R.Columns.Num(), 11); // obs action reward done logp policy_version agent_id episode_id time kind pose
		const FManipleTensor* Obs = R.Find(TEXT("obs"));
		if (TestNotNull(TEXT("obs column"), Obs))
		{
			TestEqual(TEXT("obs shape"), Obs->Shape, TArray<int64>({4, ObsDim}));
			TestEqual(TEXT("obs value"), Obs->AsFloats()[3 * ObsDim + 1], 3.5f);
		}
		const FManipleTensor* Done = R.Find(TEXT("done"));
		if (TestNotNull(TEXT("done column"), Done))
		{
			TestEqual(TEXT("done datatype"), Done->Datatype, FString(TEXT("BOOL")));
			TestEqual(TEXT("done row 2"), (int32)Done->Data[2], 1);
			TestEqual(TEXT("done row 3"), (int32)Done->Data[3], 0);
		}
		const FManipleTensor* Pose = R.Find(TEXT("pose"));
		if (TestNotNull(TEXT("pose column"), Pose))
			TestEqual(TEXT("pose shape"), Pose->Shape, TArray<int64>({4, 2}));
		const FManipleTensor* Version = R.Find(TEXT("policy_version"));
		if (TestNotNull(TEXT("version column"), Version))
			TestEqual(TEXT("version value"), Version->AsInt64s()[3], (int64)42);
		TestEqual(TEXT("meta rows"), (int64)R.Meta->GetNumberField(TEXT("rows")), (int64)4);
		TestTrue(TEXT("meta closed"), R.Meta->GetBoolField(TEXT("closed")));
		TestEqual(TEXT("meta game"), R.Meta->GetStringField(TEXT("game")), FString(TEXT("test")));
	}

	IFileManager::Get().DeleteDirectory(*Dir, false, true);
	return true;
}

#endif
