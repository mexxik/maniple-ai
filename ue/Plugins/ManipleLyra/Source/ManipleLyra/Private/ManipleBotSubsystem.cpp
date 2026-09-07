#include "ManipleBotSubsystem.h"
#include "ManipleLyra.h"
#include "ManipleLyraSchema.h"
#include "ManipleAgentComponent.h"
#include "ManipleTritonClient.h"
#include "ManipleBatchInferer.h"
#include "Player/LyraPlayerBotController.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/GameStateBase.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "HAL/PlatformTime.h"
#include "UObject/UObjectGlobals.h"

using namespace ManipleLyra;

bool UManipleBotSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	if (!Super::ShouldCreateSubsystem(Outer))
		return false;
	const UWorld* World = Cast<UWorld>(Outer);
	if (!World || !(World->WorldType == EWorldType::Game || World->WorldType == EWorldType::PIE))
		return false;
	return FManipleBotConfig::FromCommandLine().IsActive();
}

void UManipleBotSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	Config = FManipleBotConfig::FromCommandLine();
	if (Config.Brain == EManipleBrain::Triton)
	{
		Client = MakeShared<FManipleTritonClient>(Config.TritonUrl, 2.f);
		Batch = MakeShared<FManipleBatchInferer>(Client, Config.Model, TEXT("obs"), ObsDim, TEXT("action"), ActDim);
	}
	UE_LOG(LogManipleLyra, Display, TEXT("bot subsystem active: %s"), *Config.ToString());
}

void UManipleBotSubsystem::Deinitialize()
{
	Batch.Reset();
	Client.Reset();
	Super::Deinitialize();
}

void UManipleBotSubsystem::Tick(float DeltaTime)
{
	UWorld* World = GetWorld();
	if (!World || !World->HasBegunPlay() || World->GetNetMode() == NM_Client)
		return;

	++WinFrames;
	WinFrameSec += DeltaTime;

	ScanTimer += DeltaTime;
	if (ScanTimer >= 0.5f)
	{
		ScanTimer = 0.f;
		ScanForBots();
	}

	DecisionTimer += DeltaTime;
	const float Period = 1.f / Config.DecisionHz;
	if (DecisionTimer >= Period)
	{
		DecisionTimer = FMath::Fmod(DecisionTimer, Period);
		Decide();
	}

	StatsTimer += DeltaTime;
	if (StatsTimer >= 5.f)
	{
		StatsTimer = 0.f;
		LogStats();
	}
}

void UManipleBotSubsystem::SpawnExtraBots()
{
	// ULyraBotCreationComponent is not exported from LyraGame: reach it by reflection.
	AGameStateBase* GS = GetWorld()->GetGameState();
	UClass* BotClass = FindObject<UClass>(nullptr, TEXT("/Script/LyraGame.LyraBotCreationComponent"));
	UActorComponent* Comp = (GS && BotClass) ? GS->GetComponentByClass(BotClass) : nullptr;
	UFunction* Fn = Comp ? Comp->FindFunction(TEXT("SpawnOneBot")) : nullptr;
	if (!Fn)
	{
		UE_LOG(LogManipleLyra, Warning, TEXT("cannot spawn extra bots: LyraBotCreationComponent/SpawnOneBot not found"));
		return;
	}
	for (int32 i = 0; i < Config.SpawnBots; ++i)
		Comp->ProcessEvent(Fn, nullptr);
	UE_LOG(LogManipleLyra, Display, TEXT("spawned %d extra bots"), Config.SpawnBots);
}

void UManipleBotSubsystem::ScanForBots()
{
	OwnedControllers.RemoveAll([](const TWeakObjectPtr<AAIController>& C) { return !C.IsValid(); });

	int32 BotControllers = 0;
	for (TActorIterator<ALyraPlayerBotController> It(GetWorld()); It; ++It)
	{
		++BotControllers;
		AAIController* Ctrl = *It;
		APawn* Pawn = Ctrl->GetPawn();
		if (!Pawn || Pawn->FindComponentByClass<UManipleAgentComponent>())
			continue;

		if (!OwnedControllers.Contains(Ctrl))
		{
			if (Config.MaxBots >= 0 && OwnedControllers.Num() >= Config.MaxBots)
				continue;
			OwnedControllers.Add(Ctrl);
		}
		UManipleAgentComponent* Agent = NewObject<UManipleAgentComponent>(Pawn, TEXT("ManipleAgent"));
		Agent->Init(NextAgentId++);
		Agent->RegisterComponent();
	}

	// Lyra's own bots exist -> the experience is up; add ours once.
	if (!bSpawnedExtra && Config.SpawnBots > 0 && BotControllers > 0)
	{
		bSpawnedExtra = true;
		SpawnExtraBots();
	}

	Agents.Reset();
	for (const TWeakObjectPtr<AAIController>& C : OwnedControllers)
	{
		APawn* Pawn = C.IsValid() ? C->GetPawn() : nullptr;
		if (UManipleAgentComponent* A = Pawn ? Pawn->FindComponentByClass<UManipleAgentComponent>() : nullptr)
			Agents.Add(A);
	}
}

void UManipleBotSubsystem::OnTickResponse(int32 TickId, bool bOk, double LatencyMs)
{
	++WinRequests;
	if (!bOk)
		++WinFailures;
	FTickTrack* T = OpenTicks.Find(TickId);
	if (!T)
		return;
	if (++T->Received >= T->Expected)
	{
		WinTickLatencyMs.Add((FPlatformTime::Seconds() - T->StartSec) * 1000.0);
		++WinTicks;
		OpenTicks.Remove(TickId);
	}
}

void UManipleBotSubsystem::Decide()
{
	TArray<UManipleAgentComponent*> Ready;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		if (W.IsValid() && W->IsReady())
			Ready.Add(W.Get());
	}
	if (Ready.Num() == 0)
		return;

	if (Config.Brain == EManipleBrain::Random)
	{
		for (UManipleAgentComponent* A : Ready)
			A->SetRandomAction();
		WinTickLatencyMs.Add(0.0);
		++WinTicks;
		WinRequests += Ready.Num();
		return;
	}

	const int32 TickId = NextTickId++;
	FTickTrack Track;
	Track.StartSec = FPlatformTime::Seconds();
	TArray<float> Obs;
	TWeakObjectPtr<UManipleBotSubsystem> Self(this);

	if (Config.bBatch)
	{
		for (UManipleAgentComponent* A : Ready)
		{
			A->BuildObservation(Obs);
			TWeakObjectPtr<UManipleAgentComponent> WA(A);
			Batch->Submit(Obs,
				[WA](bool bOk, TConstArrayView<float> Out)
				{
					if (bOk && WA.IsValid())
						WA->SetAction(Out);
				});
		}
		Track.Expected = 1;
		OpenTicks.Add(TickId, Track);
		Batch->Flush(
			[Self, TickId](const FManipleInferResult& R, int32 Rows)
			{
				if (Self.IsValid())
					Self->OnTickResponse(TickId, R.bSuccess, R.LatencyMs);
			});
		return;
	}

	// one request per agent; skip agents whose previous request is still in flight
	for (UManipleAgentComponent* A : Ready)
	{
		if (A->bInferPending)
			continue;
		A->BuildObservation(Obs);
		const int64 Shape[2] = {1, ObsDim};
		A->bInferPending = true;
		++Track.Expected;
		TWeakObjectPtr<UManipleAgentComponent> WA(A);
		Client->Infer(Config.Model, {FManipleTensor::MakeFloat(TEXT("obs"), Shape, Obs)},
			FManipleInferComplete::CreateLambda(
				[Self, WA, TickId](const FManipleInferResult& R)
				{
					const FManipleTensor* Out = R.bSuccess ? R.FindOutput(TEXT("action")) : nullptr;
					if (WA.IsValid())
					{
						WA->bInferPending = false;
						if (Out)
							WA->SetAction(Out->AsFloats());
					}
					if (Self.IsValid())
						Self->OnTickResponse(TickId, Out != nullptr, R.LatencyMs);
				}),
			{TEXT("action")});
	}
	if (Track.Expected > 0)
		OpenTicks.Add(TickId, Track);
}

void UManipleBotSubsystem::LogStats()
{
	int32 Alive = 0;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
		if (W.IsValid() && W->IsReady())
			++Alive;

	double Avg = 0, P95 = 0, Max = 0;
	if (WinTickLatencyMs.Num() > 0)
	{
		WinTickLatencyMs.Sort();
		for (double L : WinTickLatencyMs)
			Avg += L;
		Avg /= WinTickLatencyMs.Num();
		P95 = WinTickLatencyMs[FMath::Min(WinTickLatencyMs.Num() - 1, (int32)(WinTickLatencyMs.Num() * 0.95))];
		Max = WinTickLatencyMs.Last();
	}
	const TCHAR* Mode = Config.Brain == EManipleBrain::Random ? TEXT("random") : Config.bBatch ? TEXT("batch") : TEXT("peragent");
	const double Sec = FMath::Max(WinFrameSec, 1e-6);
	UE_LOG(LogManipleLyra, Display,
		TEXT("bench: mode=%s agents=%d alive=%d hz=%.0f ticks=%d lat_avg=%.2f lat_p95=%.2f lat_max=%.2f frame_avg=%.2f req_s=%.1f "
			 "failures=%d open=%d"),
		Mode, Agents.Num(), Alive, Config.DecisionHz, WinTicks, Avg, P95, Max, WinFrames > 0 ? WinFrameSec / WinFrames * 1000.0 : 0.0,
		WinRequests / Sec, WinFailures, OpenTicks.Num());

	WinTicks = WinRequests = WinFailures = WinFrames = 0;
	WinFrameSec = 0.0;
	WinTickLatencyMs.Reset();
	// drop ticks that never completed (timeouts) so they don't accumulate
	for (auto It = OpenTicks.CreateIterator(); It; ++It)
		if (FPlatformTime::Seconds() - It->Value.StartSec > 5.0)
			It.RemoveCurrent();
}
