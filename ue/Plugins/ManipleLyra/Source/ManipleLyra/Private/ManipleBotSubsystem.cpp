#include "ManipleBotSubsystem.h"
#include "ManipleLyra.h"
#include "ManipleAgentComponent.h"
#include "ManipleTritonClient.h"
#include "Player/LyraPlayerBotController.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "EngineUtils.h"
#include "Engine/World.h"

bool UManipleBotSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	if (!Super::ShouldCreateSubsystem(Outer)) return false;
	const UWorld* World = Cast<UWorld>(Outer);
	if (!World || !(World->WorldType == EWorldType::Game || World->WorldType == EWorldType::PIE)) return false;
	return FManipleBotConfig::FromCommandLine().IsActive();
}

void UManipleBotSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	Config = FManipleBotConfig::FromCommandLine();
	if (Config.Brain == EManipleBrain::Triton)
	{
		Client = MakeShared<FManipleTritonClient>(Config.TritonUrl, 2.f);
	}
	UE_LOG(LogManipleLyra, Display, TEXT("bot subsystem active: %s"), *Config.ToString());
}

void UManipleBotSubsystem::Deinitialize()
{
	Client.Reset();
	Super::Deinitialize();
}

void UManipleBotSubsystem::Tick(float DeltaTime)
{
	UWorld* World = GetWorld();
	if (!World || !World->HasBegunPlay() || World->GetNetMode() == NM_Client) return;

	ScanTimer += DeltaTime;
	if (ScanTimer >= 0.5f) { ScanTimer = 0.f; ScanForBots(); }
	StatsTimer += DeltaTime;
	if (StatsTimer >= 5.f) { StatsTimer = 0.f; LogStats(); }
}

void UManipleBotSubsystem::ScanForBots()
{
	OwnedControllers.RemoveAll([](const TWeakObjectPtr<AAIController>& C) { return !C.IsValid(); });

	for (TActorIterator<ALyraPlayerBotController> It(GetWorld()); It; ++It)
	{
		AAIController* Ctrl = *It;
		APawn* Pawn = Ctrl->GetPawn();
		if (!Pawn || Pawn->FindComponentByClass<UManipleAgentComponent>()) continue;

		const bool bOwned = OwnedControllers.Contains(Ctrl);
		if (!bOwned)
		{
			if (Config.MaxBots >= 0 && OwnedControllers.Num() >= Config.MaxBots) continue;
			OwnedControllers.Add(Ctrl);
		}

		UManipleAgentComponent* Agent = NewObject<UManipleAgentComponent>(Pawn, TEXT("ManipleAgent"));
		Agent->Init(Config, Client, NextAgentId++);
		Agent->RegisterComponent();
	}
}

void UManipleBotSubsystem::LogStats()
{
	int32 Agents = 0, Decisions = 0, Failures = 0;
	double LatencySum = 0.0;
	for (const TWeakObjectPtr<AAIController>& C : OwnedControllers)
	{
		const APawn* Pawn = C.IsValid() ? C->GetPawn() : nullptr;
		const UManipleAgentComponent* A = Pawn ? Pawn->FindComponentByClass<UManipleAgentComponent>() : nullptr;
		if (!A) continue;
		++Agents; Decisions += A->Decisions; Failures += A->Failures; LatencySum += A->LatencySumMs;
	}
	const int32 Ok = Decisions - Failures;
	UE_LOG(LogManipleLyra, Display, TEXT("stats: %d agents, %d decisions, %d failures, avg latency %.2f ms"),
		Agents, Decisions, Failures, Ok > 0 ? LatencySum / Ok : 0.0);
}
