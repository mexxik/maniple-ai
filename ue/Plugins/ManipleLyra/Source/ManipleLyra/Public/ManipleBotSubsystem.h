#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "ManipleBotConfig.h"
#include "ManipleBotSubsystem.generated.h"

class FManipleTritonClient;
class FManipleBatchInferer;
class UManipleAgentComponent;
class AAIController;

/**
 * Created only when -ManipleBrain is set. Server side: finds Lyra bot controllers, attaches
 * UManipleAgentComponent to their pawns (also after respawn), optionally spawns extra bots,
 * runs the decision loop at DecisionHz (random / per-agent requests / one batched request),
 * and logs a grep-friendly "bench:" line every 5 s.
 */
UCLASS()
class MANIPLELYRA_API UManipleBotSubsystem : public UTickableWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override { RETURN_QUICK_DECLARE_CYCLE_STAT(UManipleBotSubsystem, STATGROUP_Tickables); }
	virtual bool IsTickable() const override { return Config.IsActive(); }

	const FManipleBotConfig& GetConfig() const { return Config; }

private:
	void ScanForBots();
	void SpawnExtraBots();
	void Decide();
	void OnTickResponse(int32 TickId, bool bOk, double LatencyMs);
	void LogStats();

	FManipleBotConfig Config;
	TSharedPtr<FManipleTritonClient> Client;
	TSharedPtr<FManipleBatchInferer> Batch;
	TArray<TWeakObjectPtr<AAIController>> OwnedControllers;
	TArray<TWeakObjectPtr<UManipleAgentComponent>> Agents;   // refreshed each scan
	int32 NextAgentId = 0;
	bool bSpawnedExtra = false;

	float ScanTimer = 0.f, DecisionTimer = 0.f, StatsTimer = 0.f;

	// per decision-tick bookkeeping: latency = tick start -> last response of that tick
	struct FTickTrack { int32 Expected = 0; int32 Received = 0; double StartSec = 0.0; };
	TMap<int32, FTickTrack> OpenTicks;
	int32 NextTickId = 0;

	// stats window
	int32 WinTicks = 0, WinRequests = 0, WinFailures = 0, WinFrames = 0;
	double WinFrameSec = 0.0;
	TArray<double> WinTickLatencyMs;
};
